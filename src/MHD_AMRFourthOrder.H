#pragma once

namespace HelioCubed
{
inline void CubedSphereAMR::prepareFourthOrder(GridLevel& grid,const Level& state,
    double time,GridLevel* parent,bool boundaryDataOnly)
{
    PR_TIMERS("HelioCubed::AMR::prepareFourthOrder");
    OP::GeometryScope geometry(grid.geometry);
    grid.timeInterpolated=false;grid.preparedState=&state;grid.preparedTime=time;
    amrSetValLocal(grid.donor,0);
    if(parent)
    {
        if(!grid.coarseSource4)
        {
            const auto coarseLayout=grid.layout.coarsen(grid.ratio);
            Point ghost;
            for(int d=0;d<DIM;++d)
                ghost[d]=(grid.donor.ghost()[0][d]+grid.ratio[d]-1)/grid.ratio[d]+(grid.ratio[d]>1 ? 2 : 0);
            grid.coarseSource4=std::make_unique<Level>(coarseLayout,amrPatchGhosts(ghost));
            amrSetValLocal(*grid.coarseSource4,std::numeric_limits<double>::quiet_NaN());
            grid.coarseCopy4=std::make_unique<AMRTransferCopier<NUMCOMPS,true>>(
                *parent->cartesian,*grid.coarseSource4,true);
        }
        auto& source=*grid.coarseSource4;
        grid.coarseCopy4->copy(*parent->cartesian,source);
        if(grid.fourthOrderGhosts.empty())
            for(auto index : grid.layout) grid.fourthOrderGhosts.emplace_back(grid.donor[index].box(),grid.ratio);
        unsigned local=0;
        for(auto index : grid.layout)
        {
            auto interpolated=amrInterpolateFourthOrder<NUMCOMPS>(source[index],grid.fourthOrderGhosts[local++]);
            for(auto p : grid.donor[index].box())
            {
                if(grid.layout[index].containsPoint(p)) continue;
                bool valid=true;
                for(int c=0;c<NUMCOMPS;++c) valid&=std::isfinite(interpolated(p,c));
                if(valid) for(int c=0;c<NUMCOMPS;++c) grid.donor[index](p,c)=interpolated(p,c);
            }
        }
    }

    // Recover <U> from <JU>, including the average-product correction. Only
    // same-block JU can be exchanged: its Jacobian belongs to that block.
    if(!grid.mapped4) grid.mapped4=std::make_unique<Level>(grid.layout,amrPatchGhosts(Point::Ones(2)));
    auto& mapped=*grid.mapped4;
    for(auto index : grid.layout)
    {
        auto& out=mapped[index];out.setVal(0);
        if(parent)
            for(auto p : out.box())
            {
                if(grid.layout[index].containsPoint(p)) continue;
                for(int c=0;c<NUMCOMPS;++c)
                {
                    double value=grid.jacobian[index](p)*grid.donor[index](p,c);
                    for(int d=0;d<DIM;++d)
                    {
                        const Point e=Point::Basis(d);
                        value+=(grid.jacobian[index](p+e)-grid.jacobian[index](p-e))
                            *(grid.donor[index](p+e,c)-grid.donor[index](p-e,c))/48;
                    }
                    out(p,c)=value;
                }
            }
        state[index].copyTo(out,grid.layout[index],Point::Zeros());
    }
    mapped.exchange0();
    for(auto index : grid.layout)
    {
        const Box valid=grid.layout[index];
        const Box block=grid.layout.domain().getBlock(grid.layout.block(index)).box();
        const Box halo=valid.grow(2)&block;
        State quotient(halo);
        for(auto p : halo) for(int c=0;c<NUMCOMPS;++c)
            quotient(p,c)=mapped[index](p,c)/grid.jacobian[index](p);
        State correction(valid,0.0);
        for(int d=0;d<DIM;++d)
        {
            auto derivative=CubedSphereShell::deriv3case(quotient,valid,block,d);
            const auto& j=grid.jacobian[index];
            for(auto p : valid)
            {
                const Point e=Point::Basis(d);
                const double dj=0.5*(j(p+e)-j(p-e));
                for(int c=0;c<NUMCOMPS;++c) correction(p,c)+=dj*derivative(p,c);
            }
        }
        for(auto p : valid) for(int c=0;c<NUMCOMPS;++c)
            grid.donor[index](p,c)=quotient(p,c)-correction(p,c)/(12*grid.jacobian[index](p));
    }
    grid.donor.exchange();
    for(auto index : grid.layout) grid.donor[index].copyTo((*grid.cartesian)[index]);
    grid.interp.apply(*grid.cartesian,grid.donor);
    grid.cartesian->exchange0();
    if(!grid.nativeConversion)
        grid.nativeConversion=std::make_unique<NativeConversionCache<double>>(grid.layout,grid.primitive.ghost()[0],true);
    for(auto index : grid.layout)
    {
        const auto block=grid.layout.block(index);
        const auto convert=[&](State& output)
        {cachedNativeConversion(output,(*grid.cartesian)[index],grid.layout,
            grid.primitive.ghost()[0],index.global(),block,grid.op[index].dx()[1],grid.nativeConversion.get());};
        if(!boundaryDataOnly) convert(grid.primitive[index]);
        else
        {
            // Children consume Cartesian averages. Only physical radial BCs
            // need the parent's native basis; convert their strips and the
            // one owned layer needed by the outflow/copy boundary condition.
            const Box physical=grid.layout.domain().getBlock(block).box();
            const Box storage=grid.primitive[index].box();
            for(int side=0;side<2;++side)
            {
                Point low=storage.low(),high=storage.high();
                if(side==0) {if(low[0]>=physical.low()[0]) continue;high[0]=physical.low()[0];}
                else {if(high[0]<=physical.high()[0]) continue;low[0]=physical.high()[0];}
                State strip(Box(low,high));convert(strip);strip.copyTo(grid.primitive[index]);
            }
        }
    }
    OP::fillRadialBoundaries(grid.primitive,grid.volume,time);
    if(!boundaryDataOnly && ParseInputs::get_seam_pressure_correction())
        for(auto index : grid.layout)
        {
            const auto block=grid.layout.block(index);
            const Box physical=grid.layout.domain().getBlock(block).box();
            const Box storage=grid.primitive[index].box();
            if(storage.low()[1]>=physical.low()[1] && storage.high()[1]<=physical.high()[1]
                && storage.low()[2]>=physical.low()[2] && storage.high()[2]<=physical.high()[2]) continue;
            const State cartPoint=(1.0*Shift::Zeros()-(1.0/24)*Stencil<double>::Laplacian())((*grid.cartesian)[index]);
            const bool magnetic=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
            const double b0=testProblemBackgroundBx(ParseInputs::get_init_condition_type());
            const double gamma=ParseInputs::get_gamma();
            for(int pass=0;pass<3;++pass)
            {
                State w,bar;grid.op[index].sphToPrim(w,bar,grid.primitive[index],block);
                for(auto p : w.box())
                {
                    if(physical.containsPoint(p) || p[0]<physical.low()[0] || p[0]>physical.high()[0]) continue;
                    LocalFaceState<double> u,desired;
                    for(int c=0;c<NUMCOMPS;++c) u(c)=cartPoint(p,c);
                    cartesianPrimitiveFromConserved(desired,u,1.0,gamma,magnetic,b0);
                    if(std::isfinite(desired(iP)) && desired(iP)>0 && std::isfinite(w(p,iP)))
                        grid.primitive[index](p,iE)+=(desired(iP)-w(p,iP))/(gamma-1);
                }
            }
        }

    // Retain the physical radial boundary extension for child interpolation.
    // Cartesian averages use the same fourth-order matrix product as the
    // inverse basis change; pointwise rotation would lose two orders.
    for(auto index : grid.layout)
    {
        const auto block=grid.layout.block(index);
        const Box physical=grid.layout.domain().getBlock(block).box();
        const Box storage=grid.primitive[index].box().grow(Point(0,-1,-1));
        for(int side=0;side<2;++side)
        {
        Point low=storage.low(),high=storage.high();
        if(side==0) high[0]=std::min(high[0],physical.low()[0]-1);
        else low[0]=std::max(low[0],physical.high()[0]+1);
        const Box box(low,high);
        if(box.empty()) continue;
        const auto& geometry=grid.nativeConversion->patches.at(index.global());
        std::array<State,2> du={Stencil<double>::Derivative(1,1)(grid.primitive[index],box),
            Stencil<double>::Derivative(1,2)(grid.primitive[index],box)};
        for(auto p : box)
        {
            if(p[0]>=physical.low()[0] && p[0]<=physical.high()[0]) continue;
            const auto& u=grid.primitive[index];auto& cart=(*grid.cartesian)[index];
            const auto& coefficient=geometry.at(p[1],p[2]);
            const auto& matrix=geometry.matrixAt(p[1],p[2]);
            cart(p,iRHO)=u(p,iRHO);cart(p,iE)=u(p,iE);
            for(int start : {iMOMX,iBX}) for(int row=0;row<DIM;++row)
            {
                double value=0;
                for(int col=0;col<DIM;++col)
                {
                    value+=matrix[3*row+col]*u(p,start+col);
                    for(int d=0;d<2;++d) value+=coefficient[9+9*d+3*row+col]*du[d](p,start+col)/12;
                }
                cart(p,start+row)=value;
            }
        }
        }
    }
}

inline State CubedSphereAMR::fourthOrderPrimitives(const GridLevel& grid,const MBIndex& index) const
{
    const Box box=grid.layout[index].grow(1);
    State point=(1.0*Shift::Zeros()-(1.0/24)*Stencil<double>::Laplacian())((*grid.cartesian)[index],box);
    State result(box);
    const bool magnetic=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
    forallInPlace([] PROTO_LAMBDA(Var<double,NUMCOMPS,HOST>& out,const Var<double,NUMCOMPS,HOST>& in,
        double gamma,bool magnetic,double b0)
    {cartesianPrimitiveFromConserved(out,in,1.0,gamma,magnetic,b0);},box,result,point,
        ParseInputs::get_gamma(),magnetic,testProblemBackgroundBx(ParseInputs::get_init_condition_type()));
    return result;
}

inline void CubedSphereAMR::prolongFourthOrder(GridLevel& target,GridLevel& parent,Point ratio)
{
    PR_TIMERS("HelioCubed::AMR::prolongFourthOrder");
    using Mapped=MBLevelBoxData<double,NUMCOMPS+1,HOST>;
    Mapped mapped(parent.layout,amrPatchGhosts(Point::Ones(4)));
    for(auto index : parent.layout)
    {
        const auto& cart=(*parent.cartesian)[index];const auto& j=parent.jacobian[index];
        for(auto p : mapped[index].box())
        {
            mapped[index](p,NUMCOMPS)=j(p);
            for(int c=0;c<NUMCOMPS;++c)
            {
                double value=j(p)*cart(p,c);
                for(int d=0;d<DIM;++d)
                {
                    const Point e=Point::Basis(d);
                    value+=(j(p+e)-j(p-e))*(cart(p+e,c)-cart(p-e,c))/48;
                }
                mapped[index](p,c)=parent.layout[index].containsPoint(p) ? parent.state[index](p,c) : value;
            }
        }
    }
    const auto coarseLayout=target.layout.coarsen(ratio);
    Mapped source(coarseLayout,amrPatchGhosts(Point::Ones(2)));
    amrSetValLocal(source,std::numeric_limits<double>::quiet_NaN());mapped.copyToFullUnique(source);
    const bool magnetic=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
    const double b0=magnetic ? testProblemBackgroundBx(ParseInputs::get_init_condition_type()) : 0;
    const double gamma=ParseInputs::get_gamma();
    int invalid=0;
    for(auto index : target.layout)
    {
      auto interpolated=amrInterpolateFourthOrder<NUMCOMPS+1>(source[index],target.layout[index],ratio);
      for(auto q : coarseLayout[index])
      {
        const Box children=Box(q,q).refine(ratio);
        double sumJ=0;std::array<double,NUMCOMPS> mean{},center{};
        bool available=true;
        for(auto p : children)
        {
            const double interpolatedJ=interpolated(p,NUMCOMPS);
            if(!std::isfinite(interpolatedJ) || !(interpolatedJ>0)) {available=false;break;}
            const double j=target.jacobian[index](p);sumJ+=j;
            for(int c=0;c<NUMCOMPS;++c)
            {
                const double physical=interpolated(p,c)/interpolatedJ;
                if(!std::isfinite(physical)) available=false;
                target.state[index](p,c)=physical;mean[c]+=j*physical;
            }
        }
        if(!available) {invalid=1;continue;}
        const double cells=double(ratio[0])*ratio[1]*ratio[2];
        for(int c=0;c<NUMCOMPS;++c) {mean[c]/=sumJ;center[c]=cells*source[index](q,c)/sumJ;}
        auto admissible=[&](double theta)
        {
            for(auto p : children)
            {
                double u[NUMCOMPS];
                for(int c=0;c<NUMCOMPS;++c) u[c]=center[c]+theta*(target.state[index](p,c)-mean[c]);
                if(theta==0)
                {
                    if(!amrAdmissibleParent(u,gamma,magnetic,b0)) return false;
                    continue;
                }
                if(!(u[iRHO]>=RHO00)) return false;
                double thermal=u[iE]+(magnetic ? c_TEST_P0/(gamma-1) : 0);
                for(int d=0;d<DIM;++d) thermal-=0.5*u[iMOMX+d]*u[iMOMX+d]/u[iRHO]+u[iBX+d]*u[iBX+d]/(8*M_PI);
                thermal-=b0*u[iBX]/(4*M_PI);
                if(!std::isfinite(thermal) || thermal<0) return false;
            }
            return true;
        };
        double theta=1;
        if(!admissible(theta))
        {
            if(!admissible(0)) {invalid=1;continue;}
            double lo=0,hi=1;
            for(int n=0;n<48;++n) {const double trial=0.5*(lo+hi);if(admissible(trial)) lo=trial;else hi=trial;}
            theta=lo;
        }
        for(auto p : children) for(int c=0;c<NUMCOMPS;++c)
            target.state[index](p,c)=target.jacobian[index](p)*(center[c]+theta*(target.state[index](p,c)-mean[c]));
      }
    }
#ifdef PR_MPI
    MPI_Allreduce(MPI_IN_PLACE,&invalid,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
    if(invalid) throw std::runtime_error("Invalid or unavailable parent data for fourth-order conservative refinement");
}
}
