#pragma once
#include "MHD_SecondOrderSeamPlan.H"

// RK stages keep Cartesian primitives for limited O2. Diagnostics and the
// viscosity path continue to use prepareState's native conserved contract.
template<typename T, typename MAP, MemType MEM>
inline void BoxOp_EulerCubedSphere<T,MAP,MEM>::prepareSecondOrderPrimitives(
    MBLevelBoxData<T,NUMCOMPS,MEM>& state, MBInterpOp& interp,
    MBLevelBoxData<T,1,MEM>& volume, T time)
{
    PR_TIMERS("HelioCubed::prepareSecondOrderPrimitives");
    const auto& layout=state.layout();
    const auto sourceGhost=state.ghost()+Point(0,2,2);
    struct DonorCache
    {
        LayoutT layout;
        Array<Point,DIM+1> ghost;
        MBLevelBoxData<T,NUMCOMPS,MEM> input;
        MBLevelBoxData<T,1,MEM> inverseJ;
        std::unique_ptr<HelioCubed::SecondOrderSeamPlan<T,NUMCOMPS>> seam;
        DonorCache(const LayoutT& l,const Array<Point,DIM+1>& g,
            MBLevelBoxData<T,1,MEM>& volume)
            : layout(l),ghost(g),input(l,g),inverseJ(l,Point::Zeros())
        {
            const T h=T(1)/l.domain().getBlock(0).sizes()[1];
            for (auto index : l)
            {
                // Only physical radial ghosts remain zero. All donor values
                // inside the domain are overwritten or exchanged each stage.
                input[index].setVal(T(0));
                auto det=CubedSphereShell::detA<T,MEM>(l[index],h,0);
                forallInPlace([] PROTO_LAMBDA(Var<T,1,MEM>& out,
                    const Var<T,1,MEM>& vol,const Var<T,1,MEM>& det)
                {out(0)=T(1)/(vol(0)*det(0));},inverseJ[index],volume[index],det);
            }
        }
    };
    static std::unique_ptr<DonorCache> cache;
    if (!cache || &cache->layout.partition()!=&layout.partition()
        || cache->ghost!=sourceGhost)
        cache=std::make_unique<DonorCache>(layout,sourceGhost,volume);

    const bool magneticTest=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
    const T b0=testProblemBackgroundBx(ParseInputs::get_init_condition_type());
    const T gamma=ParseInputs::get_gamma();
    for (auto index : layout)
        forallInPlace([] PROTO_LAMBDA(Var<T,NUMCOMPS,MEM>& out,
            const Var<T,NUMCOMPS,MEM>& ju,const Var<T,1,MEM>& invJ,
            T gamma,bool magneticTest,T b0)
        {
            HelioCubed::cartesianPrimitiveFromConserved(
                out,ju,invJ(0),gamma,magneticTest,b0);
        },layout[index],cache->input[index],state[index],cache->inverseJ[index],
            gamma,magneticTest,b0);
    cache->input.exchange();
    // Layout and owners are identical. Copy the already exchanged patch
    // storage locally, avoiding a new level copier and two further exchanges.
    for (auto index : layout) cache->input[index].copyTo(state[index]);
    if constexpr (MEM == HOST)
    {
        if (!cache->seam || !cache->seam->matches(state,cache->input,interp))
            cache->seam=std::make_unique<HelioCubed::SecondOrderSeamPlan<T,NUMCOMPS>>(
                state,cache->input,interp);
        bool boundPrimitives=false;
        for(const auto& source:ParseInputs::get_CMEs())
            if(time>=source.insertionTime) boundPrimitives=true;
        cache->seam->apply(boundPrimitives);
    }
    else interp.apply(state,cache->input);
    for (auto index : layout)
        forallInPlace([] PROTO_LAMBDA(Var<T,NUMCOMPS,MEM>& w)
        {
            w(iRHO)=max(w(iRHO),T(RHO00));
            w(iP)=max(w(iP),T(0));
        },state[index]);
    fillRadialPrimitiveBoundaries(state,volume,time);
}

template<typename T, typename MAP, MemType MEM>
struct BoxOp_EulerCubedSphere<T,MAP,MEM>::SecondOrderBoundaryCache
{
    LayoutT layout;
    Point ghost;
    bool averaged;
    std::vector<State> strips;
    SecondOrderBoundaryCache(const LayoutT& l,Point g,bool averages)
        :layout(l),ghost(g),averaged(averages)
    {
        for (auto index : l)
        {
            strips.emplace_back();
            const Box storage=l[index].grow(g);
            const Box domain=l.getBlock(l.block(index)).domain().box();
            if (storage.low()[0]>=domain.low()[0]) continue;
            Point low=storage.low(),high=storage.high();
            high[0]=domain.low()[0]-1;
            if (!averages) low[0]=high[0];
            strips.back().defineForOverwrite(Box(low,high));
        }
    }
};

template<typename T, typename MAP, MemType MEM>
inline void BoxOp_EulerCubedSphere<T,MAP,MEM>::fillRadialPrimitiveBoundaries(
    MBLevelBoxData<T,NUMCOMPS,MEM>& state,
    MBLevelBoxData<T,1,MEM>& volume,T time)
{
    PR_TIMERS("HelioCubed::fillRadialPrimitiveBoundaries");
    const auto& layout=state.layout();
    auto& geometry=levelGeometry(layout);
    const bool fileBoundary=ParseInputs::get_init_condition_type()==0;
    const bool magneticTest=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
    const T b0=testProblemBackgroundBx(ParseInputs::get_init_condition_type());
    const T gamma=ParseInputs::get_gamma();
    auto& boundary=geometry.secondOrderBoundary;
    State unused;
    if (fileBoundary)
    {
        const bool averaged=ParseInputs::get_BC_conserved_averages()!=0;
        if (!boundary || &boundary->layout.partition()!=&layout.partition()
            || boundary->ghost!=state.ghost()[0] || boundary->averaged!=averaged)
            boundary=std::make_shared<SecondOrderBoundaryCache>(layout,state.ghost()[0],averaged);
        BC_global.BoxData_to_BC<T,MEM>(state,*geometry.map,time,true,&boundary->strips);
    }
    unsigned local=0;
    for (auto index : layout)
    {
        const unsigned patch=local++;
        auto& op=(*geometry.op)[index];
        const auto block=layout.block(index);
        const Box domain=layout.getBlock(block).domain().box();
        const Box storage=state[index].box();
        for (int side=0; side<2; ++side)
        {
            const bool low=side==0;
            if (low ? storage.low()[0]>=domain.low()[0]
                    : storage.high()[0]<=domain.high()[0]) continue;
            Point lo=storage.low(),hi=storage.high();
            // Retain one owned radial layer for outflow/copy boundaries.
            if (low) hi[0]=domain.low()[0];
            else lo[0]=domain.high()[0];
            const Box strip(lo,hi);
            ScopedHostStack<MEM> scratch;
            BoxData<T,DIM,MEM,DIM> temporaryBasis,temporaryInverse;
            const T *basisData,*inverseData;
            if constexpr (MEM == HOST)
            {
                auto& cached=op.m_boundaryGeometry[side];
                if (!cached || cached->box!=strip)
                {
                    PR_TIMERS("HelioCubed::boundaryGeometrySetup");
                    cached=std::make_shared<SecondOrderBoundaryGeometry>();
                    cached->box=strip;
                    temporaryBasis=HelioCubed::secondOrderCellBasis<T,MEM>(strip,op.dx()[1],block);
                    temporaryInverse=forallOverwrite<T,DIM,MEM,DIM>(f_matinv3by3,temporaryBasis);
                    const size_t count=strip.size()*DIM*DIM;
                    // Heap-owned vectors outlive the temporary HOST arena.
                    cached->basis.assign(temporaryBasis.data(),temporaryBasis.data()+count);
                    cached->inverse.assign(temporaryInverse.data(),temporaryInverse.data()+count);
                }
                basisData=cached->basis.data();inverseData=cached->inverse.data();
            }
            else
            {
                temporaryBasis=HelioCubed::secondOrderCellBasis<T,MEM>(strip,op.dx()[1],block);
                temporaryInverse=forallOverwrite<T,DIM,MEM,DIM>(f_matinv3by3,temporaryBasis);
                basisData=temporaryBasis.data();inverseData=temporaryInverse.data();
            }
            // Matrix components are stored with the row varying fastest.
            BoxData<T,DIM*DIM,MEM> basis(basisData,strip),inverse(inverseData,strip);
            State native;
            native.defineForOverwrite(strip);
            forallInPlace([] PROTO_LAMBDA(Var<T,NUMCOMPS,MEM>& u,
                const Var<T,NUMCOMPS,MEM>& w,const Var<T,DIM*DIM,MEM>& inv,
                T gamma,bool magneticTest,T b0)
            {
                u(iRHO)=w(iRHO);
                T kinetic=0,magnetic=0;
                for (int d=0; d<DIM; ++d)
                {
                    kinetic+=T(0.5)*w(iRHO)*w(iVX+d)*w(iVX+d);
                    magnetic+=w(iBX+d)*w(iBX+d)/T(8*M_PI);
                    u(iMOMX+d)=u(iBX+d)=0;
                    for (int j=0; j<DIM; ++j)
                    {
                        u(iMOMX+d)+=inv(d+DIM*j)*w(iRHO)*w(iVX+j);
                        u(iBX+d)+=inv(d+DIM*j)*w(iBX+j);
                    }
                }
                if (magneticTest) magnetic+=b0*w(iBX)/T(4*M_PI);
                u(iE)=(w(iP)-(magneticTest ? T(c_TEST_P0) : T(0)))
                    /(gamma-T(1))+kinetic+magnetic;
            },native,state[index],inverse,gamma,magneticTest,b0);
            op.PreStagePatch(native,fileBoundary && low ? boundary->strips[patch] : unused,
                volume[index],domain,time);
            // Write only physical radial ghosts; preserve all owned values
            // and the angular interpolation at interior radial coordinates.
            if (low) --hi[0]; else ++lo[0];
            forallInPlace([] PROTO_LAMBDA(Var<T,NUMCOMPS,MEM>& w,
                const Var<T,NUMCOMPS,MEM>& u,const Var<T,DIM*DIM,MEM>& A,
                T gamma,bool magneticTest,T b0)
            {
                w(iRHO)=u(iRHO);w(iE)=u(iE);
                for (int group=0; group<2; ++group)
                {
                    const int start=group==0 ? iMOMX : iBX;
                    for (int d=0; d<DIM; ++d)
                    {
                        w(start+d)=0;
                        for (int j=0; j<DIM; ++j) w(start+d)+=A(d+DIM*j)*u(start+j);
                    }
                }
                HelioCubed::cartesianPrimitiveFromConserved(
                    w,w,T(1),gamma,magneticTest,b0);
            },Box(lo,hi),state[index],native,basis,gamma,magneticTest,b0);
        }
    }
}
