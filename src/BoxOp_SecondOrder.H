#pragma once

namespace HelioCubed
{
// Shared by materialized device faces and the fused HOST face loop. Keep
// the limiter arithmetic and complete-state positivity fallback identical.
template<typename T,class Left,class Right,class MM,class M,class C,class P>
inline ACCEL_DECORATION void reconstructSecondOrderFace(Left& lo,Right& hi,
    const MM& mm,const M& m,const C& c,const P& p,int limiter)
{
    for (int k=0; k<NUMCOMPS; ++k)
    {
        T a=m(k)-mm(k), b=c(k)-m(k), d=p(k)-c(k);
        T sl=T(0.5)*(a+b), sr=T(0.5)*(b+d);
        if (limiter != 0)
        {
            sl = a*b <= 0 ? T(0) : (a>0 ? min(a,b) : max(a,b));
            sr = b*d <= 0 ? T(0) : (b>0 ? min(b,d) : max(b,d));
            // Monotonized central: minmod(2*dL, (dL+dR)/2, 2*dR).
            // Keep zero slope at extrema and on either side of a plateau.
            if (limiter == 3)
            {
                if (a*b > 0) sl=(a>0 ? T(1) : T(-1))*
                    min(T(0.5)*fabs(a+b),T(2)*min(fabs(a),fabs(b)));
                if (b*d > 0) sr=(b>0 ? T(1) : T(-1))*
                    min(T(0.5)*fabs(b+d),T(2)*min(fabs(b),fabs(d)));
            }
        }
        lo(k)=m(k)+T(0.5)*sl;
        hi(k)=c(k)-T(0.5)*sr;
    }
    // Fall back as a complete state if unlimited extrapolation fails.
    if (!(lo(iRHO)>=RHO00 && lo(iP)>=0))
        for (int k=0; k<NUMCOMPS; ++k) lo(k)=m(k);
    if (!(hi(iRHO)>=RHO00 && hi(iP)>=0))
        for (int k=0; k<NUMCOMPS; ++k) hi(k)=c(k);
}
}

// A panel seam is one physical face. Exchange the reconstructed state from
// each real adjacent cell, rather than solving against interpolated ghosts
// independently and averaging two different Riemann problems afterward.
template<typename T, typename MAP, MemType MEM>
struct BoxOp_EulerCubedSphere<T,MAP,MEM>::SecondOrderPanelStateCache
{
    LayoutT layout;
    MBBoundaryRegister<T,NUMCOMPS,MEM> exchange;
    std::vector<PanelStates> adjacent;

    explicit SecondOrderPanelStateCache(const LayoutT& l)
        :layout(l),exchange(l,1,Point::Zeros())
    {
        for(auto index:l)
        {
            adjacent.emplace_back();
            for(const auto& boundary:exchange.bounds(index))
            {
                if(boundary.localDir.codim()!=1) continue;
                const auto face=boundary.localFace();
                const unsigned slot=2*face.dir+int(face.side);
                adjacent.back()[slot].defineForOverwrite(
                    l[index].extrude(face.dir).face(face.dir,face.side));
            }
        }
    }
};

template<typename T, typename MAP, MemType MEM>
inline void BoxOp_EulerCubedSphere<T,MAP,MEM>::prepareSharedPanelStates(
    const MBLevelBoxData<T,NUMCOMPS,MEM>& primitive)
{
    PR_TIMERS("HelioCubed::sharedPanelStates");
    const auto& layout=primitive.layout();
    auto& cache=levelGeometry(layout).secondOrderPanelStates;
    if(!cache || &cache->layout.partition()!=&layout.partition())
        cache=std::make_shared<SecondOrderPanelStateCache>(layout);
    auto& exchange=cache->exchange;
    exchange.clear();
    const int limiter=ParseInputs::get_limiter();
    for(auto index:layout)
    {
        const auto& w=primitive[index];
        for(const auto& boundary:exchange.bounds(index))
        {
            if(boundary.localDir.codim()!=1) continue;
            const auto face=boundary.localFace();
            const Point step=Point::Basis(face.dir);
            const Box faces=layout[index].extrude(face.dir).face(face.dir,face.side);
            for(auto p:faces)
            {
                HelioCubed::LocalFaceState<T> left,right;
                HelioCubed::reconstructSecondOrderFace<T>(left,right,
                    w.var(p-2*step),w.var(p-step),w.var(p),w.var(p+step),limiter);
                for(unsigned c=0;c<NUMCOMPS;++c)
                    (*boundary.localData)(p,c)=face.side==Side::Lo ? right(c) : left(c);
            }
        }
    }
    exchange.exchange();
    unsigned patch=0;
    for(auto index:layout)
    {
        auto& incoming=cache->adjacent[patch++];
        for(const auto& boundary:exchange.bounds(index))
        {
            if(boundary.localDir.codim()!=1) continue;
            const auto face=boundary.localFace();
            const auto inverse=boundary.adjToLocal.inverse();
            const Point normal=Point::Basis(face.dir,face.side);
            const Point adjacentNormal=-inverse(normal);
            const int orientation=adjacentNormal.dot(adjacentNormal*adjacentNormal);
            // Use the same face-centering offset as the boundary register.
            // The exchanged Cartesian primitive components need no sign flip.
            const Point shift=orientation*(2*int(face.side)-1)==1
                ? Point::Basis(face.dir) : Point::Zeros();
            auto& states=incoming[2*face.dir+int(face.side)];
            for(auto p:states.box()) for(unsigned c=0;c<NUMCOMPS;++c)
                states(p,c)=(*boundary.adjData)(p-shift,c);
        }
    }
}


// Face i is the low face of cell i. Reconstruction is linear in the uniform
// mapped coordinates; the geometry is evaluated at the corresponding face.
template<typename T, typename MAP, MemType MEM>
inline void BoxOp_EulerCubedSphere<T,MAP,MEM>::reconstructPLM(
    State& left, State& right, const State& w, const Box& faces, int dir) const
{
    PR_TIMERS("HelioCubed::PLM");
    left.defineForOverwrite(faces); right.defineForOverwrite(faces);
    // A shifted alias reads the same neighbors without allocating and copying
    // three complete face arrays. Alias shifts have the opposite stencil sign.
    // Proto's const alias overload returns a nonmovable const BoxData. The
    // mutable overload only shares storage; these aliases are read-only here.
    auto& source=const_cast<State&>(w);
    const auto wm2=alias(source,Point::Basis(dir,2));
    const auto wm1=alias(source,Point::Basis(dir,1));
    const auto wp1=alias(source,Point::Basis(dir,-1));
    PROTO_ASSERT((wm2.box() & wm1.box() & w.box() & wp1.box()).containsBox(faces),
        "PLM requires two normal ghost cells");
    const int limiter=ParseInputs::get_limiter();
    forallInPlace([] PROTO_LAMBDA(
        Var<T,NUMCOMPS,MEM>& lo, Var<T,NUMCOMPS,MEM>& hi,
        const Var<T,NUMCOMPS,MEM>& mm, const Var<T,NUMCOMPS,MEM>& m,
        const Var<T,NUMCOMPS,MEM>& c, const Var<T,NUMCOMPS,MEM>& p, int limiter)
    {
        HelioCubed::reconstructSecondOrderFace<T>(lo,hi,mm,m,c,p,limiter);
    },left,right,wm2,wm1,w,wp1,limiter);
}

// Reconstruct physical Cartesian primitives. One integrated face normal is
// used for both the physical flux and Riemann dissipation, so uniform vectors
// do not acquire a jump from the panel's changing coordinate basis.
template<typename T, typename MAP, MemType MEM>
inline void BoxOp_EulerCubedSphere<T,MAP,MEM>::secondOrderOperator(
    State& rhs, Array<State,DIM>& fluxes, const State& state, T scale,
    bool cartesianPrimitives,const SecondOrderFaceMask* activeFaces) const
{
    PR_TIMERS("HelioCubed::secondOrderOperator");
    const int block=this->block();
    const T h=this->dx()[1];
    const bool magneticTest=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
    const bool powell=ParseInputs::get_divergence_cleaning()==1;
    State converted;
    if (!cartesianPrimitives)
    {
        State native;
        sphToPrim_2O(native,state,block,!magneticTest);
        auto basis=pointwiseSphericalBasis(native.box(),block,-1);
        HelioCubed::nativePrimitivesToCartesian(converted,native,basis);
    }
    const State& w=cartesianPrimitives ? state : converted;
    const auto* panelStates=cartesianPrimitives ? m_panelStates : nullptr;
    std::vector<CMEParameters> emerging;
    if (cartesianPrimitives && rhs.box().low()[0]==0)
        for (const auto& source:ParseInputs::get_CMEs())
            if (source.injectionMethod==1 && CMEModel::boundaryTaperWeight(source,this->time())>0) {
                CMEParameters current;
                if(CMEModel::expandedSource(source,this->time(),current)) emerging.push_back(current);
            }
    const Box metricBox=rhs.box().grow(1);
    Scalar minusDiv(rhs.box(),T(0));
    Scalar radius,volume;
    Vector dr,adjDr;
    if constexpr (MEM != HOST)
    {
        radius.define(metricBox);volume.define(metricBox);
        dr.define(metricBox);adjDr.define(metricBox);
        radialMetrics(radius,dr,adjDr,volume,metricBox);
    }
    for (int dir=0; dir<DIM; ++dir)
    {
        ScopedHostStack<MEM> scratch;
        const Box faces=fluxes[dir].box();
        State left,right;
        if constexpr (MEM != HOST)
        {
        reconstructPLM(left,right,w,faces,dir);
        if (magneticTest)
        {
            const T b0=testProblemBackgroundBx(ParseInputs::get_init_condition_type());
            forallInPlace([] PROTO_LAMBDA(Var<T,NUMCOMPS,MEM>& l,
                Var<T,NUMCOMPS,MEM>& r,T b0)
            {l(iBX)+=b0;r(iBX)+=b0;},left,right,b0);
        }
        }
        const bool roe=ParseInputs::get_Riemann_solver_type()==2;
        const T entropy=ParseInputs::get_entropy_fix_coeff();
        Scalar bface;
        if (powell) bface.defineForOverwrite(faces);
        if constexpr (MEM == HOST)
        {
            auto& cache=m_faceGeometry[dir];
            if (!cache || cache->faces!=faces)
            {
                PR_TIMERS("HelioCubed::faceGeometrySetup");
                cache=std::make_shared<SecondOrderFaceGeometry>();
                cache->faces=faces;
                Point high=faces.high();high[0]=faces.low()[0];
                const Box plane(faces.low(),high);
                auto normals=CubedSphereShell::adjA<T,MEM>(plane,h,block,dir);
                for (auto p : plane)
                {
                    double normal[DIM];
                    for (int d=0; d<DIM; ++d) normal[d]=normals(p,d);
                    cache->angular.push_back(HelioCubed::secondOrderFrame(normal));
                }
                high=faces.low();high[0]=faces.high()[0];
                const Box line(faces.low(),high);
                Scalar r(line),v(line);Vector d(line),a(line);
                radialMetrics(r,d,a,v,line);
                for (auto p : line) cache->radial.push_back(a(p,dir));
            }
            PR_TIMERS("HelioCubed::secondOrderFaceFlux");
            const auto low=faces.low(),high=faces.high();
            const size_t stride=faces.size();
            const auto sourceBox=w.box();
            const auto sizes=sourceBox.sizes();
            const size_t sourceStride=sourceBox.size();
            const ptrdiff_t step=dir==0 ? 1 : dir==1 ? sizes[0] : size_t(sizes[0])*sizes[1];
            const T* source=w.data();
            const int limiter=ParseInputs::get_limiter();
            const T b0=magneticTest ? testProblemBackgroundBx(ParseInputs::get_init_condition_type()) : T(0);
            PROTO_ASSERT(w.box().containsBox(faces.shift(-Point::Basis(dir,2)))
                && w.box().containsBox(faces.shift(Point::Basis(dir))),
                "Fused PLM requires two normal ghost cells");
            T* output=fluxes[dir].data();T* magneticFlux=powell ? bface.data() : nullptr;
            PROTO_ASSERT(!activeFaces || (*activeFaces)[dir].box()==faces,
                "The active AMR face mask must match the flux storage");
            const unsigned char* active=activeFaces ? (*activeFaces)[dir].data() : nullptr;
            const auto* panelLow=panelStates && !(*panelStates)[2*dir].box().empty()
                ? &(*panelStates)[2*dir] : nullptr;
            const auto* panelHigh=panelStates && !(*panelStates)[2*dir+1].box().empty()
                ? &(*panelStates)[2*dir+1] : nullptr;
            size_t angular=0,offset=0;
            for (int k=low[2]; k<=high[2]; ++k) for (int j=low[1]; j<=high[1]; ++j)
            {
                const auto& frame=cache->angular[angular++];
                const size_t row=(low[0]-sourceBox.low()[0])+size_t(sizes[0])*
                    ((j-sourceBox.low()[1])+size_t(sizes[1])*(k-sourceBox.low()[2]));
                for (int i=low[0]; i<=high[0]; ++i,++offset)
                {
                    // Both adjacent coarse cells are replaced by fine-stage
                    // restriction. Their internal flux is never used by a leaf.
                    if(active && !active[offset])
                    {
                        for(int c=0;c<NUMCOMPS;++c) output[offset+c*stride]=0;
                        if(powell) magneticFlux[offset]=0;
                        continue;
                    }
                    const T* center=source+row+i-low[0];
                    const HelioCubed::StridedFaceState<const T> mm{center-2*step,sourceStride},
                        m{center-step,sourceStride},c{center,sourceStride},p{center+step,sourceStride};
                    HelioCubed::LocalFaceState<T> l,r;
                    HelioCubed::reconstructSecondOrderFace<T>(l,r,mm,m,c,p,limiter);
                    const Point facePoint(i,j,k);
                    if(panelLow && facePoint[dir]==panelLow->box().low()[dir])
                        for(unsigned c=0;c<NUMCOMPS;++c) l(c)=(*panelLow)(facePoint,c);
                    if(panelHigh && facePoint[dir]==panelHigh->box().low()[dir])
                        for(unsigned c=0;c<NUMCOMPS;++c) r(c)=(*panelHigh)(facePoint,c);
                    if(dir==0 && i==0 && !emerging.empty()) {
                        const auto& axes=frame.frame;
                        const double inner=ParseInputs::get_inner_radius();
                        const CMEModel::Vector position={inner*axes[0][0],inner*axes[0][1],inner*axes[0][2]};
                        bool footprint=false;
                        for(const auto& source:emerging) {
                            CMEModel::Sample sample;
                            if(CMEModel::sample(source,position,sample)) {footprint=true;break;}
                        }
                        if(footprint) {
                            // Extrapolate the interior from owned cells only.
                            // Prescribed ghost values must not supply outgoing waves.
                            const HelioCubed::StridedFaceState<const T> q{center+2*step,sourceStride};
                            for(int component=0;component<NUMCOMPS;++component) {
                                const T a=p(component)-c(component),b=q(component)-p(component);
                                T slope=T(.5)*(a+b);
                                if(limiter!=0) slope=a*b<=0 ? T(0) : (a>0 ? min(a,b):max(a,b));
                                if(limiter==3 && a*b>0) slope=(a>0 ? T(1):T(-1))*
                                    min(T(.5)*fabs(a+b),T(2)*min(fabs(a),fabs(b)));
                                r(component)=c(component)-T(.5)*slope;
                            }
                            if(!(r(iRHO)>=RHO00 && r(iP)>0))
                                for(int component=0;component<NUMCOMPS;++component) r(component)=c(component);
                            // Couple the positive exterior driver to the
                            // evolved interior through the Riemann solver.
                            // Share normal B so only the tangential rope field
                            // is prescribed; normal B evolves through induction.
                            T normalJump=0;
                            for(int d=0;d<3;++d) normalJump+=(r(5+d)-l(5+d))*axes[0][d];
                            for(int d=0;d<3;++d) l(5+d)+=normalJump*axes[0][d];
                        }
                    }
                    if (magneticTest) {l(iBX)+=b0;r(iBX)+=b0;}
                    HelioCubed::StridedFaceState<T> out{output+offset,stride};
                    const T area=cache->radial[i-low[0]];
                    HelioCubed::secondOrderFaceFlux(out,l,r,frame,area,m_gamma,entropy,roe);
                    if (powell)
                    {
                        T f=0;
                        for (int d=0; d<DIM; ++d)
                            f+=T(0.5)*(l(iBX+d)+r(iBX+d))*frame.normal[d]*area;
                        magneticFlux[offset]=f;
                    }
                }
            }
        }
        else
        {
            auto normal=CubedSphereShell::adjA<T,MEM>(faces,h,block,dir);
            auto radialArea=slice(adjDr,dir);
            forallInPlace([] PROTO_LAMBDA(Var<T,NUMCOMPS,MEM>& out,
                const Var<T,NUMCOMPS,MEM>& l,const Var<T,NUMCOMPS,MEM>& r,
                const Var<T,DIM,MEM>& normal,const Var<T,1,MEM>& radialArea,
                T gamma,T entropy,bool roe)
            {
                double n[DIM];for (int d=0; d<DIM; ++d)n[d]=normal(d);
                const auto frame=HelioCubed::secondOrderFrame(n);
                HelioCubed::secondOrderFaceFlux(out,l,r,frame,radialArea(0),gamma,entropy,roe);
            },fluxes[dir],left,right,normal,radialArea,m_gamma,entropy,roe);
            if (powell)
                forallInPlace([] PROTO_LAMBDA(Var<T,1,MEM>& f,
                    const Var<T,NUMCOMPS,MEM>& l,const Var<T,NUMCOMPS,MEM>& r,
                    const Var<T,DIM,MEM>& normal,const Var<T,1,MEM>& area)
                {f(0)=0;for(int j=0;j<DIM;++j)f(0)+=T(0.5)*(l(iBX+j)+r(iBX+j))*normal(j)*area(0);},
                    bface,left,right,normal,radialArea);
        }
        rhs += Stencil<T>::FluxDivergence(dir)(fluxes[dir],-scale/this->dx()[dir]);
        if (powell) minusDiv += Stencil<T>::FluxDivergence(dir)(bface,-scale/this->dx()[dir]);
    }
    if (powell)
    {
        const T b0=magneticTest
            ? testProblemBackgroundBx(ParseInputs::get_init_condition_type()) : T(0);
        forallInPlace([] PROTO_LAMBDA(Var<T,NUMCOMPS,MEM>& out,
            const Var<T,1,MEM>& minusDiv,
            const Var<T,NUMCOMPS,MEM>& w,T b0)
        {
            T vb=0;
            for (int d=0; d<DIM; ++d)
            {
                const T b=w(iBX+d)+(d==0 ? b0 : T(0));
                out(iMOMX+d)+=minusDiv(0)*b/T(4*M_PI);
                out(iBX+d)+=minusDiv(0)*w(iVX+d);
                vb+=w(iVX+d)*b;
            }
            out(iE)+=minusDiv(0)*vb/T(4*M_PI);
        },rhs,minusDiv,w,b0);
    }
}
