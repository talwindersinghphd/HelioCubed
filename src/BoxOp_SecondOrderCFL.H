#pragma once

#include <array>
#include <vector>

// Retain the existing CFL estimate, including its averaged-basis conversion
// and direction-dependent face bases. Only geometry and execution are changed.
// Angular geometry occupies one radial plane; radial geometry occupies one
// radial line. Neither is duplicated over full three-dimensional state arrays.
template<typename T, typename MAP, MemType MEM>
struct BoxOp_EulerCubedSphere<T,MAP,MEM>::SecondOrderCFLCache
{
    using Matrix=std::array<std::array<T,DIM>,DIM>;
    struct Angular
    {
        T inverseDet;
        Matrix inverseDetA,cell;
        std::array<Matrix,DIM-1> face;
        std::array<T,DIM> background,det,normal;
    };
    struct Radial
    {
        T inverseVolume,volume,width;
        std::array<T,DIM> area;
    };
    struct Patch
    {
        Box valid;
        T h;
        std::vector<Angular> angular;
        std::vector<Radial> radial;
    };
    LayoutT layout;
    std::vector<Patch> patches;
    SecondOrderCFLCache(const LayoutT& l,const MBLevelBoxData<T,1,MEM>& suppliedVolume):layout(l)
    {
        auto& ops=*levelGeometry(l).op;
        for (auto index : l)
        {
            auto& op=ops[index];
            const Box valid=l[index];
            Point high=valid.high();high[0]=valid.low()[0];
            const Box plane(valid.low(),high);
            Patch patch;
            patch.valid=valid;patch.h=op.dx()[1];
            auto det=CubedSphereShell::detA<T,HOST>(plane,patch.h,0);
            auto detA=CubedSphereShell::detA_A<T,HOST>(plane,patch.h,l.block(index),0);
            auto inverse=forallOverwrite<T,DIM,HOST,DIM>(f_matinv3by3,detA);
            auto cell=op.pointwiseSphericalBasis(plane,l.block(index),-1);
            auto cellInverse=forallOverwrite<T,DIM,HOST,DIM>(f_matinv3by3,cell);
            Array<BoxData<T,DIM,HOST,DIM>,DIM-1> face;
            Array<BoxData<T,1,HOST>,DIM> faceDet;
            Array<BoxData<T,DIM,HOST>,DIM> normal;
            for (int d=0; d<DIM; ++d)
            {
                faceDet[d]=CubedSphereShell::detA<T,HOST>(plane,patch.h,d);
                normal[d]=CubedSphereShell::adjA<T,HOST>(plane,patch.h,l.block(index),d);
                if (d) face[d-1]=op.pointwiseSphericalBasis(plane,l.block(index),d);
            }
            for (auto p : plane)
            {
                Angular a;
                a.inverseDet=T(1)/det(p,0);
                for (int i=0; i<DIM; ++i)
                {
                    a.background[i]=cellInverse(p,i,0);
                    a.det[i]=faceDet[i](p,0);
                    T n2=0;
                    for (int j=0; j<DIM; ++j)
                    {
                        a.inverseDetA[i][j]=inverse(p,i,j);
                        a.cell[i][j]=cell(p,i,j);
                        n2+=normal[i](p,j)*normal[i](p,j);
                        for (int d=1; d<DIM; ++d) a.face[d-1][i][j]=face[d-1](p,i,j);
                    }
                    a.normal[i]=std::sqrt(n2);
                }
                patch.angular.push_back(a);
            }
            high=valid.low();high[0]=valid.high()[0];
            const Box line(valid.low(),high);
            Scalar radius(line),volume(line);
            Vector dr(line),area(line);
            op.radialMetrics(radius,dr,area,volume,line);
            for (auto p : line)
            {
                Radial r;
                r.volume=volume(p,0);
                r.inverseVolume=T(1)/suppliedVolume[index](p,0);
                r.width=dr(p,0)*op.dx()[0];
                for (int d=0; d<DIM; ++d) r.area[d]=area(p,d);
                patch.radial.push_back(r);
            }
            patches.push_back(std::move(patch));
        }
    }
};

template<typename T, typename MAP, MemType MEM>
inline T BoxOp_EulerCubedSphere<T,MAP,MEM>::dtCFLSecondOrderHost(
    const MBLevelBoxData<T,NUMCOMPS,MEM>& state,
    const MBLevelBoxData<T,1,MEM>& suppliedVolume,
    const MBLevelBoxData<T,1,MEM>* mappedScale,
    T coneBoundaryDensity)
{
    static_assert(MEM == HOST && DIM == 3 && NUMCOMPS == 8);
    PR_TIMERS("HelioCubed::dtCFLSecondOrderHost");
    const auto& layout=state.layout();
    auto& cache=levelGeometry(layout).secondOrderCFL;
    if (!cache || &cache->layout.partition()!=&layout.partition())
        cache=std::make_shared<SecondOrderCFLCache>(layout,suppliedVolume);
    const bool magneticTest=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
    const T b0=testProblemBackgroundBx(ParseInputs::get_init_condition_type());
    const T gamma=ParseInputs::get_gamma();
    T localMaximum=0;
    unsigned patchIndex=0;
    for (auto index : layout)
    {
        const auto& patch=cache->patches[patchIndex++];
        const auto lo=patch.valid.low(),hi=patch.valid.high();
        const auto storage=state[index].box();
        const auto sizes=storage.sizes();
        const size_t componentStride=storage.size();
        const T* data=state[index].data();
        PROTO_ASSERT(!mappedScale || (*mappedScale)[index].box()==patch.valid,
            "Mapped CFL scaling must cover exactly the valid patch");
        const T* scaling=mappedScale ? (*mappedScale)[index].data() : nullptr;
        size_t scaleOffset=0;
        unsigned angularIndex=0;
        for (int k=lo[2]; k<=hi[2]; ++k) for (int j=lo[1]; j<=hi[1]; ++j)
        {
            const auto& a=patch.angular[angularIndex++];
            const size_t row=(lo[0]-storage.low()[0])+size_t(sizes[0])*
                ((j-storage.low()[1])+size_t(sizes[1])*(k-storage.low()[2]));
            for (int i=lo[0]; i<=hi[0]; ++i,++scaleOffset)
            {
                const auto& r=patch.radial[i-lo[0]];
                const size_t offset=row+i-lo[0];
                // Round JU*scale before the existing CFL arithmetic, exactly
                // as when the AMR driver materialized a separate state array.
                T u[NUMCOMPS];
                for(int c=0;c<NUMCOMPS;++c)
                    u[c]=scaling ? data[offset+c*componentStride]*scaling[scaleOffset]
                                 : data[offset+c*componentStride];
                T rho=u[iRHO]*a.inverseDet;
                rho*=r.inverseVolume;
                rho=std::max(rho,T(RHO00));
                T energy=u[iE]*a.inverseDet;
                energy*=r.inverseVolume;
                T v[DIM]={},b[DIM]={};
                for (int d=0; d<DIM; ++d)
                {
                    for (int q=0; q<DIM; ++q)
                    {
                        v[d]+=a.inverseDetA[d][q]*u[iMOMX+q];
                        b[d]+=a.inverseDetA[d][q]*u[iBX+q];
                    }
                    v[d]*=r.inverseVolume;b[d]*=r.inverseVolume;
                    v[d]/=rho;
                }
                T vc[DIM]={},bc[DIM]={},kinetic=0,magnetic=0,b2=0;
                for (int d=0; d<DIM; ++d)
                {
                    for (int q=0; q<DIM; ++q)
                    {
                        vc[d]+=a.cell[d][q]*v[q];
                        bc[d]+=a.cell[d][q]*b[q];
                    }
                    kinetic+=vc[d]*vc[d]/2;
                    magnetic+=bc[d]*bc[d]/8.0/M_PI;
                    b2+=bc[d]*bc[d];
                }
                kinetic*=rho;
                if (magneticTest)
                {
                    magnetic=(2*b0*bc[0]+b2)/(8*M_PI);
                    for (int d=0; d<DIM; ++d) b[d]+=a.background[d]*b0;
                }
                T pressure=magneticTest ? c_TEST_P0+(gamma-1)*(energy-kinetic-magnetic)
                                       : (energy-kinetic-magnetic)*(gamma-1);
                if (pressure<0) pressure=0;
                T inverseDt=0;
                for (int d=0; d<DIM; ++d)
                {
                    const auto& basis=d==0 ? a.cell : a.face[d-1];
                    T v2=0,fullB2=0;
                    for (int q=0; q<DIM; ++q)
                    {
                        T velocity=0,field=0;
                        for (int s=0; s<DIM; ++s)
                        {velocity+=basis[q][s]*v[s];field+=basis[q][s]*b[s];}
                        v2+=velocity*velocity;fullB2+=field*field;
                    }
                    const T width=d==0 ? r.width
                        : r.volume*a.det[d]*patch.h/r.area[d]/a.normal[d];
                    // A cone can prescribe a smaller total density while
                    // retaining ambient B. Bound the corresponding Alfven
                    // speed before that state reaches any owned cell.
                    const T magneticDensity=coneBoundaryDensity>0 ? std::min(rho,coneBoundaryDensity) : rho;
                    const T wave=std::sqrt(gamma*pressure/rho+fullB2/4.0/M_PI/magneticDensity);
                    inverseDt+=(wave+std::sqrt(v2))/width;
                }
                localMaximum=std::max(localMaximum,inverseDt);
            }
        }
    }
    Reduction<T,Operation::Max,HOST> maximum;
    maximum.reset();maximum.reduce(&localMaximum,1);
    return T(1)/maximum.fetch();
}
