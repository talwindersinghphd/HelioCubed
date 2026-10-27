#pragma once

#include "BoxOp_EulerCubedSphere.H"
#include "MHD_AMRInterpolation.H"
#include "MHD_AMRTransfers.H"
#include "MHD_AMRTagging.H"
#include "MHD_IO.H"
#include "MHD_CME.H"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>

namespace HelioCubed
{
#ifdef PR_AMR
// Multiblock flux reconciliation and mapped conservative transfer follow
// McCorquodale et al., CAMCoS 10 (2015), doi:10.2140/camcos.2015.10.121.
// The O2 path uses Cartesian PLM. The O4 path uses the existing
// fourth-order mapped flux operators and conservative quartic transfers.
// Time stepping supports SSPRK2, Kutta RK3, and classical RK4, with
// conservative multirate extrapolation at temporal orders three and four.
// Cell volumes must be additive across refinement levels. Split the spherical
// quadrilateral into two triangles; this avoids subtracting nearly equal
// corner angles, which loses accuracy on fine meshes.
inline double amrAngularJacobian(Point p, double h)
{
    const double step=M_PI*h/2;
    const double alpha=M_PI*(p[1]*h/2-0.25),beta=M_PI*(p[2]*h/2-0.25);
    const double a0=tan(alpha),a1=tan(alpha+step),b0=tan(beta),b1=tan(beta+step);
    const double r00=sqrt(1+a0*a0+b0*b0),r10=sqrt(1+a1*a1+b0*b0);
    const double r11=sqrt(1+a1*a1+b1*b1),r01=sqrt(1+a0*a0+b1*b1);
    const double diagonal=1+a0*a1+b0*b1;
    const double first=r00*r10*r11+(1+a0*a1+b0*b0)*r11
        +(1+a1*a1+b0*b1)*r00+diagonal*r10;
    const double second=r00*r11*r01+diagonal*r01
        +(1+a0*a1+b1*b1)*r00+(1+a0*a0+b0*b1)*r11;
    // tan(x+step)-tan(x) = sin(step)/(cos(x)*cos(x+step)).
    const double sine=sin(step);
    const double determinant=sine*sine/(cos(alpha)*cos(alpha+step)*cos(beta)*cos(beta+step));
    return (2*atan2(determinant,first)+2*atan2(determinant,second))/(h*h);
}

inline double amrSlope(double left,double center,double right)
{
    const double a=center-left,b=right-center;
    if (a*b<=0) return 0;
    return (a>0 ? 1.0 : -1.0)*std::min({2*abs(a),2*abs(b),0.5*abs(a+b)});
}

// Storing and recovering J*U can move density at its floor and zero pressure
// by a few ulps.
// Allow that margin only for the constant conservative fallback; interpolated
// trial children still use the strict positivity test. No energy is added.
inline bool amrAdmissibleParent(const double* u,double gamma,bool magnetic,double b0)
{
    if(!(u[iRHO]>=RHO00*(1-64*std::numeric_limits<double>::epsilon()))
        || !std::isfinite(u[iRHO])) return false;
    const double background=magnetic ? c_TEST_P0/(gamma-1) : 0;
    double thermal=u[iE]+background,scale=abs(u[iE])+abs(background);
    for(int d=0;d<DIM;++d)
    {
        const double energy=0.5*u[iMOMX+d]*u[iMOMX+d]/u[iRHO]+u[iBX+d]*u[iBX+d]/(8*M_PI);
        thermal-=energy;scale+=energy;
    }
    const double cross=b0*u[iBX]/(4*M_PI);
    thermal-=cross;scale+=abs(cross);
    return std::isfinite(thermal) && thermal>=-64*std::numeric_limits<double>::epsilon()*scale;
}

// These arrays are copied as patch data. Only the primitive donor participates
// in panel exchange; other state/scratch arrays need no separate panel buffers.
inline Array<Point,DIM+1> amrPatchGhosts(Point ghost)
{
    Array<Point,DIM+1> result;
    result.fill(Point::Zeros());result[0]=ghost;
    return result;
}

class CubedSphereAMR
{
public:
    using OP=BoxOp_EulerCubedSphere<double,MBMap_CubedSphereShell<HOST>,HOST>;
    using Level=MBLevelBoxData<double,NUMCOMPS,HOST>;
    using ScalarLevel=MBLevelBoxData<double,1,HOST>;
    using Register=MBLevelFluxRegister<double,NUMCOMPS,HOST>;
    static bool fourthOrder() {return ParseInputs::get_spatial_order()==4;}
    struct GridLevel
    {
        unsigned number;
        Point ratio;
        MBDisjointBoxLayout layout;
        OP::LevelGeometry geometry;
        OP::MapT& map;
        OP::OpT& op;
        // state/stage store mapped Cartesian conserved averages <JU>.
        // The O2 operator consumes Cartesian primitives; the O4 operators
        // consume native-basis conserved averages and reconstruct W internally.
        Level state,stage,primitive,donor,rhs,first;
        std::unique_ptr<ScalarLevel> cflScale;
        std::unique_ptr<MBLevelBoxData<int,1,HOST>> coveredCells;
        std::vector<std::unique_ptr<OP::SecondOrderFaceMask>> activeFaces;
        const Level* preparedState=nullptr;
        std::unique_ptr<Level> old,predictor;
        std::unique_ptr<Level> cartesian; // O4 Cartesian conserved averages <U>.
        std::unique_ptr<NativeConversionCache<double>> nativeConversion;
        std::array<std::unique_ptr<Level>,4> rk;
        std::unique_ptr<Level> restricted,coarseSource4,mapped4;
        std::unique_ptr<MBLevelBoxData<double,NUMCOMPS+1,HOST>> coarseMapped2,coarseSource2;
        std::unique_ptr<AMRTransferCopier<NUMCOMPS>> restrictionCopy;
        std::unique_ptr<AMRTransferCopier<NUMCOMPS,true>> coarseCopy4;
        std::unique_ptr<AMRTransferCopier<NUMCOMPS+1,true>> coarseCopy2;
        std::vector<AMRSecondOrderInterpolationPlan> secondOrderGhosts;
        std::vector<AMRFourthOrderInterpolationPlan> fourthOrderGhosts;
        std::vector<AMRRefinementMask> tagMasks;
        Stencil<double> restrictionStencil;
        std::unique_ptr<Level> extrapolatedFirst;
        std::vector<std::unique_ptr<Array<State,DIM>>> initialFlux;
        bool firstStageUsed=false;
        double startTime=0,stepDt=0,preparedTime=0;
        double preparedChildStart=0,preparedChildDt=0;
        int preparedChildStage=-1;
        int substeps=1;
        bool timeInterpolated=false;
        ScalarLevel volume,jacobian;
        MBInterpOp interp;
        std::unique_ptr<SecondOrderSeamPlan<double,NUMCOMPS>> secondOrderSeam;
        MBBoundaryRegister<double,NUMCOMPS,HOST> panelFlux;
        std::vector<std::unique_ptr<Array<State,DIM>>> flux;

        GridLevel(const MBDisjointBoxLayout& grid,unsigned level,bool parent,Point refinement=Point::Ones(2))
            :number(level),ratio(refinement),layout(grid),geometry(makeGeometry(grid)),map(*geometry.map),op(*geometry.op),
             state(grid,amrPatchGhosts(OP::ghost())),stage(grid,amrPatchGhosts(OP::ghost())),
             primitive(grid,amrPatchGhosts(fourthOrder() ? OP::ghost()+Point::Ones(ParseInputs::get_seam_pressure_correction()) : Point::Ones(level && !parent ? 2 : 4))),
             donor(grid,primitive.ghost()[0]+(fourthOrder() ? Point::Ones() : Point::Zeros())+Point(0,2,2)),
             rhs(grid,Point::Zeros()),first(grid,Point::Zeros()),
             volume(grid,amrPatchGhosts(primitive.ghost()[0]+Point::Basis(0,2))),
             jacobian(grid,amrPatchGhosts(fourthOrder() ? Point::Ones(5) : level && !parent ? Point::Zeros() : Point::Ones(4))),
             panelFlux(grid,1,Point::Zeros())
        {
            PR_TIMERS("HelioCubed::AMR::GridLevel");
            const Point interpolationGhost=primitive.ghost()[0]+(fourthOrder() ? Point::Ones() : Point::Zeros());
            if(fourthOrder()) cartesian=std::make_unique<Level>(grid,amrPatchGhosts(interpolationGhost));
            interp=level ? fineInterpolation(grid,donor,interpolationGhost,refinement[1]>1 || (!parent && !fourthOrder()))
                        : CubedSphereShell::MHDInterpOp<HOST>(grid,donor.ghost()[0],4);
            for (auto index : layout)
            {
                const Box box=volume[index].box();
                Scalar radius(box); Vector dr(box),area(box);
                op[index].radialMetrics(radius,dr,area,volume[index],box);
                const double h=op[index].dx()[1];
                const Box jacBox=jacobian[index].box();
                Point high=jacBox.high();high[0]=jacBox.low()[0];
                const Box plane(jacBox.low(),high);
                Scalar angular(plane);
                for(auto p : plane) angular(p)=amrAngularJacobian(p,h);
                for (auto p : jacobian[index].box())
                    jacobian[index](p)=volume[index](p)*angular(Point(jacBox.low()[0],p[1],p[2]));
                flux.push_back(std::make_unique<Array<State,DIM>>());
                for (int d=0; d<DIM; ++d)
                    (*flux.back())[d].define(layout[index].extrude(d));
            }
        }

        static OP::LevelGeometry makeGeometry(const MBDisjointBoxLayout& layout)
        {
            OP::LevelGeometry geometry;
            geometry.map=std::make_unique<OP::MapT>(CubedSphereShell::Map(layout,OP::ghost()));
            geometry.op=std::make_unique<OP::OpT>(
                CubedSphereShell::Operator<BoxOp_EulerCubedSphere,double,HOST>(*geometry.map));
            return geometry;
        }

        static MBInterpOp fineInterpolation(const MBDisjointBoxLayout& grid,Level& donor,Point ghost,bool coarseDonors)
        {
            PR_TIMERS("HelioCubed::AMR::fineInterpolation");
            // At a panel/refinement corner the full fine stencil may not
            // exist. Retain the coarse fill there, at the selected spatial
            // order. Use panel interpolation where its fine donors exist.
            using PolarMap=MBLevelMap<MBMap_CubedSphereShellPolar<HOST>,HOST>;
            std::vector<std::unique_ptr<PolarMap>> maps;
            for(unsigned block=0;block<6;++block)
            {
                maps.push_back(std::make_unique<PolarMap>());
                maps.back()->define(grid,ghost+Point(0,2,2),block);
            }
            MBInterpOp result(ghost,4);
            std::vector<Point> footprint;
            for(auto p : Box(Point(0,-1,-1),Point(0,1,1))) footprint.push_back(p);
            for(int d=1;d<DIM;++d) {footprint.push_back(Point::Basis(d,2));footprint.push_back(Point::Basis(d,-2));}
            const auto& domain=grid.domain();
            for(auto index : grid)
            {
                const auto block=grid.block(index);
                const auto& map=*maps[block];
                const Box physical=domain.getBlock(block).box(),valid=grid[index];
                Point angularGhost=ghost*Point(0,1,1);
                Point low=valid.low()-angularGhost,high=valid.high()+angularGhost;
                high[0]=low[0];
                for(auto target : Box(low,high))
                {
                    if(physical.containsPoint(target)) continue;
                    auto center=map.op(0).findValid(domain,MBPoint(target,block));
                    bool available=true;
                    for(auto offset : footprint)
                    {
                        MBPoint source(center.point+offset,center.block);
                        if(!domain.getBlock(source.block).box().containsPoint(source.point))
                        {
                            const Point dir=domain.getBlock(source.block).box().whichBoundaryContains(source.point);
                            const auto boundaries=domain.boundaries(source.block,dir);
                            if(dir.codim()==1 && boundaries.size()==1)
                            {
                                const auto neighbor=boundaries[0].dstBlock;
                                source.point=domain.convertPoint(source.point,source.block,neighbor,PR_CELL);
                                source.block=neighbor;
                            }
                            else source=map.op(0).findValid(domain,source);
                        }
                        const Point local=domain.convertPoint(source.point,source.block,block,PR_CELL);
                        if(source.block==block)
                        {
                            if(!donor[index].box().containsPoint(local)) available=false;
                            if(!coarseDonors && grid.find(source.point/grid.boxSize(source.block),source.block)==*grid.end())
                                available=false;
                        }
                        else
                        {
                            const auto neighbor=grid.find(source.point/grid.boxSize(source.block),source.block);
                            if(neighbor==*grid.end()) {available=false;break;}
                            const Point dir=grid.connectivity(index,neighbor);
                            bool found=false;
                            if(dir!=Point::Zeros()) for(const auto& boundary : donor.bounds(index,dir))
                                if(boundary.adjIndex==neighbor && boundary.localData->box().containsPoint(local)) found=true;
                            if(!found) available=false;
                        }
                        if(!available) break;
                    }
                    if(!available) continue;
                    auto first=std::make_shared<MBPointInterpOp>(MBDataPoint(index,target,grid),ghost+Point(0,2,2),map,4);
                    result.insert(first);
                    for(int radial=valid.low()[0]+1;radial<=valid.high()[0];++radial)
                    {
                        const Point shift=Point::Basis(0,radial-valid.low()[0]);
                        std::vector<MBDataPoint> sources;
                        for(const auto& point : first->sources()) sources.push_back(point.shift(shift,grid));
                        result.insert(std::make_shared<MBPointInterpOp>(first->target().shift(shift,grid),sources,first->coefs()));
                    }
                }
            }
            return result;
        }
    };

    explicit CubedSphereAMR(MBProblemDomain domain,Point boxSize,const std::string& restart="",
        Point regionScale=Point::Ones(),std::string outputSuffix="")
        :m_domain(domain),m_boxSize(boxSize),m_regionScale(regionScale),m_outputSuffix(std::move(outputSuffix))
    {
        configureHierarchy(!restart.empty());
        MBDisjointBoxLayout layout(m_domain,m_boxSize);
        m_levels.push_back(std::make_unique<GridLevel>(layout,0,!m_ratios.empty()));
        if(!restart.empty()) {read(restart);writeGridSummary("restart",restartTime);return;}
        initialize(*m_levels[0]);
        regrid(0,true);
    }

    unsigned numLevels() const {return m_levels.size();}
    GridLevel& level(unsigned l) {return *m_levels.at(l);}
    const GridLevel& level(unsigned l) const {return *m_levels.at(l);}

    bool insertCME(double time);
    double convergenceDt() const {return m_convergenceDt;}
    void setConvergenceDt(double dt) {m_convergenceDt=dt;}
    double nextCMETime() const
    {return m_cmeSchedule.inserted() ? std::numeric_limits<double>::infinity() : m_cmeSchedule.time();}
    std::pair<int,std::array<double,NUMCOMPS>> sampleProbe(const Array<double,DIM>& position,double time)
    {prepareStates(time);return samplePreparedProbe(position);}
    void writeProbes(int iteration,double time,bool giveSpace);
    void writeSlices(int iteration,double time);

    struct FloorRepairs
    {
        long long density=0,pressure=0;
        bool changed() const {return density || pressure;}
    };

    FloorRepairs applyFloors(bool repairPressure=true)
    {
        PR_TIMERS("HelioCubed::AMR::floors");
        const double gamma=ParseInputs::get_gamma();
        const bool magnetic=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
        const double background=magnetic ? c_TEST_P0 : 0;
        const double b0=magnetic ? testProblemBackgroundBx(ParseInputs::get_init_condition_type()) : 0;
        long long counts[3]={0,0,0}; // Density repairs, pressure repairs, invalid states.
        cacheCoveredCells();
        for(auto* grid : levels()) for(auto index : grid->layout)
        {
            auto& state=grid->state[index];
            const auto& jacobian=grid->jacobian[index];
            const auto* mask=grid->coveredCells ? &(*grid->coveredCells)[index] : nullptr;
            for(auto p : grid->layout[index])
            {
                if(mask && (*mask)(p)) continue;
                const double j=jacobian(p);
                if(!(j>0) || !std::isfinite(j)) {++counts[2];continue;}
                const double inverseJ=1/j;
                double rho=state(p,iRHO)*inverseJ;
                if(!std::isfinite(rho)) {++counts[2];continue;}
                const bool lowDensity=rho<RHO00 || state(p,iRHO)/j<RHO00;
                if(!repairPressure && !lowDensity) continue;
                bool finite=true;
                for(int c=0;c<NUMCOMPS;++c) finite=finite && std::isfinite(state(p,c));
                if(!finite) {++counts[2];continue;}
                if(lowDensity)
                {
                    // Add mass without changing momentum, B or total energy.
                    // Recheck both recovery forms to avoid rounding below RHO00.
                    double mapped=std::max(state(p,iRHO),j*RHO00);
                    while(mapped*inverseJ<RHO00 || mapped/j<RHO00)
                        mapped=std::nextafter(mapped,std::numeric_limits<double>::infinity());
                    if(!std::isfinite(mapped)) {++counts[2];continue;}
                    state(p,iRHO)=mapped;rho=mapped*inverseJ;++counts[0];
                }
                double kinetic=0,magneticEnergy=0;
                for(int d=0;d<DIM;++d)
                {
                    const double velocity=state(p,iMOMX+d)*inverseJ/rho;
                    const double field=state(p,iBX+d)*inverseJ;
                    kinetic+=0.5*rho*velocity*velocity;
                    magneticEnergy+=field*field/(8*M_PI);
                }
                magneticEnergy+=b0*(state(p,iBX)*inverseJ)/(4*M_PI);
                auto pressure=[&](double energy)
                {return background+(gamma-1)*(energy*inverseJ-kinetic-magneticEnergy);};
                const double original=state(p,iE),current=pressure(original);
                if(!std::isfinite(current)) {++counts[2];continue;}
                if(current>=0) continue;
                // JU contains Cartesian conserved cell averages at both
                // orders. Restore zero cell-average pressure by changing
                // only energy, including the magnetic test equilibrium.
                double energy=std::max(original,j*(kinetic+magneticEnergy-background/(gamma-1)));
                // Round upward when storing J*E would otherwise leave a
                // negative pressure by a few ulps. No positive floor is added.
                while(std::isfinite(energy) && pressure(energy)<0)
                    energy=std::nextafter(energy,std::numeric_limits<double>::infinity());
                if(!std::isfinite(energy) || !std::isfinite(pressure(energy)))
                {++counts[2];continue;}
                state(p,iE)=energy;++counts[1];
            }
        }
#ifdef PR_MPI
        MPI_Allreduce(MPI_IN_PLACE,counts,3,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
#endif
        if(counts[2]) throw std::runtime_error("Invalid conserved AMR state during density/pressure repair");
        if(counts[0] || counts[1])
        {
            averageDown();
            m_effectiveDt.clear();
            for(auto* grid : levels())
            {
                grid->preparedState=nullptr;grid->timeInterpolated=false;
                grid->preparedChildStage=-1;grid->firstStageUsed=false;
            }
        }
        return {counts[0],counts[1]};
    }

    long long applyPressureFloor() {return applyFloors().pressure;}

    FloorRepairs advance(double dt,double time,bool repairDensity=false)
    {
        PR_TIMERS("HelioCubed::AMR::advance");
        if(m_effectiveDt.size()!=numLevels()) timestep();
        const int order=ParseInputs::get_temporal_order();
        bool subcycle=false;
        if(ParseInputs::get_amr_subcycling())
            for(unsigned l=1;l<numLevels();++l)
                if(dt>m_effectiveDt.at(l)) {subcycle=true;break;}
        startProgress(subcycle && order>=3 ? order-1 : 1);
        if(order>=3)
        {
            if(subcycle)
            {
                if(order==3) {PR_TIMERS("HelioCubed::AMR::ThirdOrderSubcycled");advanceExtrapolated(dt,time,3);}
                else {PR_TIMERS("HelioCubed::AMR::FourthOrderSubcycled");advanceExtrapolated(dt,time,4);}
            }
            else advanceRungeKuttaSynchronous(dt,time,order);
            const auto repaired=repairDensity ? applyFloors(false) : FloorRepairs{};
            for(auto* grid : levels())
                if(!OP::validConservedState(grid->state)) throw std::runtime_error("Invalid conserved AMR state after Runge-Kutta step");
            writeProgress(time+dt);
            return repaired;
        }
        if(subcycle)
        {
            beginStep(0,dt,time);
            completeStep(0);
        }
        else advanceSynchronous(dt,time);
        const auto repaired=repairDensity ? applyFloors(false) : FloorRepairs{};
        for(auto* grid : levels())
            if(!OP::validConservedState(grid->state))
                throw std::runtime_error("Invalid conserved AMR state after SSPRK2 step");
        writeProgress(time+dt);
        return repaired;
    }

    double timestep()
    {
        double dt=std::numeric_limits<double>::max();
        m_effectiveDt.assign(numLevels(),dt);
        for(auto* grid : levels())
        {
            OP::GeometryScope geometry(grid->geometry);
            if(fourthOrder())
            {
                m_effectiveDt[grid->number]=ParseInputs::get_CFL()*OP::dtCFL(grid->state,grid->interp,grid->volume);
                continue;
            }
            if(!grid->cflScale)
            {
                grid->cflScale=std::make_unique<ScalarLevel>(grid->layout,Point::Zeros());
                for(auto index : grid->layout)
                {
                    const double h=grid->op[index].dx()[1];
                    auto det=CubedSphereShell::detA<double,HOST>(grid->layout[index],h,0);
                    for(auto p : grid->layout[index])
                        (*grid->cflScale)[index](p)=grid->volume[index](p)*det(p)/grid->jacobian[index](p);
                }
            }
            m_effectiveDt[grid->number]=ParseInputs::get_CFL()*OP::dtCFLSecondOrderHost(
                grid->state,grid->volume,grid->cflScale.get());
        }
        for(unsigned l=numLevels()-1;l>0;--l)
        {
            const double factor=ParseInputs::get_amr_subcycling() ? maxRatio(m_ratios[l-1]) : 1;
            m_effectiveDt[l-1]=std::min(m_effectiveDt[l-1],factor*m_effectiveDt[l]);
        }
        return m_effectiveDt[0];
    }

    void regrid(double time,bool initial=false)
    {
        PR_TIMERS("HelioCubed::AMR::regrid");
        const auto start=std::chrono::steady_clock::now();
        // A step or CME can invalidate pressure after the cadence repair.
        // Conservative children require admissible parent averages now.
        const auto repaired=applyFloors();
        if(repaired.changed())
            pout(0)<<"AMR regrid floors: density cells = "<<repaired.density
                <<" pressure cells = "<<repaired.pressure<<" time = "<<formatProgressTime(time)<<std::endl;
        averageDown();
        prepareStates(time);
        std::vector<PatchSet> desired(m_ratios.size()+1);
        for(unsigned l=1;l<desired.size();++l) desired[l]=taggedPatches(l-1);
        bool changed=false;
        while(true)
        {
            enforceNesting(desired);
            const bool passChanged=realizeHierarchy(desired,time,initial);
            changed=passChanged||changed;
            averageDown();
            const unsigned available=std::min(numLevels(),unsigned(m_ratios.size()));
            if(!ParseInputs::uses_amr_tagging() || !passChanged) break;
            prepareStates(time);
            bool newTags=false;
            // New patches on an existing intermediate level also need tagging.
            // Grow monotonically within this regrid so a newly exposed feature
            // reaches every requested level without waiting for another step.
            for(unsigned l=1;l<available;++l)
                for(auto key : taggedPatches(l)) newTags=desired[l+1].insert(key).second||newTags;
            if(!newTags) break;
        }
        if(initial || changed || ParseInputs::get_amr_verbosity())
        {
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            writeGridSummary("regrid",time,seconds,changed);
        }
    }

    void averageDown()
    {
        for(unsigned l=numLevels()-1;l>0;--l)
            restrictTo(l,level(l).state,level(l-1).state);
    }

    void write(int iteration,double time,double dt,bool checkpoint)
    {
        PR_TIMERS("HelioCubed::AMR::write");
        HDF5Handler h5;
        h5.setTime(time); h5.setTimestep(dt); h5.setIter(iteration);
        const std::string prefix=(checkpoint ? ParseInputs::get_checkpoint_file_prefix()
                                             : ParseInputs::get_data_file_prefix())
            +m_outputSuffix+"_amr_"+std::to_string(iteration);
        if(!checkpoint)
        {
            if(ParseInputs::get_output_divB()) writePlot<true>(h5,prefix,time);
            else writePlot<false>(h5,prefix,time);
            return;
        }
        int number=0;
        for (auto* level : levels())
            h5.writeMBLevel(level->state,prefix+"_L"+std::to_string(number++));
        int manifestError=0;
        if (checkpoint && procID()==0)
        {
            std::ofstream file(prefix+".amr");
            file<<"HelioCubedAMR 5\n"<<iteration<<' '<<std::setprecision(17)<<time<<' '<<dt<<'\n';
            file<<number<<' '<<m_ratios.size()<<'\n';
            for(int d=0;d<DIM;++d) file<<m_domain.getBlock(0).sizes()[d]<<' ';
            for(int d=0;d<DIM;++d) file<<m_boxSize[d]<<' ';
            file<<'\n'<<ParseInputs::get_inner_radius()<<' '<<ParseInputs::get_outer_radius()<<' '
                <<ParseInputs::get_radial_stretch()<<' '<<ParseInputs::get_gamma()<<' '
                <<ParseInputs::get_init_condition_type()<<'\n';
            for(auto ratio : m_ratios) file<<ratio[0]<<' '<<ratio[1]<<' '<<ratio[2]<<'\n';
            for(unsigned l=1;l<m_boxSizes.size();++l)
                file<<m_boxSizes[l][0]<<' '<<m_boxSizes[l][1]<<' '<<m_boxSizes[l][2]<<'\n';
            file<<m_cmeSchedule.countInserted()<<'\n';
            for(unsigned l=1;l<numLevels();++l)
            {
                const auto patches=level(l).layout.patches();
                file<<patches.size()<<'\n';
                for(auto patch : patches) file<<patch.block<<' '<<patch.point[0]<<' '
                    <<patch.point[1]<<' '<<patch.point[2]<<'\n';
            }
            file<<"convergence_dt "<<m_convergenceDt<<'\n';
            file.close();
            if (!file) manifestError=1;
        }
#ifdef PR_MPI
        if(checkpoint) MPI_Bcast(&manifestError,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
        if(manifestError) throw std::runtime_error("Could not write AMR checkpoint manifest");
        if(checkpoint)
        {
            m_checkpoints.emplace_back(prefix,number);
            const unsigned keep=std::max(1,ParseInputs::get_max_checkpoint_files());
            while(m_checkpoints.size()>keep)
            {
                if(procID()==0)
                {
                    std::remove((m_checkpoints.front().first+".amr").c_str());
                    for(unsigned level=0;level<m_checkpoints.front().second;++level)
                        std::remove((m_checkpoints.front().first+"_L"+std::to_string(level)+".hdf5").c_str());
                }
                m_checkpoints.erase(m_checkpoints.begin());
            }
        }
    }

    int restartIteration=0;
    double restartTime=0,restartDt=0;

private:
    template<bool DivB> void writePlot(HDF5Handler&,const std::string&,double);
    std::pair<int,std::array<double,NUMCOMPS>> samplePreparedProbe(const Array<double,DIM>&);
    void restrictTo(unsigned l,const Level& source,Level& destination)
    {
        PR_TIMERS("HelioCubed::AMR::restrict");
        auto& grid=level(l);
        if(!grid.restricted)
        {
            grid.restricted=std::make_unique<Level>(grid.layout.coarsen(grid.ratio),Point::Zeros());
            grid.restrictionCopy=std::make_unique<AMRTransferCopier<NUMCOMPS>>(*grid.restricted,destination);
            grid.restrictionStencil=Stencil<double>::AvgDown(grid.ratio);
        }
        for(auto index : grid.layout) (*grid.restricted)[index] |= grid.restrictionStencil(source[index]);
        grid.restrictionCopy->copy(*grid.restricted,destination);
    }

    void prepareFourthOrder(GridLevel&,const Level&,double,GridLevel*,bool=false);
    void prolongFourthOrder(GridLevel&,GridLevel&,Point);
    void advanceRungeKuttaSynchronous(double,double,int);
    void advanceExtrapolated(double,double,int);
    void prepareExtrapolatedRK2Stage(unsigned,double,double,int);
    State fourthOrderPrimitives(const GridLevel&,const MBIndex&) const;
    using PatchKey=std::array<int,4>;
    struct PatchLess
    {
        bool operator()(const PatchKey& a,const PatchKey& b) const
        {for(int d : {0,3,2,1}) if(a[d]!=b[d]) return a[d]<b[d];return false;}
    };
    using PatchSet=std::set<PatchKey,PatchLess>;
    struct Region {unsigned level;int block;Box cells;};
    MBProblemDomain m_domain;
    Point m_boxSize;
    Point m_regionScale;
    std::string m_outputSuffix;
    CMEInsertionSchedule m_cmeSchedule{ParseInputs::get_CMEs()};
    double m_convergenceDt=-1;
    std::vector<Point> m_ratios,m_boxSizes,m_cumulative;
    std::vector<MBProblemDomain> m_domains;
    std::vector<std::unique_ptr<GridLevel>> m_levels;
    std::vector<std::unique_ptr<Register>> m_registers;
    std::vector<Region> m_regions;
    std::vector<double> m_effectiveDt;
    std::vector<int> m_extrapolationSubsteps;
    bool m_captureInitial=false,m_reuseInitial=false;
    std::vector<std::pair<std::string,unsigned>> m_checkpoints;
    struct LevelProgress
    {
        unsigned long long steps=0,refluxCalls=0;
        double minDt=std::numeric_limits<double>::max(),maxDt=0;
    };
    std::vector<LevelProgress> m_progress;
    unsigned m_progressPasses=1;

    static int maxRatio(Point ratio)
    {int result=1;for(int d=0;d<DIM;++d) result=std::max(result,ratio[d]);return result;}

    void accumulateFlux(GridLevel& grid,double weight)
    {
        unsigned local=0;
        for(auto index : grid.layout)
        {
            for(int d=0;d<DIM;++d)
            {
                if(grid.number+1<numLevels())
                    m_registers[grid.number]->incrementCoarseRegister((*grid.flux[local])[d],index,d,weight);
                if(grid.number)
                    m_registers[grid.number-1]->incrementFineRegister((*grid.flux[local])[d],index,d,weight);
            }
            ++local;
        }
    }

    void accumulateInterfaceFlux(unsigned l,bool fine,double weight)
    {
        auto& grid=level(l+unsigned(fine));
        unsigned local=0;
        for(auto index : grid.layout)
        {
            for(int d=0;d<DIM;++d)
                if(fine) m_registers[l]->incrementFineRegister((*grid.flux[local])[d],index,d,weight);
                else m_registers[l]->incrementCoarseRegister((*grid.flux[local])[d],index,d,weight);
            ++local;
        }
    }

    void prepareAtTime(unsigned l,double time)
    {
        auto& grid=level(l);
        if(grid.timeInterpolated && grid.preparedTime==time) return;
        if(!grid.old || !(grid.stepDt>0)) throw std::runtime_error("AMR parent has no interpolation interval");
        const double end=grid.startTime+grid.stepDt;
        const double tolerance=32*std::numeric_limits<double>::epsilon()
            *std::max({1.0,abs(time),abs(grid.startTime),abs(end)});
        if(time<grid.startTime-tolerance || time>end+tolerance)
            throw std::runtime_error("AMR fine stage lies outside its parent time interval");
        const double alpha=std::clamp((time-grid.startTime)/grid.stepDt,0.0,1.0);
        if(alpha==0) amrCopyValid(*grid.old,grid.stage);
        // Dense RK2 output: the linear term uses the composite first-stage
        // derivative; the coarse RK stage difference supplies the curvature.
        // Interpolating the unrefluxed coarse endpoint would introduce an
        // O(dt) error at coarse-fine interfaces on a fixed spatial mesh.
        else for(auto index : grid.layout) for(auto p : grid.layout[index])
            for(int c=0;c<NUMCOMPS;++c)
                grid.stage[index](p,c)=(*grid.old)[index](p,c)
                    +alpha*grid.stepDt*(*grid.predictor)[index](p,c)
                    +0.5*alpha*alpha*grid.stepDt*(grid.rhs[index](p,c)-grid.first[index](p,c));
        if(l) prepareAtTime(l-1,time);
        prepare(grid,grid.stage,time,l ? &level(l-1) : nullptr);
        grid.preparedTime=time;grid.timeInterpolated=true;
    }

    void evaluateStage(unsigned l,int stage,double time)
    {
        auto& grid=level(l);
        const bool first=!stage && !grid.firstStageUsed;
        if(first && m_reuseInitial)
        {
            // Every extrapolation branch starts from the same hierarchy and
            // physical time. Reuse its initial derivative and reconciled faces;
            // accumulate their branch-specific time weights below as before.
            amrCopyValid(*grid.extrapolatedFirst,grid.rhs);
            for(unsigned local=0;local<grid.flux.size();++local)
                for(int d=0;d<DIM;++d) (*grid.initialFlux[local])[d].copyTo((*grid.flux[local])[d]);
        }
        else
        {
            amrCopyValid(grid.state,grid.stage);
            if(stage) grid.stage.increment(l+1<numLevels() ? *grid.predictor : grid.first,grid.stepDt);
            if(l)
            {
                if(!m_extrapolationSubsteps.empty())
                    prepareExtrapolatedRK2Stage(l-1,grid.startTime,grid.stepDt,stage);
                else prepareAtTime(l-1,time);
            }
            prepare(grid,grid.stage,time,l ? &level(l-1) : nullptr);
            evaluate(grid);
            if(first && m_captureInitial)
            {
                if(!grid.extrapolatedFirst)
                {
                    grid.extrapolatedFirst=std::make_unique<Level>(grid.layout,Point::Zeros());
                    for(unsigned local=0;local<grid.flux.size();++local)
                    {
                        auto faces=std::make_unique<Array<State,DIM>>();
                        for(int d=0;d<DIM;++d) (*faces)[d].define((*grid.flux[local])[d].box());
                        grid.initialFlux.push_back(std::move(faces));
                    }
                }
                amrCopyValid(grid.rhs,*grid.extrapolatedFirst);
                for(unsigned local=0;local<grid.flux.size();++local)
                    for(int d=0;d<DIM;++d) (*grid.flux[local])[d].copyTo((*grid.initialFlux[local])[d]);
            }
        }
        accumulateFlux(grid,0.5*grid.stepDt);
        if(!stage)
        {
            amrCopyValid(grid.rhs,grid.first);
            grid.firstStageUsed=true;
            // Children can reuse this preparation at the start of the step.
            grid.preparedTime=time;grid.timeInterpolated=true;
            grid.preparedChildStage=-1;
        }
    }

    void beginStep(unsigned l,double dt,double time)
    {
        PR_TIMERS("HelioCubed::AMR::beginStep");
        if(!(dt>0) || !(time+dt>time)) throw std::runtime_error("AMR substep cannot advance floating-point time");
        auto& grid=level(l);
        grid.startTime=time;grid.stepDt=dt;grid.timeInterpolated=false;
        recordLevelStep(l,dt,time);
        if(l+1<numLevels())
        {
            if(!grid.old) grid.old=std::make_unique<Level>(grid.layout,Point::Zeros());
            if(!grid.predictor) grid.predictor=std::make_unique<Level>(grid.layout,Point::Zeros());
            amrCopyValid(grid.state,*grid.old);
            auto& reg=*m_registers[l];
            reg.initCoarseRegisters();reg.initFineRegisters();reg.initIncrement();
            const double fraction=dt/m_effectiveDt[l+1];
            const double roundoff=32*std::numeric_limits<double>::epsilon()*std::max(1.0,fraction);
            if(!std::isfinite(fraction) || fraction>maxRatio(m_ratios[l])+roundoff)
                throw std::runtime_error("AMR parent timestep exceeds the fine-level CFL bound");
            grid.substeps=m_extrapolationSubsteps.empty() ? std::max(1,int(ceil(fraction-roundoff))) : m_extrapolationSubsteps[l];
        }
        evaluateStage(l,0,time);
        if(l+1<numLevels())
        {
            beginStep(l+1,dt/grid.substeps,time);
            // The coarse predictor must use the same instantaneous composite
            // derivative as the fine mesh. Keep the coarse derivative in first
            // for the conservative RK update and its time-integrated reflux.
            amrCopyValid(grid.first,*grid.predictor);
            auto& child=level(l+1);
            auto& reg=*m_registers[l];
            accumulateInterfaceFlux(l,true,0.5*(dt-child.stepDt));
            applyReflux(l,*grid.predictor,2.0/dt,time,"predictor");
            restrictTo(l+1,l+2<numLevels() ? *child.predictor : child.first,*grid.predictor);
            reg.initCoarseRegisters();reg.initFineRegisters();reg.initIncrement();
            accumulateInterfaceFlux(l,false,0.5*dt);
            accumulateInterfaceFlux(l,true,0.5*child.stepDt);
        }
    }

    void completeStep(unsigned l)
    {
        PR_TIMERS("HelioCubed::AMR::completeStep");
        auto& grid=level(l);
        evaluateStage(l,1,grid.startTime+grid.stepDt);
        grid.state.increment(grid.first,0.5*grid.stepDt);
        grid.state.increment(grid.rhs,0.5*grid.stepDt);
        if(l+1<numLevels())
        {
            completeStep(l+1);
            const double fineDt=grid.stepDt/grid.substeps;
            for(int n=1;n<grid.substeps;++n)
            {
                beginStep(l+1,fineDt,grid.startTime+n*fineDt);
                completeStep(l+1);
            }
            // Both registers contain RK-weighted time integrals. Synchronize
            // this interface only after every descendant reaches the endpoint.
            applyReflux(l,grid.state,1.0,grid.startTime+grid.stepDt,"synchronize");
            restrictTo(l+1,level(l+1).state,grid.state);
        }
        grid.timeInterpolated=false;
    }

    void advanceSynchronous(double dt,double time)
    {
        for(unsigned l=0;l<numLevels();++l) recordLevelStep(l,dt,time);
        for(int stage=0;stage<2;++stage)
        {
            for(auto* grid : levels())
            {
                amrCopyValid(grid->state,grid->stage);
                if(stage) grid->stage.increment(grid->first,dt);
            }
            for(unsigned l=numLevels()-1;l>0;--l)
                restrictTo(l,level(l).stage,level(l-1).stage);
            for(unsigned l=0;l<numLevels();++l)
                prepare(level(l),level(l).stage,time+stage*dt,l ? &level(l-1) : nullptr);
            for(auto* grid : levels()) evaluate(*grid,true);
            for(auto& reg : m_registers)
            {reg->initCoarseRegisters();reg->initFineRegisters();reg->initIncrement();}
            for(auto* grid : levels()) accumulateFlux(*grid,1.0);
            for(unsigned l=numLevels()-1;l>0;--l)
            {
                applyReflux(l-1,level(l-1).rhs,1.0,time+stage*dt,"stage",stage+1);
                restrictTo(l,level(l).rhs,level(l-1).rhs);
            }
            if(!stage) for(auto* grid : levels()) amrCopyValid(grid->rhs,grid->first);
        }
        for(auto* grid : levels())
        {
            grid->state.increment(grid->first,0.5*dt);
            grid->state.increment(grid->rhs,0.5*dt);
        }
        averageDown();
    }

    void prepareStates(double time)
    {
        for(unsigned l=0;l<numLevels();++l)
            prepare(level(l),level(l).state,time,l ? &level(l-1) : nullptr);
    }

    void writeGridSummary(const char* event,double time,double seconds=-1,bool changed=false) const
    {
        if(procID()!=0) return;
        auto coverage=[&](unsigned l)
        {
            // Normalize valid fine cells by the full logical domain at that
            // level. Nested levels overlap, so overall refinement is L1 alone.
            long double cells=level(l).layout.numBoxes(),total=6;
            const Point sizes=m_domains[l].getBlock(0).sizes();
            for(int d=0;d<DIM;++d) {cells*=m_boxSizes[l][d];total*=sizes[d];}
            return 100*cells/total;
        };
        std::ostringstream lines;
        lines<<"AMR "<<event<<": levels="<<numLevels();
        for(const auto& grid : m_levels)
        {
            const auto size=m_boxSizes[grid->number];
            lines<<", L"<<grid->number<<" patches="<<grid->layout.numBoxes()
                <<" ["<<size[0]<<'x'<<size[1]<<'x'<<size[2]<<']';
        }
        if(seconds>=0) lines<<" changed = "<<(changed ? "yes" : "no");
        lines<<" time = "<<formatProgressTime(time);
        // Wall times are root-local, like iteration timing; no added barriers.
        if(seconds>=0) lines<<formatProgressElapsed(seconds);
        lines<<'\n'<<std::fixed<<std::setprecision(2)
            <<"AMR base grid refined = "<<(numLevels()>1 ? coverage(1) : 0.0L)<<'%';
        if(numLevels()>2)
        {
            lines<<" (";
            for(unsigned l=1;l<numLevels();++l)
                lines<<(l>1 ? ", " : "")<<'L'<<l<<" = "<<coverage(l)<<'%';
            lines<<')';
        }
        lines<<'\n';
        pout(0)<<lines.str();
    }

    void startProgress(unsigned passes)
    {
        if(!ParseInputs::get_amr_verbosity() || procID()!=0) return;
        m_progress.assign(numLevels(),LevelProgress{});m_progressPasses=passes;
    }

    void recordLevelStep(unsigned l,double dt,double time)
    {
        if(!ParseInputs::get_amr_verbosity() || procID()!=0) return;
        auto& progress=m_progress[l];++progress.steps;
        progress.minDt=std::min(progress.minDt,dt);progress.maxDt=std::max(progress.maxDt,dt);
        if(ParseInputs::get_amr_verbosity()<2) return;
        std::ostringstream line;
        line<<"AMR advance level = "<<l<<" dt = "<<formatProgressDt(dt)
            <<" time = "<<formatProgressTime(time)<<" target_time = "<<formatProgressTime(time+dt)<<'\n';
        pout(0)<<line.str();
    }

    void applyReflux(unsigned coarse,Level& destination,double scale,double time,
        const char* phase,int stage=0)
    {
        const bool report=ParseInputs::get_amr_verbosity() && procID()==0;
        m_registers[coarse]->applyRefluxCorrection(destination,scale);
        if(!report) return;
        ++m_progress[coarse].refluxCalls;
        if(ParseInputs::get_amr_verbosity()<2) return;
        std::ostringstream line;
        line<<"AMR reflux coarse_level = "<<coarse<<" fine_level = "<<coarse+1
            <<" phase = "<<phase;
        if(stage) line<<" stage = "<<stage;
        line<<" time = "<<formatProgressTime(time)<<'\n';
        pout(0)<<line.str();
    }

    void writeProgress(double time) const
    {
        if(!ParseInputs::get_amr_verbosity() || procID()!=0) return;
        std::ostringstream lines;
        for(unsigned l=0;l<m_progress.size();++l)
        {
            const auto& progress=m_progress[l];
            lines<<"AMR level = "<<l<<" steps = "<<progress.steps;
            if(progress.minDt==progress.maxDt) lines<<" dt = "<<formatProgressDt(progress.minDt);
            else lines<<" dt_min = "<<formatProgressDt(progress.minDt)<<" dt_max = "<<formatProgressDt(progress.maxDt);
            if(m_progressPasses>1) lines<<" extrapolation_passes = "<<m_progressPasses;
            lines<<" time = "<<formatProgressTime(time)<<'\n';
            if(progress.refluxCalls)
            {
                lines<<"AMR reflux coarse_level = "<<l<<" fine_level = "<<l+1
                    <<" calls = "<<progress.refluxCalls<<" time = "<<formatProgressTime(time)<<'\n';
            }
        }
        pout(0)<<lines.str();
    }
    static void trace(const char* message)
    {
#ifdef HELIOCUBED_AMR_TRACE
        std::cerr<<"AMR rank "<<procID()<<": "<<message<<std::endl;
#endif
    }

    void read(const std::string& filename)
    {
        PR_TIMERS("HelioCubed::AMR::restart");
        std::ifstream file(filename);
        std::string magic;int version,number;
        Point sizes,patchSize;
        double inner,outer,stretch,gamma;int problem;
        if(!(file>>magic>>version>>restartIteration>>restartTime>>restartDt>>number)
            || magic!="HelioCubedAMR" || (version<1 || version>5)
            || number<1 || unsigned(number)>m_ratios.size()+1 || (version==1 && number>2)
            || restartIteration<0 || !std::isfinite(restartTime) || restartTime<0
            || !std::isfinite(restartDt) || restartDt<0)
            throw std::invalid_argument("Invalid AMR checkpoint manifest or configured maximum level");
        if(version>=2)
        {
            int maximum;file>>maximum;
            if(!file || maximum<0 || unsigned(maximum)!=m_ratios.size())
                throw std::invalid_argument("AMR restart maximum level differs from checkpoint");
        }
        for(int d=0;d<DIM;++d) file>>sizes[d];
        for(int d=0;d<DIM;++d) file>>patchSize[d];
        file>>inner>>outer>>stretch>>gamma>>problem;
        if(!file || sizes!=m_domain.getBlock(0).sizes() || patchSize!=m_boxSize
            || inner!=ParseInputs::get_inner_radius() || outer!=ParseInputs::get_outer_radius()
            || stretch!=ParseInputs::get_radial_stretch() || gamma!=ParseInputs::get_gamma()
            || problem!=ParseInputs::get_init_condition_type())
            throw std::invalid_argument("AMR restart geometry or physical model differs from checkpoint");
        if(version>=2)
            for(auto expected : m_ratios)
            {
                Point ratio;file>>ratio[0]>>ratio[1]>>ratio[2];
                if(!file || ratio!=expected) throw std::invalid_argument("AMR restart refinement ratios differ from checkpoint");
            }
        else if(number==2 && m_ratios[0]!=Point::Ones(2))
            throw std::invalid_argument("Version-one AMR checkpoints require ratio two at the first interface");
        const auto requestedRadial=ratioList(ParseInputs::get_amr_boxSize_rad(),m_ratios.size(),"amr_boxSize_rad",0);
        const auto requestedAngular=ratioList(ParseInputs::get_amr_boxSize_nonrad(),m_ratios.size(),"amr_boxSize_nonrad",0);
        for(unsigned l=1;l<m_boxSizes.size();++l)
        {
            const int radial=requestedRadial[requestedRadial.size()==1 ? 0 : l-1];
            const int angular=requestedAngular[requestedAngular.size()==1 ? 0 : l-1];
            if(version<4)
            {
                if(radial || angular)
                    throw std::invalid_argument("Older AMR checkpoints require automatic fine patch sizes (amr_boxSize_rad/nonrad=0)");
                continue;
            }
            Point saved;file>>saved[0]>>saved[1]>>saved[2];
            if(!file || saved[1]!=saved[2]) throw std::invalid_argument("Invalid AMR checkpoint fine patch sizes");
            const Point domain=m_domains[l].getBlock(0).sizes(),ratio=m_ratios[l-1];
            for(int d=0;d<DIM;++d)
            {
                const int requested=d==0 ? radial : angular;
                if(saved[d]<8 || saved[d]>domain[d] || saved[d]%ratio[d]
                    || (ratio[d]>1 && saved[d]/ratio[d]<2) || domain[d]%saved[d]
                    || (static_cast<long long>(m_boxSizes[l-1][d])*ratio[d])%saved[d]
                    || (requested && saved[d]!=requested))
                    throw std::invalid_argument("AMR restart fine patch sizes are invalid or differ from the inputs");
            }
            m_boxSizes[l]=saved;
        }
        if(version>=3)
        {
            int inserted;file>>inserted;
            if(!file || inserted<0 || (version<5 && inserted>1))
                throw std::invalid_argument("Invalid AMR checkpoint CME insertion state");
            if(version<5 && ParseInputs::get_CMEs().size()>1)
                throw std::invalid_argument("Legacy AMR checkpoint has no multiple-CME schedule");
            m_cmeSchedule.restore(inserted,restartTime);
        }
        // Earlier AMR versions could not insert a CME. A newly enabled source
        // must therefore still fire when loading one of their checkpoints.
        else m_cmeSchedule.setInserted(false);
        std::vector<PatchSet> desired(m_ratios.size()+1);
        for(unsigned l=1;l<unsigned(number);++l)
        {
            size_t count;file>>count;
            const Box patchDomain=m_domains[l].getBlock(0).box().coarsen(m_boxSizes[l]);
            if(!file || count==0 || count>6*patchDomain.size())
                throw std::invalid_argument("Invalid AMR checkpoint patch count");
            for(size_t n=0;n<count;++n)
            {
                int block;Point p;file>>block>>p[0]>>p[1]>>p[2];
                if(!file || block<0 || block>=6 || !patchDomain.containsPoint(p)
                    || !desired[l].insert({block,p[0],p[1],p[2]}).second)
                    throw std::invalid_argument("Invalid or duplicate AMR checkpoint patch");
            }
            MBDisjointBoxLayout layout(m_domains[l],patchVector(desired[l]),std::vector<Point>(6,m_boxSizes[l]));
            m_levels.push_back(std::make_unique<GridLevel>(layout,l,l<m_ratios.size(),m_ratios[l-1]));
        }
        if(version>=3)
        {
            std::string key;
            if(file>>key)
            {
                if(key!="convergence_dt" || !(file>>m_convergenceDt) || !std::isfinite(m_convergenceDt)
                    || (m_convergenceDt!=-1 && m_convergenceDt<=0))
                    throw std::invalid_argument("Invalid AMR checkpoint convergence timestep");
            }
        }
        auto nested=desired;enforceNesting(nested);
        if(nested!=desired) throw std::invalid_argument("AMR checkpoint hierarchy is not properly nested");
        rebuildRegisters();
        std::string prefix=filename;
        if(prefix.size()<4 || prefix.substr(prefix.size()-4)!=".amr")
            throw std::invalid_argument("restart_file must name the .amr checkpoint manifest");
        prefix.resize(prefix.size()-4);
        HDF5Handler h5;
        for(unsigned l=0;l<numLevels();++l)
        {
            h5.readMBLevel(level(l).state,prefix+"_L"+std::to_string(l));
            if(!OP::validConservedState(level(l).state)) throw std::runtime_error("Invalid AMR restart state");
        }
    }

    static std::vector<int> ratioList(std::string text,unsigned count,const char* name,int minimum=1)
    {
        std::replace(text.begin(),text.end(),',',' ');
        std::istringstream input(text);std::vector<int> values;std::string word;
        while(input>>word)
        {
            size_t used=0;long long value;
            const auto message=std::string(name)+(minimum ? " requires positive integer ratios" : " requires nonnegative integer patch sizes");
            try {value=std::stoll(word,&used);} catch(...) {throw std::invalid_argument(message);}
            if(used!=word.size() || value<minimum || value>std::numeric_limits<int>::max())
                throw std::invalid_argument(message);
            values.push_back(int(value));
        }
        if(values.empty() || (values.size()!=1 && values.size()!=count))
            throw std::invalid_argument(std::string(name)+" needs one value or one value per fine level");
        return values;
    }

    void configureHierarchy(bool restarting=false)
    {
        const unsigned maximum=ParseInputs::get_amr_max_level();
        const auto radial=ratioList(ParseInputs::get_amr_ref_ratio_rad(),maximum,"amr_ref_ratio_rad");
        const auto angular=ratioList(ParseInputs::get_amr_ref_ratio_nonrad(),maximum,"amr_ref_ratio_nonrad");
        const auto radialBoxes=ratioList(ParseInputs::get_amr_boxSize_rad(),maximum,"amr_boxSize_rad",0);
        const auto angularBoxes=ratioList(ParseInputs::get_amr_boxSize_nonrad(),maximum,"amr_boxSize_nonrad",0);
        m_domains.push_back(m_domain);m_boxSizes.push_back(m_boxSize);m_cumulative.push_back(Point::Ones());
        for(unsigned l=0;l<maximum;++l)
        {
            Point ratio=Point::Ones(angular[angular.size()==1 ? 0 : l]);
            ratio[0]=radial[radial.size()==1 ? 0 : l];
            if(maxRatio(ratio)==1) throw std::invalid_argument("Every AMR interface must refine at least one direction");
            Point sizes=m_domains.back().getBlock(0).sizes(),box,cumulative=m_cumulative.back();
            size_t count=1;
            for(int d=0;d<DIM;++d)
            {
                if(sizes[d]>(std::numeric_limits<int>::max()-32)/ratio[d])
                    throw std::invalid_argument("AMR domain exceeds the integer coordinate range");
                sizes[d]*=ratio[d];cumulative[d]*=ratio[d];
                if(count>size_t(std::numeric_limits<std::ptrdiff_t>::max())/6/size_t(sizes[d]))
                    throw std::invalid_argument("AMR logical domain exceeds the 64-bit indexing range");
                count*=sizes[d];
                const int requested=d==0 ? radialBoxes[radialBoxes.size()==1 ? 0 : l]
                                        : angularBoxes[angularBoxes.size()==1 ? 0 : l];
                long long width=requested ? requested
                    : static_cast<long long>(m_boxSize[d]/std::gcd(m_boxSize[d],ratio[d]))*ratio[d];
                if(!requested && ratio[d]>1 && width/ratio[d]<2)
                {
                    int divisor=2;
                    while(m_boxSizes.back()[d]%divisor) ++divisor;
                    width*=divisor;
                }
                // Avoid enormous patches when an odd base extent and a large
                // ratio have a large LCM. Any coarse-patch divisor gives an
                // aligned tile; prefer one close to the base patch size.
                const long long refinedParent=static_cast<long long>(m_boxSizes.back()[d])*ratio[d];
                if(!requested && (width>2LL*m_boxSize[d] || refinedParent%width))
                {
                    const int parentWidth=m_boxSizes.back()[d];
                    if(refinedParent%width) width=0;
                    auto consider=[&](int coarseCells)
                    {
                        const long long candidate=static_cast<long long>(coarseCells)*ratio[d];
                        if(candidate<8 || (ratio[d]>1 && coarseCells<2)) return;
                        if(!width || abs(candidate-m_boxSize[d])<abs(width-m_boxSize[d])) width=candidate;
                    };
                    for(int q=1;static_cast<long long>(q)*q<=parentWidth;++q)
                        if(parentWidth%q==0) {consider(q);consider(parentWidth/q);}
                }
                if(width<8 || width>std::numeric_limits<int>::max() || width>sizes[d]
                    || width%ratio[d] || (ratio[d]>1 && width/ratio[d]<2)
                    || sizes[d]%width || (!restarting && (static_cast<long long>(m_boxSizes.back()[d])*ratio[d])%width))
                    throw std::invalid_argument("AMR fine patch sizes must be at least 8 cells, aligned with the refinement ratio, and tile the domain and each refined parent patch");
                box[d]=int(width);
                if(sizes[d]/box[d]>(uint64_t(1)<<(64/DIM)))
                    throw std::invalid_argument("AMR patch coordinates exceed the 64-bit Morton range");
            }
            // Native multiblock output packs six panels along the radial index
            // with one panel-width gap between them.
            if(sizes[0]>(std::numeric_limits<int>::max()-32)/12)
                throw std::invalid_argument("AMR radial domain exceeds the mapped-output integer range");
            m_ratios.push_back(ratio);m_domains.push_back(m_domains.back().refine(ratio));
            m_boxSizes.push_back(box);m_cumulative.push_back(cumulative);
        }
        std::string raw=ParseInputs::get_amr_refine_regions();
        std::replace(raw.begin(),raw.end(),',',' ');
        std::istringstream entries(raw);std::string entry;
        while(std::getline(entries,entry,';'))
        {
            if(entry.find_first_not_of(" \t")==std::string::npos) continue;
            std::istringstream row(entry);std::vector<int> values;std::string token;
            while(row>>token)
            {
                size_t used=0;long long value;
                try {value=std::stoll(token,&used);} catch(...)
                {throw std::invalid_argument("AMR region indices must be integers");}
                if(used!=token.size() || value<std::numeric_limits<int>::min() || value>std::numeric_limits<int>::max())
                    throw std::invalid_argument("AMR region index exceeds integer range or is not an integer");
                values.push_back(int(value));
            }
            if(!row.eof() || (values.size()!=7 && values.size()!=8))
                throw std::invalid_argument("AMR regions require [level,]block,ilo,jlo,klo,ihi,jhi,khi");
            const unsigned shift=values.size()==8 ? 1 : 0;
            const int target=shift ? values[0] : 1,block=values[shift];
            if(target<1 || unsigned(target)>maximum || block< -1 || block>=6)
                throw std::invalid_argument("AMR region level or block is outside the configured hierarchy");
            Point low,high;
            for(int d=0;d<DIM;++d) {low[d]=values[shift+1+d];high[d]=values[shift+4+d];}
            const Box cells=Box(low,high).refine(m_regionScale);
            if(cells.empty() || !m_domains[target-1].getBlock(0).box().containsBox(cells))
                throw std::invalid_argument("AMR region lies outside its parent-level domain");
            m_regions.push_back({unsigned(target),block,cells});
        }
    }

    std::vector<GridLevel*> levels() const
    {
        std::vector<GridLevel*> result;
        for(const auto& grid : m_levels) result.push_back(grid.get());
        return result;
    }

    bool covered(unsigned l,Point p,unsigned block) const
    {
        return l+1<numLevels() && level(l+1).layout.find((p*m_ratios[l])/m_boxSizes[l+1],block)
            !=*level(l+1).layout.end();
    }

    void cacheCoveredCells()
    {
        for(unsigned l=0;l+1<numLevels();++l)
        {
            auto& grid=level(l);
            if(grid.coveredCells) continue;
            grid.coveredCells=std::make_unique<MBLevelBoxData<int,1,HOST>>(
                grid.layout,amrPatchGhosts(Point::Ones()));
            for(auto index : grid.layout)
                for(auto p : (*grid.coveredCells)[index].box())
                    (*grid.coveredCells)[index](p)=covered(l,p,grid.layout.block(index));
        }
    }

    static PatchKey patchKey(MBPoint patch)
    {return {int(patch.block),patch.point[0],patch.point[1],patch.point[2]};}

    static std::vector<MBPoint> patchVector(const PatchSet& patches)
    {
        std::vector<MBPoint> result;
        for(auto key : patches) result.emplace_back(Point(key[1],key[2],key[3]),key[0]);
        return result;
    }

    static bool samePatches(const PatchSet& patches,const MBDisjointBoxLayout& layout)
    {
        if(patches.size()!=layout.numBoxes()) return false;
        for(auto patch : layout.patches()) if(!patches.count(patchKey(patch))) return false;
        return true;
    }

    void initialize(GridLevel& level)
    {
        PR_TIMERS("HelioCubed::AMR::initialize");
        const bool fileBC=ParseInputs::get_init_condition_type()==0;
        const bool magnetic=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
        Level boundary;
        State unused;
        if (fileBC)
        {
            boundary.define(level.layout,Point::Basis(0)+level.primitive.ghost()[0]*Point(0,1,1));
            BC_global.BoxData_to_BC(boundary,level.map,0.0);
        }
        for (auto index : level.layout)
        {
            auto& op=level.op[index];
            const auto block=level.layout.block(index);
            const auto dx=op.dx();
            const Box box=level.volume[index].box();
            Scalar radius(box); Vector dr(box),area(box);
            op.radialMetrics(radius,dr,area,level.volume[index],box);
            auto cart=forall_p<double,DIM,HOST>(f_cubedSphereMap3,box,radius,dx,0.5,0.5,block);
            State w(box),ju;
            op.initialize(w,fileBC ? boundary[index] : unused,radius,cart,
                ParseInputs::get_gamma(),level.layout.domain().getBlock(block).sizes()[0],dx[2],block);
            op.primToCons(ju,w,level.volume[index],ParseInputs::get_gamma(),dx[2],block);
            if (magnetic)
            {
                State background(box),mapped;
                w.copyTo(background);
                forallInPlace([] PROTO_LAMBDA(Var<double,NUMCOMPS,HOST>& state)
                {state(iRHO)=c_TEST_RHO0;state(iVX)=state(iVY)=state(iVZ)=0;state(iP)=c_TEST_P0;},background);
                op.primToCons(mapped,background,level.volume[index],ParseInputs::get_gamma(),dx[2],block);
                for (auto p : level.layout[index])
                {ju(p,iE)-=mapped(p,iE);for(int d=0;d<DIM;++d) ju(p,iBX+d)-=mapped(p,iBX+d);}
            }
            auto det=CubedSphereShell::detA<double,HOST>(level.layout[index],dx[1],0);
            for (auto p : level.layout[index])
                for (int c=0;c<NUMCOMPS;++c)
                    level.state[index](p,c)=ju(p,c)*level.jacobian[index](p)/(level.volume[index](p)*det(p));
        }
    }

    void prepare(GridLevel& level,const Level& state,double time,GridLevel* parent=nullptr)
    {
        PR_TIMERS("HelioCubed::AMR::prepare");
        if(fourthOrder()) {prepareFourthOrder(level,state,time,parent);return;}
        OP::GeometryScope geometry(level.geometry);
        level.timeInterpolated=false;
        level.preparedState=&state;
        if (parent)
        {
            // Integrate a tensor quadratic over each fine cell. For offset x
            // from its parent's center and ratio r, q=x*x+(1/r^2-1)/12 and
            // the weights are ((q-x)/2,1-q,(q+x)/2). This reduces exactly to
            // (-s/8,1,s/8) at ratio two and identity in unrefined directions.
            // Interpolate JU and J together before recovering primitives.
            if(!level.coarseMapped2)
                level.coarseMapped2=std::make_unique<MBLevelBoxData<double,NUMCOMPS+1,HOST>>(
                    parent->layout,amrPatchGhosts(Point::Ones(4)));
            auto& mapped=*level.coarseMapped2;
            const bool magnetic=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
            const double b0=magnetic ? testProblemBackgroundBx(ParseInputs::get_init_condition_type()) : 0;
            const double gamma=ParseInputs::get_gamma();
            PROTO_ASSERT(parent->preparedState,"Coarse AMR stage was not prepared");
            const Point ratio=level.ratio;
            if(!level.coarseSource2)
            {
                const auto cfLayout=level.layout.coarsen(ratio);
                Point sourceGhost;
                for(int d=0;d<DIM;++d) sourceGhost[d]=std::max(4,(level.donor.ghost()[0][d]+ratio[d]-1)/ratio[d]+1);
                level.coarseSource2=std::make_unique<MBLevelBoxData<double,NUMCOMPS+1,HOST>>(
                    cfLayout,amrPatchGhosts(sourceGhost));
                amrSetValLocal(*level.coarseSource2,std::numeric_limits<double>::quiet_NaN());
                level.coarseCopy2=std::make_unique<AMRTransferCopier<NUMCOMPS+1,true>>(
                    mapped,*level.coarseSource2,false,true);
            }
            {
                PR_TIMERS("HelioCubed::AMR::prepareCoarseDonors");
                for(const auto& needed : level.coarseCopy2->sources())
                {
                    const auto index=needed.index;
                    auto& out=mapped[index];
                    const auto& w=parent->primitive[index];
                    const auto& jacobian=parent->jacobian[index];
                    const auto& state=(*parent->preparedState)[index];
                    const Box valid=parent->layout[index];
                    for(const auto& box : needed.boxes) for(auto p : box)
                    {
                        const double j=jacobian(p);
                        if(valid.containsPoint(p))
                        {
                            out(p,NUMCOMPS)=j;
                            for(int c=0;c<NUMCOMPS;++c) out(p,c)=state(p,c);
                            continue;
                        }
                        double ke=0,me=0;
                        out(p,NUMCOMPS)=j;
                        out(p,iRHO)=j*w(p,iRHO);
                        for(int d=0;d<DIM;++d)
                        {
                            out(p,iMOMX+d)=j*w(p,iRHO)*w(p,iVX+d);
                            out(p,iBX+d)=j*w(p,iBX+d);
                            ke+=0.5*w(p,iRHO)*w(p,iVX+d)*w(p,iVX+d);
                            me+=w(p,iBX+d)*w(p,iBX+d)/(8*M_PI);
                        }
                        out(p,iE)=j*((w(p,iP)-(magnetic ? c_TEST_P0 : 0))/(gamma-1)
                            +ke+me+b0*w(p,iBX)/(4*M_PI));
                    }
                }
            }
            auto& source=*level.coarseSource2;
            level.coarseCopy2->copy(mapped,source);
            amrSetValLocal(level.donor,0);
            if(level.secondOrderGhosts.empty())
                for(auto index : level.layout) level.secondOrderGhosts.emplace_back(level.donor[index].box(),ratio);
            unsigned local=0;
            for (auto index : level.layout)
            {
                const auto& src=source[index];
                auto& out=level.donor[index];
                const Box owned=level.layout[index];
                const auto& plan=level.secondOrderGhosts[local++];
                for (auto p : out.box())
                {
                    if(owned.containsPoint(p)) continue;
                    // Floor division is required for physical ghost indices.
                    Point q;const std::array<double,3>* weights[DIM];
                    for(int d=0;d<DIM;++d)
                    {
                        const auto& coordinate=plan.coordinates[d][p[d]-plan.target.low()[d]];
                        q[d]=coordinate.parent;weights[d]=&coordinate.weights;
                    }
                    // Extra angular donors on an unrefined angular grid may
                    // extend beyond the parent's panel ghosts. Interpolation
                    // there is enabled only for stencils with owned donors;
                    // the subsequent exchange supplies those values.
                    if(!std::isfinite(src(q,NUMCOMPS))) continue;
                    LocalFaceState<double> u,w;
                    for(int c=0;c<NUMCOMPS;++c) u(c)=0;
                    double j=0;
                    for(auto offset : plan.offsets)
                    {
                        double weight=1;
                        for(int d=0;d<DIM;++d) weight*=(*weights[d])[offset[d]+1];
                        if(weight==0) continue;
                        j+=weight*src(q+offset,NUMCOMPS);
                        for(int c=0;c<NUMCOMPS;++c) u(c)+=weight*src(q+offset,c);
                    }
                    bool valid=j>0 && std::isfinite(j);
                    for(int c=0;c<NUMCOMPS;++c) {u(c)/=j;valid&=std::isfinite(u(c));}
                    double thermal=u(iE)+(magnetic ? c_TEST_P0/(gamma-1) : 0);
                    if(u(iRHO)>0)
                        for(int d=0;d<DIM;++d) thermal-=0.5*u(iMOMX+d)*u(iMOMX+d)/u(iRHO)+u(iBX+d)*u(iBX+d)/(8*M_PI);
                    thermal-=b0*u(iBX)/(4*M_PI);
                    valid&=u(iRHO)>=RHO00 && thermal>=0 && std::isfinite(thermal);
                    if(!valid) for(int c=0;c<NUMCOMPS;++c) u(c)=src(q,c)/src(q,NUMCOMPS);
                    cartesianPrimitiveFromConserved(w,u,1.0,gamma,magnetic,b0);
                    for(int c=0;c<NUMCOMPS;++c) out(p,c)=w(c);
                }
            }
        }
        else for (auto index : level.layout) level.donor[index].setVal(0);
        const bool magnetic=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
        for (auto index : level.layout)
            forallInPlace([] PROTO_LAMBDA(Var<double,NUMCOMPS,HOST>& out,
                const Var<double,NUMCOMPS,HOST>& in,const Var<double,1,HOST>& j,
                double gamma,bool magnetic,double b0)
            {cartesianPrimitiveFromConserved(out,in,1/j(0),gamma,magnetic,b0);},
                level.layout[index],level.donor[index],state[index],level.jacobian[index],
                ParseInputs::get_gamma(),magnetic,testProblemBackgroundBx(ParseInputs::get_init_condition_type()));
        level.donor.exchange();
        for (auto index : level.layout) level.donor[index].copyTo(level.primitive[index]);
        if(!level.secondOrderSeam || !level.secondOrderSeam->matches(level.primitive,level.donor,level.interp))
            level.secondOrderSeam=std::make_unique<SecondOrderSeamPlan<double,NUMCOMPS>>(
                level.primitive,level.donor,level.interp);
        level.secondOrderSeam->apply();
        for (auto index : level.layout)
            forallInPlace([] PROTO_LAMBDA(Var<double,NUMCOMPS,HOST>& w)
            {w(iRHO)=std::max(w(iRHO),double(RHO00));w(iP)=std::max(w(iP),0.0);},level.primitive[index]);
        OP::fillRadialPrimitiveBoundaries(level.primitive,level.volume,time);
    }

    void evaluate(GridLevel& level,bool restrictCovered=false)
    {
        PR_TIMERS("HelioCubed::AMR::evaluate");
        // Synchronous stages restrict the fine RHS before any coarse update.
        // Retain all physical/panel and coarse-fine interface faces. Keep the
        // complete operator for magnetic-test equilibrium subtraction.
        const bool maskFaces=restrictCovered && !fourthOrder() && level.number+1<numLevels()
            && !isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
        if(maskFaces && level.activeFaces.empty())
        {
            cacheCoveredCells();
            unsigned local=0;
            for(auto index : level.layout)
            {
                auto mask=std::make_unique<OP::SecondOrderFaceMask>();
                const auto& covered=(*level.coveredCells)[index];
                const Box domain=level.layout.getBlock(level.layout.block(index)).domain().box();
                for(int d=0;d<DIM;++d)
                {
                    (*mask)[d].defineForOverwrite((*level.flux[local])[d].box());
                    for(auto p : (*mask)[d].box())
                    {
                        const auto left=p-Point::Basis(d);
                        (*mask)[d](p)=!(domain.containsPoint(p) && domain.containsPoint(left)
                            && covered(p) && covered(left));
                    }
                }
                level.activeFaces.push_back(std::move(mask));++local;
            }
        }
        amrSetValLocal(level.rhs,0);
        level.panelFlux.clear();
        unsigned local=0;
        for (auto index : level.layout)
        {
            auto& op=level.op[index];
            if (ParseInputs::get_Sun_gravity())
            {
                if(fourthOrder()) op.Gravity_Euler(level.rhs[index],level.primitive[index]);
                else op.gravitySecondOrderPrimitives(level.rhs[index],level.primitive[index]);
            }
            if(fourthOrder() && ParseInputs::get_spatial_operator()==2)
                for(int d=0;d<DIM;++d) (*level.flux[local])[d].setVal(0);
            op(level.rhs[index],*level.flux[local],level.primitive[index],1.0,1.0,-1,!fourthOrder(),
                maskFaces ? level.activeFaces[local].get() : nullptr);
            if(fourthOrder() && (ParseInputs::get_linear_viscosity() || ParseInputs::get_nonlinear_viscosity()))
            {
                Array<State,DIM> viscosity;
                for(int d=0;d<DIM;++d) viscosity[d].define(level.layout[index].extrude(d),0.0);
                op.viscFlux(level.rhs[index],viscosity,level.primitive[index],1.0);
                for(int d=0;d<DIM;++d) (*level.flux[local])[d]-=viscosity[d];
            }
            level.panelFlux.increment(*level.flux[local],index,1.0);
            ++local;
        }
        level.panelFlux.reflux(level.rhs);
        // Refluxing the RHS is only half of panel reconciliation: the AMR
        // register must also see the common flux, including at a panel/CF edge.
        local=0;
        for (auto index : level.layout)
        {
            for (const auto& boundary : level.panelFlux.bounds(index))
            {
                if (boundary.localDir.codim()!=1) continue;
                const auto face=boundary.localFace();
                const auto inverse=boundary.adjToLocal.inverse();
                const int sign=inverse(Point::Basis(face.dir)).dot(Point::Ones());
                const Point adjacentNormal=-inverse(Point::Basis(face.dir,face.side));
                const int dp=adjacentNormal.dot(adjacentNormal*adjacentNormal);
                const Point shift=dp*(2*face.side-1)==1 ? Point::Basis(face.dir) : Point::Zeros();
                const Box box=level.layout[index].extrude(face.dir).face(face.dir,face.side)&boundary.localData->box();
                for (auto p : box) for(int c=0;c<NUMCOMPS;++c)
                    (*level.flux[local])[face.dir](p,c)=0.5*((*boundary.localData)(p,c)+sign*(*boundary.adjData)(p-shift,c));
            }
            ++local;
        }
    }

    PatchSet gatherPatches(PatchSet local) const
    {
#ifdef PR_MPI
        if(local.size()>size_t(std::numeric_limits<int>::max()/4))
            throw std::runtime_error("Too many local AMR tags for an MPI message");
        std::vector<int> send;
        for(auto key : local) send.insert(send.end(),key.begin(),key.end());
        const int count=send.size();
        std::vector<int> counts(numProc()),offsets(numProc());
        MPI_Allgather(&count,1,MPI_INT,counts.data(),1,MPI_INT,MPI_COMM_WORLD);
        size_t total=0;
        for(unsigned p=0;p<counts.size();++p)
        {
            if(total+counts[p]>size_t(std::numeric_limits<int>::max()))
                throw std::runtime_error("Too many global AMR tags for an MPI message");
            offsets[p]=total;total+=counts[p];
        }
        std::vector<int> receive(total);
        MPI_Allgatherv(send.data(),count,MPI_INT,receive.data(),counts.data(),offsets.data(),MPI_INT,MPI_COMM_WORLD);
        local.clear();
        for(size_t n=0;n<total;n+=4) local.insert({receive[n],receive[n+1],receive[n+2],receive[n+3]});
#endif
        return local;
    }

    PatchSet taggedPatches(unsigned parent)
    {
        PR_TIMERS("HelioCubed::AMR::tag");
        PatchSet patches;
        const auto ratio=m_ratios[parent],boxSize=m_boxSizes[parent+1];
        const Box domain=m_domains[parent].getBlock(0).box();
        auto tag=[&](const Box& cells,unsigned block)
        {
            const Box region=cells.grow(ParseInputs::get_amr_buffer())&domain;
            if(region.empty()) return;
            for(auto patch : region.refine(ratio).coarsen(boxSize)) patches.insert({int(block),patch[0],patch[1],patch[2]});
        };
        const auto& bounds=ParseInputs::get_amr_refine_bounds();
        if(parent<numLevels() && ParseInputs::uses_amr_tagging())
        {
            auto& grid=level(parent);
            if(bounds.active() && grid.tagMasks.empty())
                for(auto index : grid.layout)
                    grid.tagMasks.emplace_back(grid.layout[index],domain.sizes(),grid.layout.block(index),
                        bounds,cubedSphereShellGeometry());
            // Each limit is the finest CHILD level that criterion can create.
            const std::array<double,4> thresholds={
                parent<unsigned(ParseInputs::get_amr_refine_density_maxlevel()) ? ParseInputs::get_amr_refine_density_thresh() : -1,
                parent<unsigned(ParseInputs::get_amr_refine_pressure_maxlevel()) ? ParseInputs::get_amr_refine_pressure_thresh() : -1,
                parent<unsigned(ParseInputs::get_amr_refine_B_maxlevel()) ? ParseInputs::get_amr_refine_B_thresh() : -1,
                parent<unsigned(ParseInputs::get_amr_refine_V_maxlevel()) ? ParseInputs::get_amr_refine_V_thresh() : -1};
            // Constant-field tests evolve the perturbation; tag the total field.
            const double b0=testProblemBackgroundBx(ParseInputs::get_init_condition_type());
            unsigned local=0;
            for(auto index : grid.layout)
            {
                const auto* mask=bounds.active() ? &grid.tagMasks[local++] : nullptr;
                if(mask && mask->empty()) continue;
                State fourth;
                if(fourthOrder()) fourth=fourthOrderPrimitives(grid,index);
                const auto& w=fourthOrder() ? fourth : grid.primitive[index];
                for(auto p : grid.layout[index])
                {
                    if(mask && !mask->contains(p)) continue;
                    bool refine=false;
                    for(unsigned variable=0;variable<thresholds.size() && !refine;++variable)
                    {
                        if(thresholds[variable]<0) continue;
                        auto sample=[&](Point cell)
                        {
                            if(variable==0) return w(cell,iRHO);
                            if(variable==1) return w(cell,iP);
                            if(variable==2) return std::hypot(w(cell,iBX)+b0,w(cell,iBY),w(cell,iBZ));
                            return std::hypot(w(cell,iVX),w(cell,iVY),w(cell,iVZ));
                        };
                        const double gradient=amrRelativeGradientMagnitude(p,domain,sample);
                        // Preserve zero's documented "any change" behavior.
                        if(gradient>0 && gradient>=thresholds[variable]) refine=true;
                    }
                    if(refine) tag(Box(p,p),grid.layout.block(index));
                }
            }
        }
        if(procID()==0) for(const auto& region : m_regions)
            if(region.level==parent+1)
                for(unsigned block=0;block<6;++block)
                    if(region.block<0 || unsigned(region.block)==block)
                    {
                        if(!bounds.active()) tag(region.cells,block);
                        else AMRRefinementMask(region.cells,domain.sizes(),block,bounds,cubedSphereShellGeometry())
                            .tagRegions([&](const Box& cells) {tag(cells,block);});
                    }
        return gatherPatches(std::move(patches));
    }

    static void insertMappedPatch(PatchSet& output,MBPoint patch,const MBProblemDomain& domain)
    {
        // Include both paths around a panel corner. Crossing angular faces can
        // lead to different valid neighbors there; all are legitimate donors.
        std::vector<MBPoint> pending{patch};PatchSet visited;
        while(!pending.empty())
        {
            const auto current=pending.back();pending.pop_back();
            if(!visited.insert(patchKey(current)).second) continue;
            const Box bounds=domain.getBlock(current.block).box();
            if(current.point[0]<bounds.low()[0] || current.point[0]>bounds.high()[0]) continue;
            if(bounds.containsPoint(current.point)) {output.insert(patchKey(current));continue;}
            for(int d=1;d<DIM;++d)
            {
                const int side=current.point[d]<bounds.low()[d] ? -1 : current.point[d]>bounds.high()[d] ? 1 : 0;
                if(!side) continue;
                for(const auto& face : domain.boundaries(current.block,Point::Basis(d,side)))
                    pending.emplace_back(domain.convertPoint(current.point,current.block,face.dstBlock,PR_CELL),face.dstBlock);
            }
        }
    }

    void enforceNesting(std::vector<PatchSet>& desired) const
    {
        for(unsigned child=m_ratios.size();child>1;--child)
        {
            const unsigned parent=child-1;
            const auto ratio=m_ratios[parent];
            const int ghost=fourthOrder() ? OP::ghost()[0]+ParseInputs::get_seam_pressure_correction()+1 : child==m_ratios.size() ? 2 : 4;
            Point support;
            for(int d=0;d<DIM;++d)
            {
                const int donors=ghost+(d ? 2 : 0);
                support[d]=ratio[d]==1 ? donors : (donors+ratio[d]-1)/ratio[d]+(fourthOrder() ? 2 : 1);
            }
            const auto patchDomain=m_domains[parent].coarsen(m_boxSizes[parent]);
            for(auto key : desired[child])
            {
                const Box footprint=Box(Point(key[1],key[2],key[3]),Point(key[1],key[2],key[3]))
                    .refine(m_boxSizes[child]).coarsen(ratio).grow(support).coarsen(m_boxSizes[parent]);
                for(auto patch : footprint) insertMappedPatch(desired[parent],MBPoint(patch,key[0]),patchDomain);
            }
        }
    }

    void resetTransfers()
    {
        for(auto* grid : levels())
        {
            grid->restrictionCopy.reset();grid->coarseCopy2.reset();grid->coarseCopy4.reset();
            grid->restricted.reset();grid->coarseMapped2.reset();
            grid->coarseSource2.reset();grid->coarseSource4.reset();
            grid->coveredCells.reset();grid->activeFaces.clear();
        }
        m_effectiveDt.clear();
    }

    void rebuildRegisters()
    {
        m_registers.clear();
        resetTransfers();
        for(unsigned l=0;l+1<numLevels();++l)
        {
            auto dx=level(l).layout.dx(0);
            m_registers.push_back(std::make_unique<Register>(level(l).layout,level(l+1).layout,m_ratios[l],dx));
        }
    }

    bool realizeHierarchy(const std::vector<PatchSet>& desired,double time,bool initial)
    {
        bool changed=false;
        for(unsigned l=1;l<desired.size();++l)
        {
            if(desired[l].empty())
            {
                if(numLevels()>l)
                {
                    if(!changed) resetTransfers();
                    m_registers.clear();m_levels.resize(l);changed=true;
                }
                break;
            }
            if(l<numLevels() && samePatches(desired[l],level(l).layout)) continue;
            // Later replacements prepare retained ancestors. Invalidate their
            // parent-dependent routes before any earlier parent is replaced.
            if(!changed) resetTransfers();
            m_registers.clear();
            MBDisjointBoxLayout layout(m_domains[l],patchVector(desired[l]),std::vector<Point>(6,m_boxSizes[l]));
            auto replacement=std::make_unique<GridLevel>(layout,l,l<m_ratios.size(),m_ratios[l-1]);
            if(initial) initialize(*replacement);
            else
            {
                // Ancestor data must already be prepared at this synchronized
                // time, including when an intermediate sparse level changes.
                for(unsigned p=0;p<l;++p)
                    prepare(level(p),level(p).state,time,p ? &level(p-1) : nullptr);
                prolong(*replacement,level(l-1),m_ratios[l-1]);
            }
            if(l<numLevels())
            {
                level(l).state.copyTo(replacement->state);
                m_levels[l]=std::move(replacement);
            }
            else m_levels.push_back(std::move(replacement));
            changed=true;
        }
        if(changed) rebuildRegisters();
        return changed;
    }

    void prolong(GridLevel& target,GridLevel& parentGrid,Point ratio)
    {
        PR_TIMERS("HelioCubed::AMR::prolong");
        if(fourthOrder()) {prolongFourthOrder(target,parentGrid,ratio);return;}
        // Prolong physical Cartesian conserved values, then subtract the
        // volume-weighted child offset. This conserves every parent integral
        // and preserves constant physical states on the mapped grid.
        Level physical(parentGrid.layout,amrPatchGhosts(Point::Ones(2)));
        for (auto index : parentGrid.layout)
            for(auto p : physical[index].box())
            {
                const auto& w=parentGrid.primitive[index];
                double rho=w(p,iRHO),ke=0,me=0;
                const bool magnetic=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
                const double b0=magnetic ? testProblemBackgroundBx(ParseInputs::get_init_condition_type()) : 0;
                physical[index](p,iRHO)=rho;
                for(int d=0;d<DIM;++d)
                {
                    physical[index](p,iMOMX+d)=rho*w(p,iVX+d);
                    physical[index](p,iBX+d)=w(p,iBX+d);
                    ke+=0.5*rho*w(p,iVX+d)*w(p,iVX+d);
                    me+=w(p,iBX+d)*w(p,iBX+d)/(8*M_PI);
                }
                physical[index](p,iE)=(w(p,iP)-(magnetic ? c_TEST_P0 : 0))/(ParseInputs::get_gamma()-1)+ke+me+b0*w(p,iBX)/(4*M_PI);
            }
        // Owned values come directly from JU; primitive floors must never
        // change the conserved parent that is being split.
        for(auto index : parentGrid.layout) for(auto p : parentGrid.layout[index])
            for(int c=0;c<NUMCOMPS;++c) physical[index](p,c)=parentGrid.state[index](p,c)/parentGrid.jacobian[index](p);
        const auto cfLayout=target.layout.coarsen(ratio);
        Level source(cfLayout,amrPatchGhosts(Point::Ones(1)));
        physical.copyToFull(source);
        ScalarLevel parentJacobian(cfLayout,Point::Zeros());
        parentGrid.jacobian.copyTo(parentJacobian);
        int invalid=0;
        for(auto index : target.layout)
        {
            for(auto parent : cfLayout[index])
            {
                const Box children=Box(parent,parent).refine(ratio);
                Array<double,DIM> mean;mean.fill(0);double sumJ=0;
                for(auto p : children)
                {
                    const double j=target.jacobian[index](p);sumJ+=j;
                    for(int d=0;d<DIM;++d) mean[d]+=j*((p[d]+0.5)/ratio[d]-(parent[d]+0.5));
                }
                for(int d=0;d<DIM;++d) mean[d]/=sumJ;
                // Remove the floating-point nonadditivity of independently
                // evaluated analytic volumes, including on very fine grids.
                const double normalization=double(ratio[0])*ratio[1]*ratio[2]*parentJacobian[index](parent)/sumJ;
                for(int c=0;c<NUMCOMPS;++c)
                {
                    double slope[DIM];
                    for(int d=0;d<DIM;++d) slope[d]=amrSlope(source[index](parent-Point::Basis(d),c),
                        source[index](parent,c),source[index](parent+Point::Basis(d),c));
                    for(auto p : children)
                    {
                        double u=normalization*source[index](parent,c);
                        for(int d=0;d<DIM;++d) u+=slope[d]*(((p[d]+0.5)/ratio[d]-(parent[d]+0.5))-mean[d]);
                        target.state[index](p,c)=target.jacobian[index](p)*u;
                    }
                }
                const bool magnetic=isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
                const double b0=magnetic ? testProblemBackgroundBx(ParseInputs::get_init_condition_type()) : 0;
                auto admissible=[&](double theta)
                {
                    for(auto p : children)
                    {
                        double u[NUMCOMPS];
                        for(int c=0;c<NUMCOMPS;++c) u[c]=normalization*source[index](parent,c)+theta*
                            (target.state[index](p,c)/target.jacobian[index](p)-normalization*source[index](parent,c));
                        if(theta==0)
                        {
                            if(!amrAdmissibleParent(u,ParseInputs::get_gamma(),magnetic,b0)) return false;
                            continue;
                        }
                        if(!(u[iRHO]>=RHO00)) return false;
                        double thermal=u[iE]+(magnetic ? c_TEST_P0/(ParseInputs::get_gamma()-1) : 0);
                        for(int d=0;d<DIM;++d) thermal-=0.5*u[iMOMX+d]*u[iMOMX+d]/u[iRHO]+u[iBX+d]*u[iBX+d]/(8*M_PI);
                        thermal-=b0*u[iBX]/(4*M_PI);
                        if(!std::isfinite(thermal) || thermal<0) return false;
                    }
                    return true;
                };
                if(!admissible(1))
                {
                    if(!admissible(0)) {invalid=1;continue;}
                    double lo=0,hi=1;
                    for(int n=0;n<48;++n)
                    {const double theta=0.5*(lo+hi);if(admissible(theta)) lo=theta;else hi=theta;}
                    for(auto p : children) for(int c=0;c<NUMCOMPS;++c)
                    {
                        const double j=target.jacobian[index](p),center=normalization*source[index](parent,c);
                        target.state[index](p,c)=j*(center+lo*(target.state[index](p,c)/j-center));
                    }
                }
            }
        }
#ifdef PR_MPI
        MPI_Allreduce(MPI_IN_PLACE,&invalid,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
        if(invalid) throw std::runtime_error("Cannot conservatively refine a parent with invalid density or internal energy");
    }
};

#else
inline void runCubedSphereAMR()
{throw std::invalid_argument("Build with PR_AMR to enable cubed-sphere AMR");}
#endif
}

#ifdef PR_AMR
#include "MHD_AMRRungeKutta.H"
#include "MHD_AMRFourthOrder.H"
#include "MHD_AMRDiagnostics.H"
#include "MHD_AMRSlices.H"
namespace HelioCubed
{
inline bool CubedSphereAMR::insertCME(double time)
{
    if (!m_cmeSchedule.due(time)) return false;
    PR_TIMERS("HelioCubed::AMR::insertCME");
    while (m_cmeSchedule.due(time)) {
        const auto& source = ParseInputs::get_CMEs()[m_cmeSchedule.index()];
        prepareStates(time);
        long long affected = 0;
        for (auto* grid : levels()) for (auto index : grid->layout) {
            const Box valid = grid->layout[index];
            const Box box = fourthOrder() ? valid.grow(1) : valid;
            Scalar radius(box), volume(box);
            Vector dr(box), area(box);
            auto& op = grid->op[index];
            const auto block = grid->layout.block(index);
            op.radialMetrics(radius, dr, area, volume, box);
            auto position = forall_p<double,DIM,HOST>(f_cubedSphereMap3,
                box, radius, op.dx(), 0.5, 0.5, block);
            State delta(box, 0.0), fourth;
            if (fourthOrder()) fourth = fourthOrderPrimitives(*grid, index);
            const auto& primitive = fourthOrder() ? fourth : grid->primitive[index];
            const double backgroundBx = testProblemBackgroundBx(ParseInputs::get_init_condition_type());
            for (auto point : box) {
                CMEModel::Sample cme;
                if (!CMEModel::sample(source,
                    {position(point,0),position(point,1),position(point,2)}, cme)) continue;
                if (valid.containsPoint(point) && !covered(grid->number, point, block)) ++affected;
                std::array<double,8> ambient{};
                ambient[0] = primitive(point,iRHO);
                ambient[4] = primitive(point,iP);
                for (int d = 0; d < 3; ++d) {
                    ambient[1+d] = primitive(point,iVX+d);
                    ambient[5+d] = primitive(point,iBX+d) + (d == 0 ? backgroundBx : 0);
                }
                CMEModel::loadDensity(cme, ambient);
                const auto change = CMEModel::increment(ambient, cme,
                    ParseInputs::get_gamma(), source.plasmaBeta);
                for (int component = 0; component < 8; ++component)
                    delta(point,component) = change[component];
            }
            State average;
            if (fourthOrder())
                average = (1.0*Shift::Zeros() + (1.0/24)*Stencil<double>::Laplacian())(delta, valid);
            const auto& jacobian = grid->jacobian[index];
            for (auto point : valid) for (int component = 0; component < NUMCOMPS; ++component) {
                double increment = jacobian(point) *
                    (fourthOrder() ? average(point,component) : delta(point,component));
                if (fourthOrder()) for (int d = 0; d < DIM; ++d) {
                    const auto e = Point::Basis(d);
                    increment += (jacobian(point+e) - jacobian(point-e)) *
                        (delta(point+e,component) - delta(point-e,component)) / 48;
                }
                grid->state[index](point,component) += increment;
            }
        }
        averageDown();
        for (auto* grid : levels()) if (!OP::validConservedState(grid->state))
            throw std::runtime_error("Invalid conserved AMR state after CME insertion");
        m_cmeSchedule.markInserted();
        m_effectiveDt.clear();
#ifdef PR_MPI
        MPI_Allreduce(MPI_IN_PLACE, &affected, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        regrid(time);
        pout(0) << "CME" << source.sourceIndex << " inserted at simulation time "
            << std::setprecision(17) << time << " s" << std::endl;
        pout(0) << "AMR CME affected composite cells = " << affected << std::endl;
    }
    return true;
}
}

#include "MHD_AMRDriver.H"
#endif
