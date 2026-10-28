#pragma once

namespace HelioCubed
{
inline void writeAMRConvergenceProjection(CubedSphereAMR& simulation,int study,
    int iteration,double time,double dt)
{
    // Every fine level contributes conservatively to the full base grid.
    simulation.averageDown();
    auto& grid=simulation.level(0);
    CubedSphereAMR::Level projected(grid.layout,Point::Zeros());
    amrCopyValid(grid.state,projected);
    HDF5Handler h5;h5.setTime(time);h5.setTimestep(dt);h5.setIter(iteration);
    h5.writeMBLevel({},grid.map,projected,"U_conv_test_"+std::to_string(study));
}

inline void compareAMRConvergence(Point domain,Point box,Point ratio)
{
    using Level=CubedSphereAMR::Level;
    std::array<std::array<double,NUMCOMPS>,2> errors{};
    double comparisonTime=-1;
    pout(0)<<"AMR convergence: compare conservative composite projections on the base grids; "
        <<"exclude three radial boundary layers (mapped Cartesian JU maximum norm)."<<std::endl;
    for(int study=0;study<2;++study)
    {
        auto coarseDomain=CubedSphereShell::Domain(domain[1],domain[0],0);
        MBDisjointBoxLayout layout(coarseDomain,box);
        const Point fineDomain=domain*ratio;
        auto refinedDomain=CubedSphereShell::Domain(fineDomain[1],fineDomain[0],0);
        MBDisjointBoxLayout fineLayout(refinedDomain,box);
        Level coarse(layout,Point::Zeros()),fine(fineLayout,Point::Zeros());
        HDF5Handler h5;
        h5.readMBLevel(coarse,"U_conv_test_"+std::to_string(study));
        const double coarseTime=h5.time();
        h5.readMBLevel(fine,"U_conv_test_"+std::to_string(study+1));
        if(study==0) comparisonTime=coarseTime;
        const double tolerance=32*std::numeric_limits<double>::epsilon()*std::max(1.0,abs(comparisonTime));
        if(abs(coarseTime-comparisonTime)>tolerance || abs(h5.time()-comparisonTime)>tolerance)
            throw std::runtime_error("AMR convergence snapshots must have the same physical time");
        Level restricted(fineLayout.coarsen(ratio),Point::Zeros()),error(layout,Point::Zeros());
        auto average=Stencil<double>::AvgDown(ratio);
        for(auto index : fineLayout) restricted[index] |= average(fine[index]);
        restricted.copyTo(error);
        for(auto index : layout)
        {
            const auto physical=layout.domain().getBlock(layout.block(index)).box();
            for(auto p : layout[index]) for(int c=0;c<NUMCOMPS;++c)
            {
                const double value=p[0]<physical.low()[0]+3 || p[0]>physical.high()[0]-3
                    ? 0 : error[index](p,c)-coarse[index](p,c);
                error[index](p,c)=value;
                errors[study][c]=std::max(errors[study][c],abs(value));
            }
        }
#ifdef PR_MPI
        MPI_Allreduce(MPI_IN_PLACE,errors[study].data(),NUMCOMPS,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
        auto map=CubedSphereShell::Map(layout,Point::Ones());
        h5.setTime(comparisonTime);
        h5.writeMBLevel({},map,error,"Err"+std::to_string(study));
        for(int c=0;c<NUMCOMPS;++c)
            pout(0)<<"Lev = "<<study<<", component = "<<c<<", error = "<<std::setprecision(17)<<errors[study][c]<<std::endl;
        domain=fineDomain;
    }
    for(int c=0;c<NUMCOMPS;++c)
    {
        const double rate=errors[0][c]>0 && errors[1][c]>0
            ? log2(errors[0][c]/errors[1][c]) : std::numeric_limits<double>::quiet_NaN();
        pout(0)<<"order of accuracy for var "<<c<<" = "<<rate<<std::endl;
    }
}

inline void runCubedSphereAMR()
{
    using OP=CubedSphereAMR::OP;
    const bool fourth=ParseInputs::get_spatial_order()==4;
    const int order=ParseInputs::get_temporal_order(),mode=ParseInputs::get_convTestType();
    const int riemannSolver=ParseInputs::get_input_Riemann_solver_type();
    const int selected=ParseInputs::get_convergence_level();
    const double prescribed=ParseInputs::get_convergence_dt();
    const bool convergence=mode>=1 && mode<=3,split=selected>=0 && selected<=2;
    if((!fourth && ((order!=2 && order!=3) || !OP::usesSecondOrderPrimitives()))
        || (fourth && order!=3 && order!=4) || ParseInputs::get_boundary_reflux()!=1)
        throw std::invalid_argument("AMR requires temporal_order=2 or 3 at O2, temporal_order=3 or 4 at O4, and boundary_reflux=1; O2 requires no artificial viscosity");
    if(fourth && ParseInputs::get_state_conversion()!=1)
        throw std::invalid_argument("Fourth-order AMR requires state_conversion=1 (Cartesian conserved state)");
    if(mode<0 || mode>4 || selected< -1 || selected>3
        || (selected!=-1 && mode!=1 && mode!=2))
        throw std::invalid_argument("convTestType must be 0..4; convergence_level=-1, 0..2 (one study), or 3 (compare) applies to modes 1/2");
    if(!std::isfinite(prescribed) || (prescribed!=-1 && prescribed<=0))
        throw std::invalid_argument("convergence_dt must be -1 or a finite positive value");
    if(convergence && !split && !ParseInputs::get_restart_file().empty())
        throw std::invalid_argument("AMR convergence restart requires a single convergence_level=0, 1, or 2");
    if(ParseInputs::get_P_floor_cadence()<=0)
        throw std::invalid_argument("P_floor_cadence must be positive");
    const bool slices=ParseInputs::get_slice_X() || ParseInputs::get_slice_Y()
        || ParseInputs::get_slice_Z() || ParseInputs::get_slice_R();
    if(slices && (ParseInputs::get_slice_cadence()<=0
        || !std::isfinite(ParseInputs::get_slice_time_cadence()) || ParseInputs::get_slice_time_cadence()<=0))
        throw std::invalid_argument("AMR slice cadences must be finite and positive");
    if(ParseInputs::get_output_divB()!=0 && ParseInputs::get_output_divB()!=1)
        throw std::invalid_argument("output_divB must be 0 or 1");
    if(ParseInputs::get_write_cadence()<=0 || ParseInputs::get_checkpoint_cadence()<=0
        || !std::isfinite(ParseInputs::get_write_time_cadence()) || ParseInputs::get_write_time_cadence()<=0
        || !std::isfinite(ParseInputs::get_CFL()) || ParseInputs::get_CFL()<=0
        || !std::isfinite(ParseInputs::get_max_time()) || ParseInputs::get_max_time()<0
        || ParseInputs::get_max_iter()<0)
        throw std::invalid_argument("AMR requires positive output cadences/CFL and finite nonnegative stopping limits");
    if(ParseInputs::get_num_probes())
    {
        if(!std::isfinite(ParseInputs::get_Probe_cadence()) || ParseInputs::get_Probe_cadence()<=0)
            throw std::invalid_argument("Probe_cadence must be finite and positive");
        for(int p=0;p<ParseInputs::get_num_probes();++p)
            if(ParseInputs::get_probe_coords()[p].empty() || ParseInputs::get_probe_filenames()[p].empty())
                throw std::invalid_argument("Each AMR probe needs coords and filename");
    }
    Point box=Point::Ones(ParseInputs::get_boxSize_nonrad());box[0]=ParseInputs::get_boxSize_rad();
    Point domain=Point::Ones(ParseInputs::get_domainSize());domain[0]=ParseInputs::get_thickness();
    for(int d=0;d<DIM;++d)
        if(domain[d]<12 || box[d]<8 || domain[d]%box[d])
            throw std::invalid_argument("AMR domain sizes must be at least 12 and divisible by patch sizes of at least 8");
    if(fourth && domain[1]<20)
        throw std::invalid_argument("Fourth-order AMR requires at least 20 angular cells per panel");
    const Point studyRatio=ParseInputs::get_radial_refinement() ? Point(2,1,1) : Point::Ones(2);
    if(selected==3) {compareAMRConvergence(domain,box,studyRatio);return;}
    const int studies=convergence && !split ? 3 : 1;
    Point studyScale=Point::Ones();double nominal=prescribed,targetTime=-1;
    for(int run=0;run<studies;++run)
    {
        const int study=split ? selected : run;
        const Point sizes=domain*studyScale;
        const std::string suffix=convergence ? "_conv_"+std::to_string(study) : "";
        CubedSphereAMR simulation(CubedSphereShell::Domain(sizes[1],sizes[0],0),box,
            ParseInputs::get_restart_file(),studyScale,suffix);
        double time=simulation.restartTime,dt=simulation.restartDt;
        int iteration=simulation.restartIteration;
        bool giveSpace=true;
        const bool insertedAtStart=simulation.insertCME(time);
        auto writeMaps=[&](bool force)
        {
            if(ParseInputs::get_jmaps().empty()) return;
            std::vector<JmapGrid> grids;
            for(unsigned l=0;l<simulation.numLevels();++l)
            {
                const auto& grid=simulation.level(l);
                grids.push_back({&grid.state,&grid.jacobian,true});
            }
            writeJmaps(grids,iteration,time,force,suffix);
        };
        writeMaps(true);
        if(mode<3)
        {
            simulation.write(iteration,time,dt,false);
            if(insertedAtStart) simulation.write(iteration,time,dt,true);
            if(slices) simulation.writeSlices(iteration,time);
            if(insertedAtStart)
            {simulation.writeProbes(iteration,time,giveSpace);giveSpace=false;}
        }
        if(mode==3 || mode==4)
        {
            simulation.write(iteration,time,0,false);simulation.write(iteration,time,0,true);
            if(slices) simulation.writeSlices(iteration,time);
            simulation.writeProbes(iteration,time,giveSpace);
            if(mode==3) writeAMRConvergenceProjection(simulation,study,iteration,time,0);
        }
        else
        {
            if(convergence && nominal<0) nominal=simulation.convergenceDt()>0
                ? simulation.convergenceDt() : 0.2*simulation.timestep(time);
            const double fixedDt=convergence ? nominal/(mode==2 && !split ? (1<<run) : 1) : -1;
            simulation.setConvergenceDt(fixedDt);
            const long long maximumSteps=static_cast<long long>(ParseInputs::get_max_iter())
                *(mode==2 && !split ? (1<<run) : 1);
            if(maximumSteps>=std::numeric_limits<int>::max())
                throw std::invalid_argument("Convergence step count exceeds integer range");
            if(convergence && !split && targetTime<0)
                targetTime=std::min(ParseInputs::get_max_time(),ParseInputs::get_max_iter()*nominal);
            const double maximum=convergence && !split ? targetTime : ParseInputs::get_max_time();
            if(convergence) pout(0)<<"Convergence level = "<<study<<"\nConvergence timestep = "
                <<std::setprecision(17)<<fixedDt<<std::endl;
            // An event may split one nominal convergence step. Keep the common
            // end time while allowing that extra synchronized source insertion.
            const int lastIteration=int(maximumSteps)+(convergence && !split && ParseInputs::get_CME_type()==1 ? 1 : 0);
            double writeTrigger=floor(time/ParseInputs::get_write_time_cadence()),probeTrigger=-1;
            double sliceTrigger=floor(time/ParseInputs::get_slice_time_cadence());
            const double regridTimeCadence=ParseInputs::get_amr_regrid_time_cadence();
            double regridTimeTrigger=regridTimeCadence>0 ? floor(time/regridTimeCadence) : 0;
            int completed=iteration;
            for(++iteration;iteration<=lastIteration && time<maximum;++iteration)
            {
                const auto start=std::chrono::steady_clock::now();
                const bool floorDue=iteration%ParseInputs::get_P_floor_cadence()==0;
                if(floorDue)
                {
                    const auto repaired=simulation.applyFloors();
                    pout(0)<<"AMR pressure floor at iteration "<<iteration
                        <<": repaired composite cells = "<<repaired.pressure<<std::endl;
                    if(repaired.density)
                        pout(0)<<"AMR density floor at iteration "<<iteration
                            <<": repaired composite cells = "<<repaired.density<<std::endl;
                }
                const double cflDt=simulation.timestep(time);
                dt=std::min(convergence ? fixedDt : cflDt,maximum-time);
                const double rusanovEnd=simulation.cmeRusanovEndTime();
                const bool cmeRusanov=riemannSolver==2 && time<rusanovEnd;
                const bool reachesRusanovEnd=cmeRusanov && rusanovEnd-time<=dt;
                if(reachesRusanovEnd) dt=rusanovEnd-time;
                const double event=simulation.nextCMETime();
                const bool reachesCME=event>time && event-time<=dt;
                if(reachesCME) dt=event-time;
                if(!std::isfinite(dt) || !(dt>0) || !(time+dt>time))
                    throw std::runtime_error("Invalid AMR timestep");
                const int previousSolver=ParseInputs::get_Riemann_solver_type();
                ParseInputs::set_CME_Rusanov(cmeRusanov);
                if(ParseInputs::get_Riemann_solver_type()!=previousSolver)
                {
                    pout(0)<<"Riemann solver = "<<(cmeRusanov ? "Rusanov (CME insertion window)" : "Roe (input solver restored)")
                        <<" at time="<<std::setprecision(17)<<time;
                    if(cmeRusanov) pout(0)<<" s, until="<<rusanovEnd;
                    pout(0)<<" s"<<std::endl;
                }
                const auto repaired=simulation.advance(dt,time,floorDue);
                if(repaired.changed())
                    pout(0)<<"AMR density repair after iteration "<<iteration
                        <<": density cells = "<<repaired.density<<" pressure cells = "<<repaired.pressure<<std::endl;
                completed=iteration;
                time=reachesCME ? event : reachesRusanovEnd && dt==rusanovEnd-time ? rusanovEnd
                    : dt>=maximum-time ? maximum : time+dt;
                const bool inserted=simulation.insertCME(time);
                const int cadence=ParseInputs::get_amr_regrid_interval();
                const double nextRegridTimeTrigger=regridTimeCadence>0 ? floor(time/regridTimeCadence) : 0;
                const bool timeRegrid=regridTimeCadence>0
                    && (nextRegridTimeTrigger!=regridTimeTrigger || !std::isfinite(nextRegridTimeTrigger));
                // Regrid only synchronized states; crossing several time
                // intervals in one coarse step still needs just one regrid.
                if(!inserted && ((cadence && iteration%cadence==0) || timeRegrid)) simulation.regrid(time);
                regridTimeTrigger=nextRegridTimeTrigger;
                const bool last=iteration==lastIteration || time==maximum;
                const double nextTrigger=floor(time/ParseInputs::get_write_time_cadence());
                if(last || inserted || iteration%ParseInputs::get_write_cadence()==0 || nextTrigger!=writeTrigger)
                {simulation.write(iteration,time,dt,false);writeTrigger=nextTrigger;}
                if(last || inserted || iteration%ParseInputs::get_checkpoint_cadence()==0)
                    simulation.write(iteration,time,dt,true);
                if(slices)
                {
                    const double nextSlice=floor(time/ParseInputs::get_slice_time_cadence());
                    if(last || inserted || iteration%ParseInputs::get_slice_cadence()==0 || nextSlice!=sliceTrigger)
                    {simulation.writeSlices(iteration,time);sliceTrigger=nextSlice;}
                }
                if(ParseInputs::get_num_probes())
                {
                    const double nextProbe=floor(time/ParseInputs::get_Probe_cadence());
                    if(last || inserted || nextProbe!=probeTrigger)
                    {simulation.writeProbes(iteration,time,giveSpace);giveSpace=false;probeTrigger=nextProbe;}
                }
                writeMaps(last);
                const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                pout(0)<<formatIterationProgress(iteration,dt,time,seconds)<<std::endl;
                if(last) break;
            }
            if(convergence) writeAMRConvergenceProjection(simulation,study,
                completed,time,dt);
        }
        if(run+1<studies)
        {
            for(int d=0;d<DIM;++d)
                if(domain[d]>std::numeric_limits<int>::max()/studyScale[d]/studyRatio[d]/12)
                    throw std::invalid_argument("Convergence mesh exceeds mapped-output integer range");
            studyScale=studyScale*studyRatio;
        }
    }
    if(convergence && !split) compareAMRConvergence(domain,box,studyRatio);
}
}
