#include "Proto.H"
#include "InputParser.H"
#include "BoxOp_EulerCubedSphere.H"
#include "MHD_IO.H"
#include "MHD_CME.H"
#include "MHD_LevelRK.H"
#include "MHD_AMR.H"
#include <chrono> // Used by timer

MHDReader BC_global;

void setTimerReportFile(const std::string& stem)
{
  // Proto uses the supplied filename verbatim on every rank.
  std::string filename="/dev/null";
  if(procID()==0) filename=stem+".time.table";
  PR_TIMER_SETFILE(filename);
}

int main(int argc, char *argv[])
{
  #ifdef PR_MPI
    MPI_Init(&argc, &argv);
  #endif
  try
  {
    ParseInputs::getInstance().parsenow(argc, argv);
  }
  catch (const std::exception& error)
  {
    pout(0) << "Error: " << error.what() << endl;
#ifdef PR_MPI
    MPI_Finalize();
#endif
    return 1;
  }
  const int riemannSolver = ParseInputs::get_Riemann_solver_type();
  const double entropyFix = ParseInputs::get_entropy_fix_coeff();
  if ((riemannSolver != 1 && riemannSolver != 2)
      || !std::isfinite(entropyFix) || entropyFix < 0.0)
  {
    pout(0) << "Error: Riemann_solver_type must be 1 (Rusanov) or 2 (Roe 8-wave). "
            << "entropy_fix_coeff must be finite "
            << "and nonnegative." << endl;
#ifdef PR_MPI
    MPI_Finalize();
#endif
    return 1;
  }
  HDF5Handler h5;

  int domainSize = ParseInputs::get_domainSize();
  int thickness = ParseInputs::get_thickness();
  int boxSize_nonrad = ParseInputs::get_boxSize_nonrad();
  int boxSize_rad = ParseInputs::get_boxSize_rad();
  int max_iter = ParseInputs::get_max_iter();
  double max_time = ParseInputs::get_max_time();
  int temporal_order = ParseInputs::get_temporal_order();
  double gamma = ParseInputs::get_gamma();
  double dt = 0.0;
  double dt_next = 0.0;
  double time = 0.0;
  int restart_step = 0;
  int write_cadence = ParseInputs::get_write_cadence();
  int checkpoint_cadence = ParseInputs::get_checkpoint_cadence();
  int slice_cadence = ParseInputs::get_slice_cadence();
  int convTestType = ParseInputs::get_convTestType();
  int convergence_level = ParseInputs::get_convergence_level();
  double convergence_dt = ParseInputs::get_convergence_dt();
  int init_condition_type = ParseInputs::get_init_condition_type();
  int radial_refinement = ParseInputs::get_radial_refinement();
  int MBInterp_define = ParseInputs::get_MBInterp_define();
  string BC_file = ParseInputs::get_BC_file();
  string restart_file = ParseInputs::get_restart_file();
  if (init_condition_type == 0) BC_global.file_to_BoxData_vec(BC_file);
  if (ParseInputs::get_amr_max_level())
  {
    setTimerReportFile("CubedSphere_AMR");
    int status=0;
    try { HelioCubed::runCubedSphereAMR(); }
    catch (const std::exception& error)
    {
      pout(0)<<"Error: "<<error.what()<<endl;
      status=1;
    }
    PR_TIMER_REPORT();
#ifdef PR_MPI
    MPI_Finalize();
#endif
    return status;
  }
  
  double probe_cadence = ParseInputs::get_Probe_cadence();
  auto num_probes  = ParseInputs::get_num_probes();
  auto& probe_filenames = ParseInputs::get_probe_filenames();
  auto& probe_coords = ParseInputs::get_probe_coords();
  double write_time_cadence = ParseInputs::get_write_time_cadence();
  double slice_time_cadence = ParseInputs::get_slice_time_cadence();

  int radialDir = CUBED_SPHERE_SHELL_RADIAL_COORD;
  Array<double, DIM> offset = {0., 0., 0.};
  Array<double, DIM> exp = {1., 1., 1.};
  MBLevelBoxData<double, NUMCOMPS, HOST> U_conv_test[3]; 
  setTimerReportFile(to_string(thickness) + "_DIM" + to_string(DIM) +"_"+to_string(boxSize_nonrad) + "_" + to_string(boxSize_rad)
                   + "_CubeSphereTest");
  PR_TIMERS("MMB_MHD");
  int levmax = 3;
  auto domain =
    CubedSphereShell::Domain(domainSize, thickness, radialDir);
  bool split_convergence_level = (convergence_level >= 0 && convergence_level <= 2);
  bool compare_convergence_levels = (convergence_level == 3);

  if (convergence_level < -1 || convergence_level > 3)
    {
      pout(0) << "Error: convergence_level must be -1, 0, 1, 2, or 3." << endl;
#ifdef PR_MPI
      MPI_Finalize();
#endif
      return 1;
    }
  if (convergence_level != -1 && (convTestType < 1 || convTestType > 2))
    {
      pout(0) << "Error: split convergence levels require convTestType 1 or 2." << endl;
#ifdef PR_MPI
      MPI_Finalize();
#endif
      return 1;
    }
  if (convergence_dt != -1.0 && convergence_dt <= 0.0)
    {
      pout(0) << "Error: convergence_dt must be -1 or a positive value." << endl;
#ifdef PR_MPI
      MPI_Finalize();
#endif
      return 1;
    }

  if (convTestType == 0) levmax = 1;
  if (convTestType == 4) levmax = 1;
  if (split_convergence_level) levmax = 1;
  if (compare_convergence_levels)
    {
      levmax = 0;
      auto comparison_domain = domain;
      Point refRatio;
      if (radial_refinement)
        {
          refRatio = 2*Point::Ones() - Point::Basis(1) - Point::Basis(2);
        }
      else
        {
          refRatio = 2*Point::Ones();
        }

      Array<Point, DIM+1> zeroGhost;
      for (int d = 0; d < DIM + 1; ++d)
        {
          zeroGhost[d] = Point::Zeros();
        }
      for (int lev = 0; lev < 3; ++lev)
        {
          // The separately run levels use fixed-size patches so finer meshes
          // have more boxes and can use more MPI ranks. Read that layout first.
          Point fileBoxSize = Point::Ones(boxSize_nonrad);
          fileBoxSize[radialDir] = boxSize_rad;
          MBDisjointBoxLayout fileLayout(comparison_domain, fileBoxSize);
          MBLevelBoxData<double, NUMCOMPS, HOST> fileData(
              fileLayout, zeroGhost);
          h5.readMBLevel(fileData, "U_conv_test_" + to_string(lev));

          // Redistribute onto the nested layouts used by the original
          // convergence driver. A fine comparison patch then coarsens exactly
          // onto its corresponding patch at the preceding level, even when
          // the base patch size is odd.
          const int levelScale = 1 << lev;
          Point comparisonBoxSize =
              levelScale*Point::Ones(boxSize_nonrad);
          comparisonBoxSize[radialDir] = levelScale*boxSize_rad;
          MBDisjointBoxLayout comparisonLayout(
              comparison_domain, comparisonBoxSize);
          U_conv_test[lev].define(comparisonLayout, zeroGhost);
          fileData.copyTo(U_conv_test[lev]);
          pout(0) << "Loaded convergence level " << lev
                  << " using " << numProc() << " MPI processes" << endl;
          comparison_domain = comparison_domain.refine(refRatio);
        }
    }
  for (int lev=0; lev<levmax; lev++)
    {
      typedef BoxOp_EulerCubedSphere<double, MBMap_CubedSphereShell<HOST>, HOST> OP;
      
      Array<Array<uint, DIM>, 6> permute = {{2, 1, 0}, {2, 1, 0}, {1, 0, 2}, {0, 1, 2}, {1, 0, 2}, {0, 1, 2}};
      Array<Array<int, DIM>, 6> sign = {{-1, 1, 1}, {1, 1, -1}, {-1, 1, 1}, {1, 1, 1}, {1, -1, 1}, {-1, -1, 1}}; 
      Point boxSizeVect = Point::Ones(boxSize_nonrad);
      boxSizeVect[radialDir] = boxSize_rad;
      MBDisjointBoxLayout layout(domain, boxSizeVect);
      int count = 0;
      for (auto dit : layout)
        {
          count++;
        }
      pout(0) << "proc_id: " << procID() << ";      num boxes: " << count << std::endl;

      // initialize data and map
      auto& geometry = OP::levelGeometry(layout);
      auto& map = *geometry.map;
      MBLevelBoxData<double, NUMCOMPS, HOST> JU(layout, OP::ghost());
      MBLevelBoxData<double, 1, HOST> dVolrLev(layout, OP::ghost() + Point::Basis(0, 2));
      Array<double, DIM> dx;
      auto& eulerOp = *geometry.op;
      CMEInsertionSchedule cmeSchedule(ParseInputs::get_CMEs());
      double dxradius = 1.0 / thickness;
      auto C2C = Stencil<double>::CornersToCells(4);

      int rCoord = CUBED_SPHERE_SHELL_RADIAL_COORD;
      int thetaCoord = (rCoord + 1) % 3;
      int phiCoord = (rCoord + 2) % 3;

      MBLevelBoxData<double, 8, HOST> dstData;
      BoxData<double, NUMCOMPS, HOST> unusedBoundary;
      if (init_condition_type == 0)
      {
        const int boundaryGhost = ParseInputs::get_spatial_order() == 2
            ? OP::ghost().max() : NGHOST;
        dstData.define(layout, Point::Basis(rCoord) + boundaryGhost*Point::Basis(thetaCoord) + boundaryGhost*Point::Basis(phiCoord));
        BC_global.BoxData_to_BC(dstData, map, time);
      }

      MBInterpOp iop;
      Point interpSourceGhost = HelioCubed::interpolationSourceGhost(OP::ghost());
      iop = CubedSphereShell::MHDInterpOp<HOST>(JU.layout(), interpSourceGhost, 4);

      // Set input solution.
      for (auto dit : layout)
        {
          dx = eulerOp[dit].dx();
          BoxData<double> radius(dVolrLev[dit].box());
          BoxData<double, DIM, HOST> Dr(dVolrLev[dit].box());
          BoxData<double, DIM, HOST> adjDr(dVolrLev[dit].box());
          eulerOp[dit].radialMetrics(radius, Dr, adjDr, dVolrLev[dit], Dr.box());
        }
      if (restart_file.empty())
        {
        Reduction<double,Operation::Max,HOST> dtinv;
        dtinv.reset();
        for (auto dit : layout)
          {
            dx = eulerOp[dit].dx();
            BoxData<double> radius(dVolrLev[dit].box());
            BoxData<double, DIM, HOST> Dr(dVolrLev[dit].box());
            BoxData<double, DIM, HOST> adjDr(dVolrLev[dit].box());
            eulerOp[dit].radialMetrics(radius, Dr, adjDr, dVolrLev[dit], Dr.box());
            auto block = layout.block(dit);
            auto &JU_i = JU[dit];
            BoxData<double, NUMCOMPS, HOST> JUTemp;
            BoxData<double, NUMCOMPS, HOST> WPoint_i(JU_i.box());
            double half = 0.5;
            BoxData<double, DIM, HOST> XCart = forall_p<double,DIM,HOST>
              (f_cubedSphereMap3,radius.box(),radius,dx,half,half,block);  
            eulerOp[dit].initialize(WPoint_i, init_condition_type == 0 ? dstData[dit] : unusedBoundary, radius, XCart, gamma, thickness, dx[2], block);
            eulerOp[dit].dtInv(dtinv,WPoint_i);
            eulerOp[dit].primToCons(JUTemp, WPoint_i, dVolrLev[dit], gamma, dx[2], block);
            if (isConstantMagneticFieldTest(init_condition_type))
            {
              // Magnetized tests evolve perturbations about a uniform low-beta
              // MHD background. Subtract its mapped energy and magnetic field
              // so primitive recovery does not lose pressure to cancellation.
              BoxData<double,NUMCOMPS,HOST> WBackground(WPoint_i.box());
              WPoint_i.copyTo(WBackground);
              forallInPlace([] PROTO_LAMBDA(
                  Var<double,NUMCOMPS,HOST>& a_W)
                {
                  a_W(iRHO) = c_TEST_RHO0;
                  a_W(iVX) = 0.0;
                  a_W(iVY) = 0.0;
                  a_W(iVZ) = 0.0;
                  a_W(iP) = c_TEST_P0;
                },WBackground);
              BoxData<double,NUMCOMPS,HOST> JUBackground;
              eulerOp[dit].primToCons(
                  JUBackground,WBackground,dVolrLev[dit],gamma,dx[2],block);
              forallInPlace([] PROTO_LAMBDA(
                  Var<double,NUMCOMPS,HOST>& a_JU,
                  const Var<double,NUMCOMPS,HOST>& a_background)
                {
                  a_JU(iE) -= a_background(iE);
                  a_JU(iBX) -= a_background(iBX);
                  a_JU(iBY) -= a_background(iBY);
                  a_JU(iBZ) -= a_background(iBZ);
                },JUTemp,JUBackground);
            }
            JU_i.setVal(0.);
            JUTemp.copyTo(JU_i, layout[dit]);
          }
        } else {
          ReadCheckpointState(h5, JU, restart_file);
          time = h5.time();
		      dt = h5.dt();   
          restart_step = h5.iter();
          int inserted = ReadCheckpointCMEState(restart_file);
          cmeSchedule.restore(inserted,time);
          pout(0) << "Restarting from step " << restart_step << " at time " << time << " with dt " << dt << endl;
        }

      auto insertCMEIfDue = [&]() -> bool
      {
        if (!cmeSchedule.due(time)) return false;
        while (cmeSchedule.due(time)) {
          const auto& source=ParseInputs::get_CMEs()[cmeSchedule.index()];
          if(source.injectionMethod==0) OP::Insert_CME(JU,dVolrLev,iop,layout,time,dt,gamma,source);
          cmeSchedule.markInserted();
          pout(0) << "CME" << source.sourceIndex << " inserted at simulation time " << setprecision(17)
                  << time << " s (decimal year "
                  << source.enterTime << ", method "<<(source.injectionMethod ? "boundary" : "volume")<<")"
                  << setprecision(6) << endl;
          if (convTestType == 0)
            pout(0) << "CME CFL reset to " << cmeSchedule.cfl(time, ParseInputs::get_CFL())
                    << "; input target " << ParseInputs::get_CFL()
                    << ", ramp starts after 600 s and reaches target after 2400 s" << endl;
        }
        return true;
      };
      insertCMEIfDue();

      HelioCubed::MBLevelRK<BoxOp_EulerCubedSphere, MBMap_CubedSphereShell<HOST>, double> rk4(map, iop);
      // OP::P_floor(JU,dVolrLev,iop,layout,time,dt,gamma);
      Write_W(JU, eulerOp, iop, restart_step, time, dt, true, true);            
      // Write_W and preStageLevel fill their own boundary states. The former
      // throwaway JUTemp fill passed mapped JU as if it were primitive file
      // data and discarded the result.

    bool give_space_in_probe_file = true;
    int probe_trigger_count_temp = -1;
    int write_trigger_count_temp = -1;
    int slice_trigger_count_temp = -1;

    if (lev == 0)
      {
        double dtcfl1 = OP::dtCFL(JU,iop,dVolrLev);
        dt = 0.2*dtcfl1*ParseInputs::get_CFL();
      }
    if (split_convergence_level && convergence_dt > 0.0)
      {
        dt = convergence_dt;
      }
    if (convTestType == 1 || convTestType == 2)
      {
        int output_level = split_convergence_level ? convergence_level : lev;
        pout(0) << "Convergence level = " << output_level << endl;
        pout(0) << "Convergence timestep = " << setprecision(17) << dt
                << setprecision(6) << endl;
      }
    if (convTestType > 2) max_iter = 1;
    // A final step may be shortened to reach max_time. Preserve the nominal
    // convergence timestep for the next level when running levels in one launch.
    const double convergence_step_dt = dt;
    
    bool time_exceeded = false;
    for (int iter = restart_step + 1;
         iter <= max_iter && (convTestType >= 3 || time < max_time); iter++)
    {
      auto start = chrono::steady_clock::now();
      if ((iter % ParseInputs::get_P_floor_cadence()) == 0) OP::P_floor(JU,dVolrLev,iop,layout,time,dt,gamma);
      const double stepCFL = cmeSchedule.cfl(time, ParseInputs::get_CFL());
      if (convTestType == 0)
        {
          double dtcfl1 = OP::dtCFL(JU,iop,dVolrLev);
          dt = dtcfl1*stepCFL;
        }
      else if (convTestType == 1 || convTestType == 2)
        {
          dt = convergence_step_dt;
        }
      // Normal simulations and time-evolving convergence tests share the same
      // end-time limit. Convergence timesteps otherwise remain fixed.
      if (convTestType < 3 &&
          (!std::isfinite(dt) || !(dt > 0.0) || !(time + dt > time)))
        {
          pout(0) << "Error: invalid or non-advancing timestep at iteration "
                  << iter << ": time=" << time << ", dt=" << dt << endl;
#ifdef PR_MPI
          MPI_Finalize();
#endif
          return 1;
        }
      if (convTestType < 3 && dt >= max_time - time)
        {
          dt = max_time - time;
          time_exceeded = true;
        }
      // Keep one solver throughout every RK step, landing exactly on the
      // end of the ten-minute Rusanov window before restoring the input solver.
      const double cmeRusanovEnd = cmeSchedule.rusanovEndTime();
      const bool cmeRusanov = riemannSolver == 2 && time < cmeRusanovEnd;
      bool reachesRusanovEnd = convTestType < 3 && cmeRusanov
          && dt >= cmeRusanovEnd - time;
      if (reachesRusanovEnd)
        {
          dt = cmeRusanovEnd - time;
          time_exceeded = cmeRusanovEnd == max_time;
        }
      // Land on the event before advancing the CME. Recompute the normal CFL
      // timestep on the next iteration after the inserted state is available.
      bool reachesCME = convTestType < 3 && !cmeSchedule.inserted()
          && cmeSchedule.time() > time && dt >= cmeSchedule.time() - time;
      if (reachesCME)
        {
          dt = cmeSchedule.time() - time;
          time_exceeded = cmeSchedule.time() == max_time;
          reachesRusanovEnd = cmeSchedule.time() == cmeRusanovEnd;
        }
      bool cmeInsertedThisStep = false;
      if (convTestType < 3)
        {
          const int previousSolver = ParseInputs::get_Riemann_solver_type();
          ParseInputs::set_CME_Rusanov(cmeRusanov);
          if (ParseInputs::get_Riemann_solver_type() != previousSolver)
            {
              pout(0) << "Riemann solver = "
                      << (cmeRusanov ? "Rusanov (CME insertion window)" : "Roe (input solver restored)")
                      << " at time=" << setprecision(17) << time;
              if (cmeRusanov) pout(0) << " s, until=" << cmeRusanovEnd;
              pout(0) << " s" << setprecision(6) << endl;
            }
          rk4.advance(JU, dVolrLev, dt, time, temporal_order);
          time = reachesCME ? cmeSchedule.time()
                           : reachesRusanovEnd ? cmeRusanovEnd
                           : time_exceeded ? max_time : time + dt;
          cmeInsertedThisStep = insertCMEIfDue();
          if (!OP::validConservedState(JU))
            {
              pout(0) << "Error: nonfinite conserved state or nonpositive cell-average "
                      << "density after iteration " << iter << ", time=" << time << endl;
#ifdef PR_MPI
              MPI_Finalize();
#endif
              return 1;
            }
        }

      int write_time_cadence_new = floor(time/write_time_cadence);
      int slice_time_cadence_new = floor(time/slice_time_cadence);
      int write_trigger_count_now = floor(time/write_time_cadence);
      int slice_trigger_count_now = floor(time/slice_time_cadence);
      bool write_data = ((time > write_trigger_count_now * write_time_cadence - dt) && (time < write_trigger_count_now * write_time_cadence + dt) && (write_trigger_count_now != write_trigger_count_temp)) || (iter % write_cadence == 0) || (iter == max_iter) || time_exceeded || cmeInsertedThisStep;
      bool slice_data = ((time > slice_trigger_count_now * slice_time_cadence - dt) && (time < slice_trigger_count_now * slice_time_cadence + dt) && (slice_trigger_count_now != slice_trigger_count_temp)) || (iter % slice_cadence == 0) || (iter == max_iter) || time_exceeded;
      if (write_data || slice_data)
        {
          Write_W(JU, eulerOp, iop, iter, time, dt, write_data, slice_data);
          if (write_data) write_trigger_count_temp = write_trigger_count_now;
          if (slice_data) slice_trigger_count_temp = slice_trigger_count_now;
        }
      // Checkpointing.
      if ((iter % checkpoint_cadence == 0) || (iter == max_iter) || time_exceeded)
        {
          Write_Checkpoint(JU, iter, restart_step, time, dt, cmeSchedule.countInserted());
        }
      
      int probe_trigger_count_now = floor(time/probe_cadence);
      bool do_probes = (time > probe_trigger_count_now * probe_cadence - dt) && (time < probe_trigger_count_now * probe_cadence + dt) && (probe_trigger_count_now != probe_trigger_count_temp);
      if (do_probes || iter == 1 || time_exceeded){
        for (int p = 0; p < num_probes; ++p) {
          string probe_coord = probe_coords[static_cast<size_t>(p)];
          string probe_filename = probe_filenames[static_cast<size_t>(p)];
          Probe(JU, map, eulerOp, iop, iter, time, dx, give_space_in_probe_file,
                probe_coord, probe_filename, &dVolrLev);
        }
        give_space_in_probe_file = false;
        probe_trigger_count_temp = probe_trigger_count_now;
        pout(0) << "Probed at iteration " << iter << endl;
      }
      
      auto end = chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_seconds = end-start;
      pout(0) << formatIterationProgress(iter,dt,time,elapsed_seconds.count()) << endl;
      if (time_exceeded) {
        pout(0) << "Maximum simulation time reached. Ending simulation." << endl;
        break;
      }
    }

      if ((convTestType > 0) && (convTestType != 4)) {
        int output_level = split_convergence_level ? convergence_level : lev;
        Array<Point, DIM+1> zeroGhost;
          for (int d = 0; d < DIM + 1; ++d)
            {
              zeroGhost[d] = Point::Zeros();
            }
      U_conv_test[output_level].define(layout, zeroGhost);
      for (auto dit : layout)
      {
        JU[dit].copyTo(U_conv_test[output_level][dit]);
      }
      h5.writeMBLevel({}, map, U_conv_test[output_level], "U_conv_test_" + to_string(output_level));
      if (!split_convergence_level)
        {
          Point refRatio = 2*Point::Ones();
          if (radial_refinement)
            {
              refRatio = 2*Point::Ones() - Point::Basis(1) - Point::Basis(2);
              domainSize *= refRatio[1];
              boxSize_nonrad *= refRatio[1];
            } else {
              domainSize *= 2;
              boxSize_nonrad *= 2;
            }

          thickness *= 2;
          boxSize_rad *= 2;
          time = 0.;
          domain = domain.refine(refRatio);
          dt = convergence_step_dt;
          if (convTestType == 2){
            dt /= 2;
            max_iter *= 2;
          }
          pout(0) << "thickness = " << thickness/2 << " is done" << endl;
        }
      }
  }

    if((convTestType > 0) && (convTestType != 4) && !split_convergence_level)
      {
        Array<Array<double,8>,2> errmaxglobal;
        for(int lev=0; lev<2; lev++)
          {
            Array<Reduction<double,Operation::Max,HOST>,8> errmax;
            for (int comp = 0; comp < 8;comp++)
              {
                errmax[comp].reset();
              }
            auto layout = U_conv_test[lev].layout();
            MBLevelBoxData<double,8,HOST> err(layout,Point::Zeros());
            Point refRatio;
            if (radial_refinement)
              {
                refRatio = 2*Point::Ones() - Point::Basis(1) - Point::Basis(2);
              }
            else
              {
                refRatio = 2*Point::Ones();
              }          
            U_conv_test[lev + 1].coarsenTo(err,refRatio);
            
            for (auto dit : layout)
              {
                unsigned int block = layout.block(dit);
                Box blockBox = layout.getBlock(block).domain().box();
                Box radialBlockFaceHi = blockBox.face(0,Side::Hi,3);
                Box radialPatchFaceHi = radialBlockFaceHi&err[dit].box();
                Box radialBlockFaceLo = blockBox.face(0,Side::Lo,3);
                Box radialPatchFaceLo = radialBlockFaceLo&err[dit].box();
                err[dit] -= U_conv_test[lev][dit];
                for (int comp = 0; comp < 8; comp++)
                  {
                    auto errslice = slice(err[dit],comp);
                    if (!radialPatchFaceHi.empty())
                      {
                        errslice.setVal(0.,radialPatchFaceHi);
                      }
                    if (!radialPatchFaceLo.empty())
                      {
                        errslice.setVal(0.,radialPatchFaceLo);
                      }
                    double errPatch = errslice.absMax();
                    errmax[comp].reduce(&errPatch,1);
                  }
              }
            HDF5Handler h5;
            auto map = CubedSphereShell::Map(err);
            h5.writeMBLevel({"err"}, map, err,"Err"+to_string(lev));
            for (int comp = 0; comp < 8; comp++)
              {
                errmaxglobal[lev][comp] = errmax[comp].fetch();

               pout(0) << "Lev = " << lev << ", component = "
                        << comp <<", error = " << errmaxglobal[lev][comp] << std::endl;

              }
          }
        for (int comp = 0; comp < 8 ; comp++)
          {
            double rate = log(abs(errmaxglobal[0][comp]/errmaxglobal[1][comp]))/log(2.0);
            pout(0) << "order of accuracy for var " << comp << " = " << rate << std::endl;
          }    
      }
  PR_TIMER_REPORT();
#ifdef PR_MPI
  MPI_Finalize();
#endif
}
