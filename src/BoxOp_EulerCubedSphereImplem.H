// ====================================================================================
// New factorization of operator implems start here.
// ====================================================================================
template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::operator()
  (
   State&                          a_Rhs,
   Array<State, DIM>&              a_fluxes,
   const State&                    a_USph,
   T                               a_dt,
   T                               a_scale,
   int                             a_counter,
   bool                            a_cartesianPrimitives,
   const SecondOrderFaceMask*       a_activeFaces) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::operator()");
  // Magnetized test problems evolve perturbations about a uniform Cartesian
  // magnetic equilibrium. Subtract the numerical operator evaluated on that
  // exact background. Its continuum divergence is zero, so this preserves the
  // MHD equations while removing cubed-sphere metric/interpolation residuals
  // that are otherwise amplified by the very low plasma beta.
  static thread_local bool evaluatingBackground = false;
  const bool subtractBackground =
      isConstantMagneticFieldTest(ParseInputs::get_init_condition_type())
      && !evaluatingBackground;
  if (subtractBackground)
  {
    const bool boxesMatch = m_testBackgroundReady
        && m_testBackgroundRhs.box() == a_Rhs.box();
    if (!boxesMatch || m_testBackgroundScale != a_scale)
    {
      State backgroundUSph(a_USph.box());
      forallInPlace([] PROTO_LAMBDA(
          Var<T,NUMCOMPS,MEM>& a_state)
        {
          a_state(iRHO) = c_TEST_RHO0;
          a_state(iMOMX) = 0.0;
          a_state(iMOMY) = 0.0;
          a_state(iMOMZ) = 0.0;
          // Energy and magnetic conserved variables store perturbations;
          // primitive recovery restores their backgrounds.
          a_state(iE) = 0.0;
          a_state(iBX) = 0.0;
          a_state(iBY) = 0.0;
          a_state(iBZ) = 0.0;
        },backgroundUSph);

      m_testBackgroundRhs.define(a_Rhs.box());
      m_testBackgroundRhs.setVal(0.0);
      for (int dir = 0; dir < DIM; ++dir)
      {
        m_testBackgroundFluxes[dir].define(a_fluxes[dir].box());
        m_testBackgroundFluxes[dir].setVal(0.0);
      }
      evaluatingBackground = true;
      this->operator()(
          m_testBackgroundRhs,
          m_testBackgroundFluxes,
          backgroundUSph,
          a_dt,
          a_scale,
          a_counter);
      evaluatingBackground = false;
      m_testBackgroundScale = a_scale;
      m_testBackgroundReady = true;
    }
  }

  if (ParseInputs::get_spatial_order() == 2)
  {
    secondOrderOperator(a_Rhs,a_fluxes,a_USph,a_scale,a_cartesianPrimitives,a_activeFaces);
    if (subtractBackground)
    {
      a_Rhs -= m_testBackgroundRhs;
      for (int dir=0; dir<DIM; ++dir) a_fluxes[dir] -= m_testBackgroundFluxes[dir];
    }
    return;
  }
  if (ParseInputs::get_spatial_operator() == 2)
  {
    BertaOperator(a_Rhs,a_fluxes,a_USph,a_dt,a_scale,a_counter);
    if (subtractBackground)
    {
      a_Rhs -= m_testBackgroundRhs;
      for (int dir = 0; dir < DIM; ++dir)
      {
        a_fluxes[dir] -= m_testBackgroundFluxes[dir];
      }
    }
    return;
  }

  int a_block = this->block();
  HDF5Handler h5;
  Box bxRad = a_USph.box().grow(Point::Basis(0,2));
  BoxData<T,DIM,MEM> Dr(bxRad);
  BoxData<T,1,MEM> radius(bxRad);
  BoxData<T,DIM,MEM> adjDr(bxRad);
  BoxData<T,1,MEM> dVolr(bxRad);
  radialMetrics(radius,Dr,adjDr,dVolr,bxRad);
  auto domainBox = this->layout().domain().getBlock(a_block).box();
  // Primitive Variables. 
  // Transform to primitive variables, spherical velocities.
  Box bx = a_USph.box();
  State W;
  State WBar;
  const bool magneticTest =
      isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
  // Reconstruct the evolved perturbation for magnetized test problems.  A
  // single face representation of the analytic background is added below so
  // that the dissipative jump cannot contain two independently reconstructed
  // copies of the same constant Cartesian field.
  BoxData<T,1,MEM> me;
  sphToPrim(W,WBar,a_USph,a_block,!magneticTest,magneticTest ? nullptr : &me);

  State testBackgroundW;
  State testBackgroundWBar;
  State WBarForLimiter;
  if (magneticTest)
  {
    testBackgroundWBar.define(WBar.box());
    testBackgroundWBar.setVal(0.0);
    addTestProblemBackgroundB(testBackgroundWBar,a_block);

    testBackgroundW.define(W.box());
    testBackgroundWBar.copyTo(testBackgroundW);
    testBackgroundW +=
        Stencil<T>::Laplacian()(testBackgroundWBar,T(1.0/24.0));

    WBarForLimiter.define(WBar.box());
    WBar.copyTo(WBarForLimiter);
    WBarForLimiter += testBackgroundWBar;
  }

  if (magneticTest) { me = magneticEnergyW(WBarForLimiter,a_block,0); }
  // Compute Fluxes, rhs.
  // a_Rhs.setVal(0.0);
  T dxiPerp = this->dx()[1];

  Box outputbox = bx;
  // Loop over directions.
  T one = 1.0;
  T half = 0.5;
  T dxiR = this->dx()[0];
  
  BoxData<T,DIM,MEM,DIM> A_matrix0(bx);

  const bool powellEnabled = ParseInputs::get_divergence_cleaning() == 1;
  // One source halo supplies the cell-average product in DivBPowell2.
  // The existing primitive ghosts cover its face reconstruction/products.
  BoxData<T,1,MEM> minusJDivB(a_Rhs.box().grow(1), T(0));
  
  for (int dir = 0; dir < DIM; dir++)
    {
      PR_TIME("BoxOp_Euler::operator()::directional flux calculation");
      // Flux outputs and persistent background caches are outside this scope.
      Proto::ScopedHostStack<MEM> scratch;
      
      State WFace_L = Stencil<T>::CellToFace(dir)(W);   
      State WFace_R;
      WFace_R.defineForOverwrite(WFace_L.box());
      WFace_L.copyTo(WFace_R);
      // Two transverse layers feed the two metric products; one more
      // supplies the face convolution/deconvolution. No normal halo is used.
      const Box faceBarBox = a_fluxes[dir].box().grow(
          3*(Point::Ones()-Point::Basis(dir)));
      State WFaceBar = Stencil<T>::CellToFace(dir,Side::Lo,2)(WBar,faceBarBox);
      
      //Required mapping variables.
      Box bxface = WFace_L.box().grow(-2);  
      int limiter = ParseInputs::get_limiter();
      if (limiter != 0 && limiter != 1 && limiter != 2)
      {
        limiter = 1;
      }   
      if (limiter == 2) bxface = WFace_L.box(); 
      BoxData<T,1,MEM> adjDrDir = slice(adjDr,dir);  
      BoxData<T,DIM,MEM,DIM> deta_a = CubedSphereShell::detA_A<T,MEM>(bxface,dxiPerp,a_block,dir);
      BoxData<T,1,MEM> deta = CubedSphereShell::detA<T,MEM>(bxface,dxiPerp,dir); 
      Box bx = W.box();
      Box bxdir = bx.extrude(Point::Basis(dir));
      BoxData<T,DIM,MEM> ntrow = CubedSphereShell::adjA<T,MEM>(bxdir,dxiPerp,a_block,dir);
      Array<Array<uint,DIM>,6> permute = {{1,2,0},{1,2,0},{1,0,2},{0,1,2},{1,0,2},{0,1,2}};
      Array<Array<int,DIM>,6> sign = {{1,-1,-1},{1,1,1},{1,-1,1},{1,1,1},{-1,1,1},{-1,-1,1}}; 
      Point high = bx.high();
      Point low = bx.low();
      high[0] = low[0];
      Box bx0(low,high);
      BoxData<T,DIM,MEM,DIM> A_matrix(bx);
      T offseta = half;
      T offsetb = half;
      if (dir == 1) offseta = 0.0;
      if (dir == 2) offsetb = 0.0;
      forallInPlace_p(f_Amatrix,bx0,A_matrix,permute[a_block],sign[a_block],
                      dxiPerp,offseta,offsetb);
      spreadSlice(A_matrix);
      if (dir == 0) A_matrix.copyTo(A_matrix0);

      if (powellEnabled)
      {
        const Box powellMetricBox = minusJDivB.box().extrude(dir).grow(
            2*(Point::Ones()-Point::Basis(dir)));
        auto powellDetA = CubedSphereShell::detA<T,MEM>(
            powellMetricBox,dxiPerp,dir);
        auto Bnormal = slice(WFace_L,iBX + dir);
        DivBPowell1(minusJDivB,Bnormal,powellDetA,adjDrDir,dir);
      }
      
      

      if (limiter == 1){
        PR_TIMERS("BoxOp_Euler::operator()::limiters");
        Limiters_4O(WFace_L,WFace_R,WFace_L,WFace_R,W,WBar,me,dir,dxiR, dxiPerp, dxiPerp);
      }

      if (limiter == 2){
        BoxData<T,DIM> dx(bxface);
        forallInPlace([ ] PROTO_LAMBDA
        ( Var<T,DIM,MEM>& a_dx_d,
          const Var<T,DIM,MEM>& a_adjA,
          const Var<T,1,MEM>& a_adjDrDir,
          const Var<T,1,MEM>& a_detA,
          const Var<T,1,MEM>& a_dVolr,
          const Var<T,DIM,MEM>& a_Dr,
          const T a_dxiPerp,
          const T a_dxiR,
          const int a_d)
        { 
          if (a_d > 0){
            T adjA_mag = sqrt(a_adjA(0)*a_adjA(0) + a_adjA(1)*a_adjA(1) + a_adjA(2)*a_adjA(2));
            a_dx_d(a_d) = a_dVolr(0) * a_detA(0) * a_dxiPerp / a_adjDrDir(0) / adjA_mag;
          } 
          if (a_d == 0){
            a_dx_d(a_d) = a_Dr(0) * a_dxiR;
          }                   
        },dx,ntrow,adjDrDir,deta,dVolr,Dr,dxiPerp,dxiR,dir);
        Limiters_minmod(WFace_L,WFace_R,W,dx,dir);
        WFaceBar *= 0.0;
        WFaceBar += WFace_L;
        WFaceBar += WFace_R;
        WFaceBar *= 0.5;
      }

      if (ParseInputs::get_Riemann_solver_type() == 2)
      {
        // Repair delta-B states before adding the common background field.
        // Both the physical flux and Roe dissipation use these repaired states.
        repairRoeFaceStates(WFace_L,WFace_R,WFaceBar,WBar,dir);
      }

      if (magneticTest)
      {
        // W and WBar contain delta-B.  Add the same centered, fourth-order
        // face average of B0 to both Riemann states, while WFaceBar receives
        // the corresponding second-order face-center value.  This keeps the
        // physical flux and wave speed based on the full field but removes
        // B0 exactly from the dissipative left/right jump.
        State backgroundFace =
            Stencil<T>::CellToFace(dir)(testBackgroundW);
        State backgroundFaceBar =
            Stencil<T>::CellToFace(dir,Side::Lo,2)(testBackgroundWBar);
        WFace_L += backgroundFace;
        WFace_R += backgroundFace;
        WFaceBar += backgroundFaceBar;
      }

      // Fluxes on dir face.
      // Average of advective fluxes in spherical variables at the
      // face center, given point values of W.
      // All face mutations are complete. Reuse these values only within
      // this direction and stage; repair's earlier point states may be stale.
      State facePoint_L = _deconvolveFace(WFace_L,WFaceBar,dir);
      State facePoint_R = _deconvolveFace(WFace_R,WFaceBar,dir);
      BoxData<T,1,MEM> radialPoint;
      if (dir == 0)
      {
        radialPoint.defineForOverwrite(adjDrDir.box());
        adjDrDir.copyTo(radialPoint);
      }
      else
      {
        radialPoint = (1.0*Shift(Point::Zeros())
            - (1.0/24.0)*Stencil<T>::Derivative(2,0,2))(adjDrDir);
      }
      // The pointwise coordinate basis is identical for all three states.
      // Cover their full boxes so sharing does not crop any calculation.
      auto energyBasis = pointwiseSphericalBasis(
          WFaceBar.box()+facePoint_L.box()+facePoint_R.box(),a_block,dir);
      auto energy_L = faceEnergies(facePoint_L,a_block,dir,&energyBasis);
      auto energy_R = faceEnergies(facePoint_R,a_block,dir,&energyBasis);
      auto energyBar = faceEnergies(WFaceBar,a_block,dir,&energyBasis);
      // Both physical fluxes use exactly the same auxiliary face state.
      State commonFluxBar = forallOverwrite<T,NUMCOMPS>(f_primflux,WFaceBar,
          energyBar.kinetic,energyBar.magnetic,energyBar.velocityDotMagnetic,
          radialPoint,m_gamma,a_block,dir);
      BoxData<T,NUMCOMPS> physFlux_L,physFlux_R;
      primToFlux(physFlux_L,WFace_L,WFaceBar,adjDrDir,a_block,dir,
          &facePoint_L,&radialPoint,&energy_L,&commonFluxBar);
      primToFlux(physFlux_R,WFace_R,WFaceBar,adjDrDir,a_block,dir,
          &facePoint_R,&radialPoint,&energy_R,&commonFluxBar);
      Box bxface0 = a_fluxes[dir].box();
      State fluxdir_L(bxface),fluxdir_R(bxface),fluxdir,fluxdir_Diss;
      
      // det(A) A is shared by both face states and both vector fluxes.
      // Keep its differences within the directional scratch lifetime.
      std::array<BoxData<T,DIM,MEM,DIM>,2> metricDifferences;
      if constexpr (MEM == HOST && DIM == 3)
      {
        const int firstDir = (dir == 0) ? 1 : 0;
        const int secondDir = (dir == 2) ? 1 : 2;
        metricDifferences[0] = (0.5*Shift::Basis(firstDir,1)
            - 0.5*Shift::Basis(firstDir,-1))(deta_a);
        metricDifferences[1] = (0.5*Shift::Basis(secondDir,1)
            - 0.5*Shift::Basis(secondDir,-1))(deta_a);
      }
      // Auxiliary pressure is identical for the final left and right states.
      auto pressureBar = slice(WFaceBar,iP) + energyBar.magnetic;
      auto radialPressureBar = pressureBar * adjDrDir;
      primToMappedFlux(fluxdir_L,physFlux_L,WFace_L,WFaceBar,adjDrDir,deta_a,deta,ntrow,a_block,dir,&facePoint_L,&metricDifferences,&energy_L.magnetic,&energyBar.magnetic,&pressureBar,&radialPressureBar);
      primToMappedFlux(fluxdir_R,physFlux_R,WFace_R,WFaceBar,adjDrDir,deta_a,deta,ntrow,a_block,dir,&facePoint_R,&metricDifferences,&energy_R.magnetic,&energyBar.magnetic,&pressureBar,&radialPressureBar);
      if (ParseInputs::get_Riemann_solver_type() == 2)
      {
        fluxdir_Diss.define(bxface0);
        primToMappedRoeFlux(fluxdir_Diss,WFace_L,WFace_R,WFaceBar,adjDrDir,deta_a,deta,A_matrix,dir,&facePoint_L,&facePoint_R,&radialPoint,&metricDifferences);
      }
      else
      {
        fluxdir_Diss.define(bxface);
        primToMappedRusanovFlux(fluxdir_Diss,WFace_L,WFaceBar,WFace_R,WFaceBar,adjDrDir,deta_a,deta,A_matrix,ntrow,a_block,dir,&facePoint_L,&facePoint_R,&radialPoint,&energy_L,&energy_R);
      }
      
      // Write directly into the caller-owned flux array when every output
      // cell is covered. It outlives the directional scratch arena.
      const bool fullFlux = MEM == HOST && (fluxdir_L.box() & fluxdir_R.box() & fluxdir_Diss.box()).containsBox(bxface0);
      if (fullFlux) { fluxdir = alias(a_fluxes[dir]); }
      else { fluxdir.define(bxface0); }
      forallInPlace([ ] PROTO_LAMBDA
        (Var<T,NUMCOMPS,MEM>& a_fluxout,
          const Var<T,NUMCOMPS,MEM>& a_fluxin_L,
          const Var<T,NUMCOMPS,MEM>& a_fluxin_R,
          const Var<T,NUMCOMPS,MEM>& a_fluxin_Diss)
        {  
          for (int comp = 0; comp < NUMCOMPS;comp++)
          {                  
            a_fluxout(comp) = 0.5*(a_fluxin_L(comp) + a_fluxin_R(comp) - a_fluxin_Diss(comp));
          }
        },fluxdir,fluxdir_L,fluxdir_R,fluxdir_Diss);

      // Save fluxes, update RHS.
      if (!fullFlux) { fluxdir.copyTo(a_fluxes[dir]); }
      T dxInv = -a_scale/(this->dx()[dir]);
      if constexpr (MEM == HOST)
      {
        auto ahead = alias(fluxdir,Point::Basis(dir,-1));
        const T lowCoefficient = T(-1)*dxInv;
        const T highCoefficient = T(1)*dxInv;
        const Box region = a_Rhs.box() & fluxdir.box() & ahead.box();
        for (int c=0; c<NUMCOMPS; ++c)
        for (auto p : region.flatten(0))
        {
          T* __restrict rhs = a_Rhs.data(p,c);
          const T* lo = fluxdir.data(p,c);
          const T* hi = ahead.data(p,c);
          for (int i=0; i<region.size(0); ++i)
          {
            T increment = 0;
            increment += lowCoefficient*lo[i];
            increment += highCoefficient*hi[i];
            rhs[i] += increment;
          }
        }
      }
      else
      {
        State rhsIncr = Stencil<T>::FluxDivergence(dir)(fluxdir,dxInv);
        a_Rhs += rhsIncr;
      }

    }

  // Multiply -J*div(B) by the remaining Powell terms for the mapped RHS.
  if (powellEnabled)
  {
    if (magneticTest) W += testBackgroundW;
    DivBPowell2(a_Rhs, minusJDivB, W, A_matrix0);
  }

  if (subtractBackground)
  {
    a_Rhs -= m_testBackgroundRhs;
    for (int dir = 0; dir < DIM; ++dir)
    {
      a_fluxes[dir] -= m_testBackgroundFluxes[dir];
    }
  }

    
}
// primToMappedFlux(fluxdir_L,physFlux_L,WFace_L,WFaceBar,adjDrDir,deta_a,deta,ntrow,a_block,dir);
template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::primToMappedFlux(
                                          State& a_fluxdir,
                                          const State& a_physFlux,
                                          const State& a_WFace,
                                          const State& a_WFaceBar,
                                          const BoxData<T,1,MEM>& a_adjDrDir,
                                          const BoxData<T,DIM,MEM,DIM>& a_deta_a,
                                          const BoxData<T,1,MEM>& a_deta,
                                          const BoxData<T,DIM,MEM>& a_ntrow,
                                          uint a_block,
                                          uint a_dir,
                                          const State* a_point,
                                          const std::array<BoxData<T,DIM,MEM,DIM>,2>* a_metricDifferences,
                                          const BoxData<T,1,MEM>* a_magneticPoint,
                                          const BoxData<T,1,MEM>* a_magneticBar,
                                          const BoxData<T,1,MEM>* a_pressureBar,
                                          const BoxData<T,1,MEM>* a_radialPressureBar) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::primToMappedFlux()");
  // Multiply momentum and B fluxes by det(A) A.
  BoxData<T,DIM,MEM> fluxdirMom = slice<T,NUMCOMPS,DIM,MEM>(a_fluxdir,iMOMX);
  BoxData<T,DIM,MEM> physFluxMom = slice<T,NUMCOMPS,DIM,MEM>(a_physFlux,iMOMX);
  BoxData<T,DIM,MEM> tmp =
    Operator::_faceMatrixProductAB(a_deta_a,physFluxMom,a_deta_a,physFluxMom,a_dir,a_metricDifferences);
  tmp.copyTo(fluxdirMom);

  BoxData<T,DIM,MEM> fluxdirB = slice<T,NUMCOMPS,DIM,MEM>(a_fluxdir,iBX);
  BoxData<T,DIM,MEM> physFluxB = slice<T,NUMCOMPS,DIM,MEM>(a_physFlux,iBX);
  BoxData<T,DIM,MEM> tmp2 =
    Operator::_faceMatrixProductAB(a_deta_a,physFluxB,a_deta_a,physFluxB,a_dir,a_metricDifferences);
  tmp2.copyTo(fluxdirB);

  // Both factors are face averages. Include their transverse covariance,
  // as the momentum and magnetic matrix products above already do.
  for (int comp : {iRHO, iE})
  {
    auto physical = slice(a_physFlux,comp);
    auto mapped = Operator::_faceProduct(a_deta,physical,a_deta,physical,a_dir);
    auto destination = slice(a_fluxdir,comp);
    // Only the covered stencil contributes to owned faces. Initialize the
    // unused margin before the subsequent radial metric product reads it.
    destination.setVal(0.);
    mapped.copyTo(destination);
    Box needed = this->box().extrude(a_dir).grow(1).grow(a_dir,-1);
    PROTO_ASSERT((mapped.box() & needed) == needed,
        "Scalar metric product does not cover the required flux stencil");
  }

  // Apply the radial metric to all components together so its transverse
  // differences are computed once. The product is separate from a_fluxdir
  // until every component has been evaluated.
  auto advfluxprod = Operator::_faceTensorProduct(
      a_fluxdir,a_adjDrDir,a_fluxdir,a_adjDrDir,a_dir);
  advfluxprod.copyTo(a_fluxdir);

  //adding pressure forces to momentum fluxes.
   BoxData<T,1,MEM> pslice = slice(a_WFace,iP);
   BoxData<T,1,MEM> psliceBar = slice(a_WFaceBar,iP);
   State ownedPoint;
   if (!a_point) { ownedPoint = _deconvolveFace(a_WFace,a_WFaceBar,a_dir); a_point = &ownedPoint; }
   const State& WPoint = *a_point;
   BoxData<T,1,MEM> ownedMagneticBar,ownedMagneticPoint;
   if (!a_magneticBar)
   {
     ownedMagneticBar = magneticEnergyW(a_WFaceBar,a_block,a_dir);
     a_magneticBar = &ownedMagneticBar;
   }
   if (!a_magneticPoint)
   {
     ownedMagneticPoint = magneticEnergyW(WPoint,a_block,a_dir);
     a_magneticPoint = &ownedMagneticPoint;
   }
   const auto& meBar = *a_magneticBar;
   const auto& mePoint = *a_magneticPoint;
   auto me = _convolveFace(mePoint,meBar,a_dir);
   BoxData<T,1,MEM> p0 = pslice + me;
   BoxData<T,1,MEM> ownedPressureBar, ownedRadialPressureBar;
   if (!a_pressureBar)
   {
     ownedPressureBar = psliceBar + meBar;
     a_pressureBar = &ownedPressureBar;
   }
   const auto& p0Bar = *a_pressureBar;
   if (!a_radialPressureBar)
   {
     ownedRadialPressureBar = p0Bar * a_adjDrDir;
     a_radialPressureBar = &ownedRadialPressureBar;
   }
   const auto& tmp4 = *a_radialPressureBar;
   BoxData<T,DIM,MEM> pforce(a_ntrow.box(), T(0));
   BoxData<T,1,MEM> tmp3 =
     _faceProduct(a_adjDrDir,p0,a_adjDrDir,p0Bar,a_dir);
   if constexpr (MEM == HOST && DIM == 3)
   {
     // The same pressure derivative serves all three momentum components.
     // Retain the scalar products' factor and accumulation order.
     for (int transverse = 0; transverse < DIM; ++transverse)
     {
       if (transverse == a_dir) { continue; }
       const Stencil<T> difference = 0.5*Shift::Basis(transverse,1)
           - 0.5*Shift::Basis(transverse,-1);
       BoxData<T,1,MEM> dp = difference(tmp4);
       BoxData<T,DIM,MEM> dn = difference(a_ntrow);
       for (int component = 0; component < DIM; ++component)
       {
         auto output = slice(pforce,component);
         output.incrementProduct(dp,slice(dn,component),T(1.0/12.0));
       }
     }
     for (int component = 0; component < DIM; ++component)
     {
       auto output = slice(pforce,component);
       output.incrementProduct(tmp3,slice(a_ntrow,component));
     }
   }
   else
   {
     for (int component = 0; component < DIM; ++component)
     {
       auto normal = slice(a_ntrow,component);
       auto output = slice(pforce,component);
       _faceProduct(output,tmp3,normal,tmp4,normal,a_dir);
     }
   }
    fluxdirMom += pforce;   
}

template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::primToMappedRusanovFlux(
                                  State& a_fluxdir,
                                  const State& a_WFace_L,
                                  const State& a_WFaceBar_L,
                                  const State& a_WFace_R,
                                  const State& a_WFaceBar_R,
                                  const BoxData<T,1,MEM>& a_adjDdir,
                                  const BoxData<T,DIM,MEM,DIM>& a_deta_a,
                                  const BoxData<T,1,MEM>& a_deta,
                                  const BoxData<T,DIM,MEM,DIM>& a_A,
                                  const BoxData<T,DIM,MEM>& a_ntrow,
                                  uint a_block,
                                  uint a_dir,
                                  const State* a_leftPoint,
                                  const State* a_rightPoint,
                                  const BoxData<T,1,MEM>* a_radialPoint,
                                  const FaceEnergies* a_leftEnergy,
                                  const FaceEnergies* a_rightEnergy) const
{

  // Compute fluxes at face centers.
  PR_TIMERS("BoxOp_EulerCubedSphere::primToMappedRusanovFlux()");
  Stencil<T> deconvolveStencilRadial = (1.0*Shift(Point::Zeros())
                                    - (1.0/24.0)*Stencil<T>::Derivative(2,0,2));
  // Deconvolve W to obtain point values.
  State ownedLeft;
  if (!a_leftPoint) { ownedLeft = _deconvolveFace(a_WFace_L,a_WFaceBar_L,a_dir); a_leftPoint = &ownedLeft; }
  const State& WFacePoint_L = *a_leftPoint;
  FaceEnergies ownedEnergy_L;
  if (!a_leftEnergy)
  {
    ownedEnergy_L.kinetic.define(WFacePoint_L.box());
    ownedEnergy_L.magnetic.define(WFacePoint_L.box());
    EnergyW(ownedEnergy_L.kinetic,ownedEnergy_L.magnetic,WFacePoint_L,a_block,a_dir);
    a_leftEnergy = &ownedEnergy_L;
  }
  const auto& kePoint_L = a_leftEnergy->kinetic;
  const auto& mePoint_L = a_leftEnergy->magnetic;

  BoxData<T,1,MEM> ownedRadial;
  if (!a_radialPoint)
  {
    if (a_dir != 0)
    {
      ownedRadial = deconvolveStencilRadial(a_adjDdir);
    }
    else
    {
      ownedRadial.define(a_adjDdir.box());
      a_adjDdir.copyTo(ownedRadial);
    }
    a_radialPoint = &ownedRadial;
  }
  const auto& adjDdirPoint = *a_radialPoint;

  State q_L = forall<T,NUMCOMPS>
    ([ ] PROTO_LAMBDA
     (
      Var<T,NUMCOMPS>& a_u,
      const Var<T,NUMCOMPS>& a_wface,      
      const Var<T,1>& a_ke,
      const Var<T,1>& a_me,
      const Var<T>& a_adjddir,
      const T a_gamma)
     {
        a_u(iRHO ) = a_adjddir(0)*a_wface(iRHO);
        a_u(iMOMX) = a_adjddir(0)*a_wface(iRHO)*a_wface(iVX);
        a_u(iMOMY) = a_adjddir(0)*a_wface(iRHO)*a_wface(iVY);
        a_u(iMOMZ) = a_adjddir(0)*a_wface(iRHO)*a_wface(iVZ);
        a_u(iE   ) = a_adjddir(0)*(a_ke(0) + a_me(0) + a_wface(iP)/(a_gamma - 1.0));
        a_u(iBX  ) = a_adjddir(0)*a_wface(iBX);
        a_u(iBY  ) = a_adjddir(0)*a_wface(iBY);
        a_u(iBZ  ) = a_adjddir(0)*a_wface(iBZ);
     },WFacePoint_L,kePoint_L,mePoint_L,adjDdirPoint,m_gamma);

  // Compute Average conserved variables from pointwise values.
  State q_avg_L = (1.0*Shift(Point::Zeros()) +
                (1.0/24.0)*Stencil<T>::LaplacianFace(a_dir))(q_L);


  // Deconvolve W to obtain point values.
  State ownedRight;
  if (!a_rightPoint) { ownedRight = _deconvolveFace(a_WFace_R,a_WFaceBar_R,a_dir); a_rightPoint = &ownedRight; }
  const State& WFacePoint_R = *a_rightPoint;
  FaceEnergies ownedEnergy_R;
  if (!a_rightEnergy)
  {
    ownedEnergy_R.kinetic.define(WFacePoint_R.box());
    ownedEnergy_R.magnetic.define(WFacePoint_R.box());
    EnergyW(ownedEnergy_R.kinetic,ownedEnergy_R.magnetic,WFacePoint_R,a_block,a_dir);
    a_rightEnergy = &ownedEnergy_R;
  }
  const auto& kePoint_R = a_rightEnergy->kinetic;
  const auto& mePoint_R = a_rightEnergy->magnetic;

  State q_R = forall<T,NUMCOMPS>
    ([ ] PROTO_LAMBDA
     (
      Var<T,NUMCOMPS>& a_u,
      const Var<T,NUMCOMPS>& a_wface,      
      const Var<T,1>& a_ke,
      const Var<T,1>& a_me,
      const Var<T>& a_adjddir,
      const T a_gamma)
     {
        a_u(iRHO ) = a_adjddir(0)*a_wface(iRHO);
        a_u(iMOMX) = a_adjddir(0)*a_wface(iRHO)*a_wface(iVX);
        a_u(iMOMY) = a_adjddir(0)*a_wface(iRHO)*a_wface(iVY);
        a_u(iMOMZ) = a_adjddir(0)*a_wface(iRHO)*a_wface(iVZ);
        a_u(iE   ) = a_adjddir(0)*(a_ke(0) + a_me(0) + a_wface(iP)/(a_gamma - 1.0));
        a_u(iBX  ) = a_adjddir(0)*a_wface(iBX);
        a_u(iBY  ) = a_adjddir(0)*a_wface(iBY);
        a_u(iBZ  ) = a_adjddir(0)*a_wface(iBZ);
     },WFacePoint_R,kePoint_R,mePoint_R,adjDdirPoint,m_gamma);

  // Compute Average conserved variables from pointwise values.
  State q_avg_R = (1.0*Shift(Point::Zeros()) +
                (1.0/24.0)*Stencil<T>::LaplacianFace(a_dir))(q_R);
  
  // Difference of conserved variables.
  q_avg_R -= q_avg_L;

  T one = 1.0;
  BoxData<T,DIM,MEM> fluxdirMom = slice<T,NUMCOMPS,DIM,MEM>(a_fluxdir,iMOMX);
  BoxData<T,DIM,MEM> Con_diff_Mom = slice<T,NUMCOMPS,DIM,MEM>(q_avg_R,iMOMX);
  forallInPlace(f_matVecProd,fluxdirMom,a_deta_a,Con_diff_Mom,one);

  BoxData<T,DIM,MEM> fluxdirB = slice<T,NUMCOMPS,DIM,MEM>(a_fluxdir,iBX);
  BoxData<T,DIM,MEM> Con_diff_B = slice<T,NUMCOMPS,DIM,MEM>(q_avg_R,iBX);
  forallInPlace(f_matVecProd,fluxdirB,a_deta_a,Con_diff_B,one);
    
  forallInPlace([ ] PROTO_LAMBDA
                (Var<T,NUMCOMPS,MEM>& a_Rusflux,
                  const Var<T,NUMCOMPS,MEM>& a_fluxin,
                  const Var<T,1,MEM>& a_deta)
                {                    
                  a_Rusflux(iRHO) = a_deta(0)*a_fluxin(iRHO);
                  a_Rusflux(iE  ) = a_deta(0)*a_fluxin(iE);  
                },a_fluxdir,q_avg_R,a_deta);

  // Find (|w_d| + c_fast*||A||_d) and multiply with a_fluxdir.
  forallInPlace([ ] PROTO_LAMBDA
               (Var<T,NUMCOMPS,MEM>& a_Rusflux,
                const Var<T,NUMCOMPS>& a_wface_L,
                const Var<T,NUMCOMPS>& a_wface_R,
                const Var<T,DIM,MEM,DIM>& a_Amatrix,
                const Var<T,1,MEM>& a_detA,
                const Var<T,DIM,MEM>& a_adjArow,
                const T a_gamma,
                const int a_d)
                {        
                  double rho = 0.5*(a_wface_L(iRHO)+a_wface_R(iRHO));
                  double w_d = 0.5*(a_wface_L(iVX+a_d)+a_wface_R(iVX+a_d));
                  double B_x = 0.5*(a_wface_L(iBX)+a_wface_R(iBX));
                  double B_y = 0.5*(a_wface_L(iBY)+a_wface_R(iBY));
                  double B_z = 0.5*(a_wface_L(iBZ)+a_wface_R(iBZ));
                  double p = 0.5*(a_wface_L(iP)+a_wface_R(iP));
                  if (p < 0.0) p = 0.0;
                  rho = max(rho,RHO00);
                  vecType bSph;
                  vecType bCart;
              
                  bSph[0] = B_x;       
                  bSph[1] = B_y;
                  bSph[2] = B_z;
                  bCart[0] = 0.;       
                  bCart[1] = 0.;
                  bCart[2] = 0.;
                  double B_n = 0.;
                  double B_mag = 0.;
                  for (int dir0 = 0; dir0 < DIM; dir0++)
                    {
                      for (int dir1 = 0; dir1 < DIM; dir1++)
                        {
                          bCart[dir0] += a_Amatrix(dir0,dir1)*bSph[dir1];
                        }
                        B_n += a_Amatrix(dir0,0)*bCart[dir0];
                        B_mag += bCart[dir0]*bCart[dir0];
                    }
                  B_mag = sqrt(B_mag);
    
                  double ce = sqrt(a_gamma*p/rho);
                  double a = ce*ce + B_mag*B_mag/4.0/M_PI/rho;
                  double b = abs(B_n)*ce/sqrt(M_PI*rho);
                  double af = 0.5*(sqrt(a + b) + sqrt(a - b));
                  double A_d_mag = sqrt(a_Amatrix(0,a_d)*a_Amatrix(0,a_d) + a_Amatrix(1,a_d)*a_Amatrix(1,a_d) + a_Amatrix(2,a_d)*a_Amatrix(2,a_d));
                  
                  for (int comp = 0; comp < NUMCOMPS;comp++)
                  {
                    a_Rusflux(comp) *= (abs(w_d) + af*A_d_mag);
                  }
                },a_fluxdir,a_WFace_L,a_WFace_R,a_A,a_deta,a_ntrow,m_gamma,a_dir);
}

template<
    typename T,
    typename MAP,
    MemType MEM>
inline void BoxOp_EulerCubedSphere<T,MAP,MEM>::
primToFlux(
           State& a_physFlux,
           const State& a_WFace,
           const State& a_WFaceBar,
           const BoxData<T,1,MEM>& a_adjDdir,
           uint a_block,
           uint a_centering,
           const State* a_point,
           const BoxData<T,1,MEM>* a_radialPoint,
           const FaceEnergies* a_pointEnergy,
           const State* a_fluxBar) const
{
  // Compute fluxes at face centers.
  PR_TIMERS("BoxOp_EulerCubedSphere::primToFlux()");
  T dxiPerp = this->dx()[1];
  Stencil<T> deconvolveStencilRadial = (1.0*Shift(Point::Zeros())
                                    - (1.0/24.0)*Stencil<T>::Derivative(2,0,2));
  // Deconvolve W to obtain point values.
  State ownedPoint;
  if (!a_point) { ownedPoint = _deconvolveFace(a_WFace,a_WFaceBar,a_centering); a_point = &ownedPoint; }
  const State& WFacePoint = *a_point;
  FaceEnergies ownedPointEnergy;
  if (!a_pointEnergy)
  {
    ownedPointEnergy = faceEnergies(WFacePoint,a_block,a_centering);
    a_pointEnergy = &ownedPointEnergy;
  }
  const auto& kePoint = a_pointEnergy->kinetic;
  const auto& mePoint = a_pointEnergy->magnetic;
  const auto& vbPoint = a_pointEnergy->velocityDotMagnetic;

  BoxData<T,1,MEM> ownedRadial;
  if (!a_radialPoint)
  {
    if (a_centering != 0)
      {
        ownedRadial = deconvolveStencilRadial(a_adjDdir);
      }
    else
      {
        ownedRadial.define(a_adjDdir.box());
        a_adjDdir.copyTo(ownedRadial);
      }
    a_radialPoint = &ownedRadial;
  }
  const auto& adjDdirPoint = *a_radialPoint;

  State fluxPoint = forallOverwrite<T,NUMCOMPS>(f_primflux,WFacePoint,kePoint,mePoint,vbPoint,adjDdirPoint,m_gamma,a_block,a_centering);
  State ownedFluxBar;
  if (!a_fluxBar)
  {
    auto energyBar = faceEnergies(a_WFaceBar,a_block,a_centering);
    ownedFluxBar = forallOverwrite<T,NUMCOMPS>(f_primflux,a_WFaceBar,
        energyBar.kinetic,energyBar.magnetic,energyBar.velocityDotMagnetic,
        adjDdirPoint,m_gamma,a_block,a_centering);
    a_fluxBar = &ownedFluxBar;
  }

  a_physFlux = _convolveFace(fluxPoint,*a_fluxBar,a_centering);
  // Compute Average fluxes from pointwise values.
  //a_physFlux = (1.0*Shift(Point::Zeros()) +
  //             (1.0/24.0)*Stencil<T>::LaplacianFace(a_centering))(fluxPoint);
}


// ====================================================================================
// New factorization of operator implems ends here.
// ====================================================================================


// ====================================================================================
// Pre and Post implems start here.
// ====================================================================================
template<
    typename T,
    typename MAP,
    MemType MEM>
inline bool BoxOp_EulerCubedSphere<T,MAP,MEM>::validConservedState(
    const MBLevelBoxData<T,NUMCOMPS,MEM>& a_JU)
{
  Reduction<T,Operation::Sum,MEM> invalid;
  invalid.reset();
  if constexpr (MEM == HOST && DIM == 3)
  {
    PR_TIMERS("HelioCubed::validConservedStateHost");
    unsigned invalidFlag=0;
    for (auto dit : a_JU.layout())
    {
      const auto& data=a_JU[dit];
      const auto valid=a_JU.layout()[dit];
      const auto lo=valid.low(),hi=valid.high(),origin=data.box().low();
      const auto sizes=data.box().sizes();
      const size_t stride=data.box().size();
      const T* values=data.data();
      // The existing predicate requires every component to be finite and
      // density to be positive. Scan the SoA streams without a flag array.
      for (int c=0; c<NUMCOMPS && !invalidFlag; ++c)
        for (int k=lo[2]; k<=hi[2]; ++k) for (int j=lo[1]; j<=hi[1]; ++j)
        {
          size_t offset=c*stride+(lo[0]-origin[0])+size_t(sizes[0])*
              ((j-origin[1])+size_t(sizes[1])*(k-origin[2]));
          for (int i=lo[0]; i<=hi[0]; ++i,++offset)
          {
            const T value=values[offset];
            const bool bad=!std::isfinite(value) || (c==0 && !(value>0));
            if (bad && !invalidFlag)
            {
              std::ostringstream message;
              message << std::setprecision(17)
                  << "Invalid conserved cell: rank=" << procID()
                  << " block=" << a_JU.layout().block(dit)
                  << " index=(" << i << ',' << j << ',' << k << ")"
                  << " component=" << c << " value=" << value << '\n';
              std::cerr << message.str() << std::flush;
            }
            invalidFlag |= bad;
          }
        }
      if (invalidFlag) break;
    }
    // Every rank still participates, including ranks without local patches.
    const T localInvalid=invalidFlag ? T(1) : T(0);
    invalid.reduce(&localInvalid,1);
    return invalid.fetch()==0;
  }
  for (auto dit : a_JU.layout())
  {
    BoxData<T,1,MEM> bad(a_JU.layout()[dit]);
    forallInPlace([] PROTO_LAMBDA(
        Var<T,1,MEM>& a_bad, const Var<T,NUMCOMPS,MEM>& a_state)
      {
        a_bad(0) = MHD_Reconstruction::finiteConservedStateWithPositiveDensity<NUMCOMPS>(a_state) ? 0.0 : 1.0;
      },bad,a_JU[dit]);
    bad.reduce(invalid);
  }
  return invalid.fetch() == 0.0;
}

template<typename T, typename MAP, MemType MEM>
inline T BoxOp_EulerCubedSphere<T,MAP,MEM>::
dtCFL(MBLevelBoxData<T,NUMCOMPS,MEM>& a_JU,
      MBInterpOp& a_iop,
      MBLevelBoxData<T,1,MEM>& a_dVolr)
{
  PR_TIMERS("BoxOp_EulerCubedSphere::dtCFL()");
  if constexpr (MEM == HOST)
    if (usesSecondOrderPrimitives()) return dtCFLSecondOrderHost(a_JU,a_dVolr);
  auto layout = a_JU.layout();
  
  auto& geometry = levelGeometry(layout);
  auto& map = *geometry.map;
  auto& eulerOp = *geometry.op;



  MBLevelBoxData<T,NUMCOMPS,MEM,PR_CELL> USph(layout,Point::Zeros());
  a_JU.copyTo(USph);
  Reduction<T,Operation::Max,MEM> dtinv;
  dtinv.reset();
  for (auto dit : layout)
    {
      Box bx0 = layout[dit];
      auto block = layout.block(dit);
      Box bxblock = layout.domain().getBlock(block).box();
      T dx = 1./(bxblock.high()[2] + 1);
      BoxData<T,1,MEM> deta = CubedSphereShell::detA<T,MEM>(bx0,dx,0); 
      BoxData<T,1,MEM> detaInv = forall<T,1,MEM>
        ([ ] PROTO_LAMBDA
         (Var<T,1,MEM>& a_output, const Var<T,1,MEM>& a_deta)
         {
           a_output(0) = 1.0/a_deta(0);
         }
         ,deta);
      BoxData<T,DIM,MEM,DIM> deta_amat =
        CubedSphereShell::detA_A<T,MEM>(bx0,dx,block,0);
      BoxData<T,DIM,MEM,DIM> invcoef(bx0, T(0));
      forallInPlace(f_matinv3by3,invcoef,deta_amat);

      BoxData<T,1,MEM> JRho = slice(USph[dit],0);
      BoxData<T,1,MEM> JRhoE = slice(USph[dit],DIM+1);
      BoxData<T,DIM,MEM> JUMom = slice<T,NUMCOMPS,DIM,MEM>(USph[dit],1);
      BoxData<T,DIM,MEM> JUB = slice<T,NUMCOMPS,DIM,MEM>(USph[dit],DIM+2);
      JRho *= detaInv;
      JRhoE *= detaInv;
      BoxData<T,DIM,MEM> vectemp(layout[dit]);
      T one = 1.0;
      forallInPlace(f_matVecProd,vectemp,invcoef,JUMom,one);
      vectemp.copyTo(JUMom);
      forallInPlace(f_matVecProd,vectemp,invcoef,JUB,one);
      vectemp.copyTo(JUB);
      forallInPlace([] PROTO_LAMBDA(Var<T,NUMCOMPS,MEM>& a_usph,
                                    Var<T,1,MEM>& a_dvolr)
                    {
                      T dvolrInv = 1./a_dvolr(0);
                      for (int comp = 0; comp < NUMCOMPS; comp++)
                        {
                          a_usph(comp) *= dvolrInv;
                        }
                    },USph[dit],a_dVolr[dit]);
     }
  HDF5Handler h5;
  for (auto dit : layout)
    {
      int block = layout.block(dit);
      int globalIndex = dit.global();
      BoxData<T,NUMCOMPS,MEM> W(layout[dit]);
      BoxData<T,NUMCOMPS,MEM>& sph_i = USph[dit];
      BoxData<T,NUMCOMPS,MEM> sphTemp(layout[dit]);
      sph_i.copyTo(sphTemp);
      eulerOp[dit].sphToPrim_2O(W,sphTemp,block);
      eulerOp[dit].dtInv(dtinv,W);
    }
  return 1.0/dtinv.fetch(); 
}
template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::
StageLevel(MBLevelBoxData<T,NUMCOMPS,MEM>& a_k,
           MBLevelBoxData<T,NUMCOMPS,MEM>& a_UStage,
           MBBoundaryRegister<T,NUMREG,MEM>& a_blockreg,
           Array<Reduction<T,Operation::Sum,MEM>,NUMCOMPS>& a_consSum,
           T a_time, 
           T a_dt,
           T a_weight,
           int a_kstage)
{
  PR_TIMERS("BoxOp_EulerCubedSphere::StageLevel()");
  auto layout = a_UStage.layout();

  auto& geometry = levelGeometry(layout);
  auto& map = *geometry.map;
  auto& eulerOp = *geometry.op;

  T wgtxdt = a_weight*a_dt;
  if constexpr (MEM == HOST)
    if(usesSecondOrderPrimitives()) prepareSharedPanelStates(a_UStage);
  unsigned panelPatch=0;
  for (auto dit : a_UStage.layout())
  {
    PROTO_ASSERT(a_k[dit].box() == a_k.layout()[dit],"k_i isn't exactly the size of a rhs");
    auto &k_i = a_k[dit];
    auto &USph_i = a_UStage[dit];
    Array<BoxData<double, NUMCOMPS>, DIM> fluxes;
    fluxes[0].define(k_i.box().extrude(0));
    fluxes[1].define(k_i.box().extrude(1));
    fluxes[2].define(k_i.box().extrude(2));
    // The standard operator overwrites or copies every flux cell. Preserve
    // the existing initialization for the separate Berta operator.
    if (MEM != HOST || ParseInputs::get_spatial_operator() == 2)
    {
      fluxes[0].setToZero();
      fluxes[1].setToZero();
      fluxes[2].setToZero();
    }
    unsigned int block = layout.block(dit);
    eulerOp[dit].setTime(a_time);
    if constexpr (MEM == HOST)
      if(usesSecondOrderPrimitives())
        eulerOp[dit].m_panelStates=&geometry.secondOrderPanelStates->adjacent[panelPatch];
    eulerOp[dit](k_i, fluxes, USph_i, 1.0, 1.0, -1,
        usesSecondOrderPrimitives());
    eulerOp[dit].m_panelStates=nullptr;
    ++panelPatch;
    auto dx = eulerOp[dit].dx();
    Box blockBox = layout.getBlock(block).domain().box();
    a_blockreg.increment(fluxes,dit,wgtxdt);
    CubedSphereShell::radialCons<T,NUMCOMPS,MEM>(fluxes[0],a_consSum,dx[0],blockBox,wgtxdt); 
  }
}
template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::
dtInv(Reduction<T,Operation::Max,MEM>& a_dtInv,
      const State& a_W) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::dtInv()");
  Box bx= this->box();  
  T dxiR = this->dx()[0];
  T dxiPerp = this->dx()[1];
  // Match the switches used by preStageLevel() and viscFlux().
  const bool linearViscosityEnabled = (ParseInputs::get_linear_viscosity() == 1);
  const bool nonlinearViscosityEnabled = (ParseInputs::get_nonlinear_viscosity() == 1);
  const bool viscosityEnabled = linearViscosityEnabled || nonlinearViscosityEnabled;
  const bool scaleViscosityWithR =
    (ParseInputs::get_scale_viscosity_with_r() == 1);
  int block = this->block();
  Box bxRadius = bx;
  BoxData<T,DIM,MEM> Dr(bxRadius);
  BoxData<T,1,MEM> radius(bxRadius);
  BoxData<T,DIM,MEM> adjDr(bxRadius);
  BoxData<T,1,MEM> dVolr(bxRadius);
  BoxData<T,1,MEM> speedMax(bxRadius, T(0));
  radialMetrics(radius,Dr,adjDr,dVolr,bxRadius);
  T half = .5;
  BoxData<T,1,MEM> visccoef(bx.grow(2), T(0));
  for (int dir = 0 ; dir < DIM; dir++)
    {
      Box bxface = bx;
      BoxData<T,1,MEM> adjDrDir = slice(adjDr,dir); 
      BoxData<T,1,MEM> deta =
        CubedSphereShell::detA<T,MEM>(bxface,dxiPerp,dir);
      BoxData<T,DIM,MEM> ntrow = CubedSphereShell::adjA<T,MEM>(bxface,dxiPerp,block,dir);
      Array<Array<uint,DIM>,6> permute = {{1,2,0},{1,2,0},{1,0,2},{0,1,2},{1,0,2},{0,1,2}};
      Array<Array<int,DIM>,6> sign = {{1,-1,-1},{1,1,1},{1,-1,1},{1,1,1},{-1,1,1},{-1,-1,1}}; 
      Point high = bxface.high();
      Point low = bxface.low();
      high[0] = low[0];
      Box bx0(low,high);
      BoxData<T,DIM,MEM,DIM> A_matrix(bx);
      T offseta = half;
      T offsetb = half;
      if (dir == 1) offseta = 0.0;
      if (dir == 2) offsetb = 0.0;
      forallInPlace_p(f_Amatrix,bx0,A_matrix,permute[block],sign[block],
                      dxiPerp,offseta,offsetb);
      spreadSlice(A_matrix);
      forallInPlace(f_speedmax,
                    speedMax,a_W,A_matrix,
                    ntrow,adjDrDir,deta,
                    dVolr,Dr,dxiPerp,
                    dxiR,dir,m_gamma);

      if (!viscosityEnabled) continue;

      auto dxi = this->dx();
      State WFace = viscosityFaceState(a_W,dir);
      T D_rScale;
      double r0 = ParseInputs::get_inner_radius();
      double r1 = ParseInputs::get_outer_radius();
      double C_rad = ParseInputs::get_radial_stretch();
      T R_t = C_rad == 0.0 ? (r1 - r0) : (r1 - r0) / std::expm1(C_rad);
      if (dir == 0)
        {
          D_rScale = C_rad == 0.0 ? (r1 - r0)
            : R_t*std::expm1(C_rad*dxi[dir])/dxi[dir];
        }
      else
        {
          D_rScale = r0;
        }
      
      BoxData<T,1,MEM> dx(bxface);
      forallInPlace(f_dx,dx,ntrow,adjDrDir,deta,
                    dVolr,Dr,dxiPerp,dxiR,dir);
      if (linearViscosityEnabled)
        {
          BoxData<T,1,MEM> linvisccoef(bxface, T(0));
          LinearViscCoef(linvisccoef,WFace,deta,A_matrix,ntrow,block,dir);
          if (scaleViscosityWithR && dir == 0)
            {
              auto Drslice = slice(Dr,0);
              linvisccoef *= Drslice;
              linvisccoef /= D_rScale;
            }
          else if (scaleViscosityWithR)
            {
              linvisccoef *= radius;
              linvisccoef /= D_rScale;
            }
          // Convert the scaled damping coefficient to an inverse timescale.
          linvisccoef /= dx;
          visccoef += linvisccoef;
        }
      if (nonlinearViscosityEnabled)
        {
          BoxData<T,1,MEM> nlvisccoef(bxface, T(0));
          NonLinearViscCoef(nlvisccoef,WFace,Dr,adjDr,A_matrix,ntrow,dVolr,deta,dxi,block,dir);
          if (scaleViscosityWithR && dir == 0)
            {
              auto Drslice = slice(Dr,0);
              nlvisccoef *= Drslice;
              nlvisccoef /= D_rScale;
            }
          else if (scaleViscosityWithR)
            {
              nlvisccoef *= radius;
              nlvisccoef /= D_rScale;
            }
          nlvisccoef /= dx;
          visccoef += nlvisccoef;
        }      
    }
  speedMax.reduce(a_dtInv);
  if (viscosityEnabled) visccoef.reduce(a_dtInv);
}

// ====================================================================================
// Pre and Post implems end here.
// ====================================================================================


template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::
P_floor(MBLevelBoxData<T, NUMCOMPS, MEM> &a_JU,
    MBLevelBoxData<T, 1, MEM> &a_dVolrLev,
    MBInterpOp &a_iop,
    MBDisjointBoxLayout &a_layout,
    double a_time,
    double a_dt,
    double a_gamma)
{
  PR_TIMERS("BoxOp_EulerCubedSphere::P_floor()");
  // Magnetized tests store perturbation energy and magnetic field in a_JU.
  // The legacy mapped-state floor below cannot interpret that representation.
  if (isConstantMagneticFieldTest(ParseInputs::get_init_condition_type())) return;
  // auto map = CubedSphereShell::Map(a_JU.layout(), ghost());
  // auto eulerOp = CubedSphereShell::Operator<BoxOp_EulerCubedSphere, double, HOST>(map);
  // MBLevelBoxData<double, NUMCOMPS, HOST> USph(a_layout, ghost());
  // a_JU.exchange();
  // a_JU.copyTo(USph);

  // CubedSphereShell
  //   ::consToSphInterpEuler<T,NUMCOMPS,MEM,PR_CELL>(USph,a_iop,a_dVolrLev, 4,0);
  // // Transform a_JU -> USph.
  // int rCoord = CUBED_SPHERE_SHELL_RADIAL_COORD;
  // int thetaCoord = (rCoord + 1) % DIM;
  // int phiCoord = (rCoord + 2) % DIM;
  // MBLevelBoxData<double, 8, HOST> dstData(a_layout, Point::Basis(rCoord) + NGHOST*Point::Basis(thetaCoord) + NGHOST*Point::Basis(phiCoord));
  // if (ParseInputs::get_init_condition_type() == 0) BC_global.BoxData_to_BC<double,HOST>(dstData, map, a_time);
  // for (auto dit : a_layout)
  // {
  //     auto block = a_layout.block(dit);
  //     Box blockBox = a_layout.getBlock(block).domain().box();
  //     eulerOp[dit].PreStagePatch(USph[dit],dstData[dit],a_dVolrLev[dit],blockBox);
  //     Array<double, DIM> dx = eulerOp[dit].dx();
  //     BoxData<double, NUMCOMPS, HOST> W(USph[dit].box());
  //     BoxData<double, NUMCOMPS, HOST> WBar(USph[dit].box());
  //     eulerOp[dit].sphToPrim(W,WBar,USph[dit],block); // Transform to primitive variables, spherical velocities.
  //     BoxData<double, NUMCOMPS, HOST> W_point(USph[dit].box());
  //     W.copyTo(W_point);
  //     W_point += Stencil<double>::Laplacian()(W,-1./24.);  
  //     BoxData<double,1,HOST> ke(W_point.box());
  //     BoxData<double,1,HOST> me(W_point.box());
  //     eulerOp[dit].EnergyW(ke,me,W_point,block,-1);
  //     forallInPlace([ ] PROTO_LAMBDA
  //                 (Var<double, NUMCOMPS, HOST> &a_W,
  //                   const Var<double, 1, HOST> &a_ke,
  //                   const Var<double, 1, HOST> &a_me)                    
  //     { 
  //       double eps = 1.0e-10;
  //       a_W(iP) = std::max(a_W(iP),eps*(a_ke(0) + a_me(0)));
  //     },W_point, ke, me);
  //     eulerOp[dit].primToCons(USph[dit], W_point, a_dVolrLev[dit], a_gamma, dx[2], block);
  // }
  // USph.copyTo(a_JU);

  Point size = a_layout.domain().getBlock(0).sizes();
  double dxiPerp = 1.0/size[1];
  for (auto dit : a_layout)
  {
  auto& JU_i = a_JU[dit];
  BoxData<double,1,HOST> deta = CubedSphereShell::detA<double,HOST>(JU_i.box(),dxiPerp,0); 
  auto& dVolr_i = a_dVolrLev[dit];
  forallInPlace([ ] PROTO_LAMBDA
    (Var<double,NUMCOMPS,HOST>& l_JU,
      const Var<double,1,HOST>& l_deta,
      const Var<double,1,HOST>& l_dVolr,
      const double l_gamma)
    {
      double rho = l_JU(iRHO)/(l_deta(0)*l_dVolr(0));
      double Vx = l_JU(iMOMX)/l_JU(iRHO);
      double Vy = l_JU(iMOMY)/l_JU(iRHO);
      double Vz = l_JU(iMOMZ)/l_JU(iRHO);
      double Bx = l_JU(iBX)/(l_deta(0)*l_dVolr(0));
      double By = l_JU(iBY)/(l_deta(0)*l_dVolr(0));
      double Bz = l_JU(iBZ)/(l_deta(0)*l_dVolr(0));
      double p = (l_gamma - 1.0)*(l_JU(iE)/(l_deta(0)*l_dVolr(0)) - 0.5*rho*(Vx*Vx + Vy*Vy + Vz*Vz) - (Bx*Bx + By*By + Bz*Bz)/8.0/M_PI);
      if (p < 0.0) l_JU(iE) = l_deta(0)*l_dVolr(0)*(0.5*rho*(Vx*Vx + Vy*Vy + Vz*Vz) + (Bx*Bx + By*By + Bz*Bz)/8.0/M_PI);
    },JU_i,deta,dVolr_i,a_gamma);
  }

}

// Insert a CME by changing only its physical support. All geometry, field,
// loading and energy rules are shared with boundary injection in MHD_CME.H.
template<typename T, typename MAP, MemType MEM>
inline void BoxOp_EulerCubedSphere<T,MAP,MEM>::Insert_CME(
    MBLevelBoxData<T,NUMCOMPS,MEM>& state,
    MBLevelBoxData<T,1,MEM>& radialVolume,
    MBInterpOp& interpolation,
    MBDisjointBoxLayout& layout,
    double time, double dt, double gamma, const CMEParameters& cme)
{
    PR_TIME("BoxOp_EulerCubedSphere::Insert_CME()");
    auto& operators = *levelGeometry(layout).op;

    if (ParseInputs::get_spatial_order() == 2) {
        long long affected = 0;
        double addedMass = 0, baselineMass = 0, backgroundMass = 0, maxVa2 = 0;
        double minAdded = std::numeric_limits<double>::infinity(), maxAdded = 0;
        for (auto index : layout) {
            const Box valid = layout[index];
            auto& op = operators[index];
            const auto dx = op.dx();
            Scalar radius(valid), volume(valid);
            Vector dr(valid), area(valid);
            op.radialMetrics(radius, dr, area, volume, valid);
            auto position = forall_p<double,DIM,HOST>(f_cubedSphereMap3, valid,
                radius, dx, 0.5, 0.5, layout.block(index));
            auto determinant = CubedSphereShell::detA<double,HOST>(valid, dx[1], 0);
            auto& conserved = state[index];

            for (auto point : valid) {
                CMEModel::Sample source;
                if (!CMEModel::sample(cme,
                    {position(point,0), position(point,1), position(point,2)}, source)) continue;

                const double jacobian = volume(point) * determinant(point);
                std::array<double,8> ambient{};
                ambient[0] = conserved(point,iRHO) / jacobian;
                double kinetic = 0, magnetic = 0, totalFieldSquared = 0;
                for (int d = 0; d < 3; ++d) {
                    ambient[1+d] = conserved(point,iMOMX+d) / (jacobian * ambient[0]);
                    ambient[5+d] = conserved(point,iBX+d) / jacobian;
                    kinetic += 0.5 * ambient[0] * ambient[1+d] * ambient[1+d];
                    magnetic += ambient[5+d] * ambient[5+d] / (8 * M_PI);
                    totalFieldSquared += std::pow(ambient[5+d] + source.field[d], 2);
                }
                ambient[4] = std::max(0.,
                    (conserved(point,iE) / jacobian - kinetic - magnetic) * (gamma - 1));
                const double cellVolume = jacobian * dx[0] * dx[1] * dx[2];
                baselineMass += source.density * cellVolume;
                backgroundMass += ambient[0] * cellVolume;

                CMEModel::loadDensity(source, ambient);
                const auto change = CMEModel::increment(ambient, source, gamma, cme.plasmaBeta);
                for (int component = 0; component < 8; ++component)
                    conserved(point,component) += jacobian * change[component];

                ++affected;
                addedMass += source.density * cellVolume;
                minAdded = std::min(minAdded, source.density);
                maxAdded = std::max(maxAdded, source.density);
                maxVa2 = std::max(maxVa2,
                    totalFieldSquared / (4 * M_PI * (ambient[0] + source.density)));
            }
        }
#ifdef PR_MPI
        MPI_Allreduce(MPI_IN_PLACE, &affected, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &addedMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &baselineMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &backgroundMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maxVa2, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &minAdded, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maxAdded, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        if (affected == 0) {
            pout(0) << "CME source contains no resolved insertion cells" << std::endl;
            return;
        }
        pout(0) << std::setprecision(17)
            << "CME positive-energy insertion: cells " << affected
            << ", added mass " << addedMass << " g" << std::endl;
        pout(0) << "CME automatic density loading: baseline mass " << baselineMass
            << " g, max vA " << std::sqrt(maxVa2)/1e5 << " km/s"
            << ", added n range [" << minAdded/c_MP << ", " << maxAdded/c_MP << "] cm^-3"
            << ", enclosed background mass " << backgroundMass << " g"
            << std::setprecision(6) << std::endl;
        return;
    }

    // Fourth-order integration uses the same pointwise source, then maps its
    // increment back to cell averages. Keep the background outside the source.
    MBLevelBoxData<T,NUMCOMPS,MEM> native(layout, ghost());
    state.exchange();
    state.copyTo(native);
    prepareState(native, interpolation, radialVolume, time);
    for (auto index : layout) {
        auto& op = operators[index];
        const auto block = layout.block(index);
        const auto dx = op.dx();
        Scalar radius(native[index].box()), volume(native[index].box());
        Vector dr(native[index].box()), area(native[index].box());
        op.radialMetrics(radius, dr, area, volume, radius.box());
        State point, average;
        op.sphToPrim(point, average, native[index], block);
        point += Stencil<double>::Laplacian()(point, -1./24.);
        auto position = forall_p<double,DIM,HOST>(f_cubedSphereMap3,
            radius.box(), radius, dx, 0.5, 0.5, block);
        auto basis = op.pointwiseSphericalBasis(point.box(), block, -1);
        auto inverse = forallOverwrite<T,DIM,MEM,DIM>(f_matinv3by3, basis);
        State baseline;
        op.primToCons(baseline, point, radialVolume[index], gamma, dx[2], block);
        for (auto p : point.box()) {
            CMEModel::Sample source;
            if (!CMEModel::sample(cme, {position(p,0),position(p,1),position(p,2)}, source)) continue;
            std::array<double,8> ambient{};
            ambient[0] = point(p,iRHO);
            ambient[4] = std::max(0., point(p,iP));
            for (int d = 0; d < 3; ++d) for (int q = 0; q < 3; ++q) {
                ambient[1+d] += basis(p,d,q) * point(p,iVX+q);
                ambient[5+d] += basis(p,d,q) * point(p,iBX+q);
            }
            CMEModel::loadDensity(source, ambient);
            point(p,iRHO) = ambient[0] + source.density;
            point(p,iP) = CMEModel::thermalPressure(ambient, source, cme.plasmaBeta);
            for (int d = 0; d < 3; ++d) {
                point(p,iVX+d) = point(p,iBX+d) = 0;
                for (int q = 0; q < 3; ++q) {
                    point(p,iVX+d) += inverse(p,d,q) * source.velocity[q];
                    point(p,iBX+d) += inverse(p,d,q) * (ambient[5+q] + source.field[q]);
                }
            }
        }
        State inserted;
        op.primToCons(inserted, point, radialVolume[index], gamma, dx[2], block);
        inserted -= baseline;
        state[index] += inserted;
    }
}
