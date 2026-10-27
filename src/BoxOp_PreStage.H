#include "MHDReader.H"
#include "MHD_Constants.H"
#include <memory>  
extern MHDReader BC_global;
template <typename T, MemType MEM>
PROTO_KERNEL_START
void f_radialoutflowIBCs_F(
    Point a_pt,
    Var<T, NUMCOMPS, MEM> &a_USph,
    Var<T> &a_radius,
    Var<T, DIM> &a_X_cart,
    T a_gamma,
    int a_nradius,
    T a_time,
    T a_strength,
    double r_BC)
{
  T r = sqrt(a_X_cart(0) * a_X_cart(0) + a_X_cart(1) * a_X_cart(1) + a_X_cart(2) * a_X_cart(2));
  T theta = atan2(sqrt(a_X_cart(0) * a_X_cart(0) + a_X_cart(1) * a_X_cart(1)), a_X_cart(2)); // polar angle
  T phi = atan2(a_X_cart(1), a_X_cart(0)); // azimuthal angle
  // Compute spherical BCs.
  T p0 = 1.0e-7; // p near 21.5 c_SR is about 1e-7 dyne/cm2
  T rho0 = 700*c_MP; // rho at 21.5 c_SR is about 700/cm3
  T u0 = 500.0*1e5; // v at 21.5 c_SR is about 500 km/s
  T rho = rho0*pow(r_BC/r,2.0);
  T p = p0*pow(r_BC/r,2.0*a_gamma);
  double carr_rot_time = 25.38 * 24 * 60 * 60; // Seconds
  T omega = 2.0 * M_PI / carr_rot_time; 
  T phit  = phi - omega * a_time; // or + for opposite rotation
  const T perturbationScale = a_strength/100.0;
  const T sinTheta = sin(theta);
  // Match the smooth-on-the-sphere azimuthal profile used at initialization.
  const T angularPattern =
      pow(sinTheta,5)*sin(5*phit) + pow(sinTheta,10)*sin(10*phit);
  T ur = u0 - perturbationScale*u0/3
      * angularPattern;
  a_USph(iRHO) = rho;
  a_USph(iMOMX) = ur * rho;
  a_USph(iMOMY) = 0.0;
  a_USph(iMOMZ) = 0.0;
  a_USph(iBX) = 0.0;
  a_USph(iBY) = 0.0;
  a_USph(iBZ) = 0.0;
  T umag = sqrt(a_USph(iMOMX) * a_USph(iMOMX) + a_USph(iMOMY) * a_USph(iMOMY) + a_USph(iMOMZ) * a_USph(iMOMZ)) / a_USph(iRHO);
  T Bmag = sqrt(a_USph(iBX) * a_USph(iBX) + a_USph(iBY) * a_USph(iBY) + a_USph(iBZ) * a_USph(iBZ));
  a_USph(iE) = p / (a_gamma - 1.0) + 0.5 * a_USph(iRHO) * umag * umag + Bmag * Bmag/8.0/M_PI;
}
PROTO_KERNEL_END(f_radialoutflowIBCs_F, f_radialoutflowIBCs)

// Radial boundary conditions here.
template<
  typename T,
  typename MAP,
  MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::
PreStagePatch(BoxData<T,NUMCOMPS,MEM>& a_USphStage,
              BoxData<T,NUMCOMPS,MEM>& a_BC,
              BoxData<T,1,MEM>& a_dVolr,
              Box a_blockBox,
              T a_time)
{
  PR_TIMERS("BoxOp_EulerCubedSphere::PreStagePatch()");
  HDF5Handler h5;

  // if (procID() == 0) h5.writePatch(1,a_USphStage,"a_USphStage_noBC");

  int a_block = this->block();
  Box bxGhosted = a_USphStage.box();
  // O2 primitive stages pass thin boundary strips. Only those cells need
  // radial metrics; the level volume was already populated at setup.
  const Box metricBox=ParseInputs::get_spatial_order()==2 ? bxGhosted : a_dVolr.box();
  BoxData<double> radius(metricBox);
  BoxData<double, DIM, HOST> Dr(metricBox);
  BoxData<double, DIM, HOST> adjDr(metricBox);
  radialMetrics(radius, Dr, adjDr, a_dVolr, Dr.box());
  Array<double, DIM> dx;
  dx = this->dx();
  double dxradius = dx[0];
  double a_h = dx[2];
  int thickness = std::round(1.0 / dxradius);
  // Set radial boundary condtions in radial ghost cells.
  int init_condition_type = ParseInputs::get_init_condition_type();
  double innerRadius = ParseInputs::get_inner_radius();
  Point low = bxGhosted.low();
  low[0] = a_blockBox.high()[0] + 1;
  Box bdryBoxHigh(low, bxGhosted.high());
  Point high = bxGhosted.high();
  high[0] = a_blockBox.low()[0] - 1;
  Box bdryBoxLow(bxGhosted.low(), high);

  // Outer BCs
  if (a_blockBox.high()[0] < bxGhosted.high()[0])
  {
    if (init_condition_type == 2
        || init_condition_type == 4
        || init_condition_type == 6
        || init_condition_type == 8){
      forallInPlace_p(
          f_testProblemHydroBoundary,
          bdryBoxHigh,
          a_USphStage,
          m_gamma);
    } 
    if (init_condition_type == 0 || init_condition_type == 1)
      {
        // Copy last layer to ghost cells and scale by radius.
        // Extrapolation can make -ve P in ghost cells.
        Point source_low = Point(a_blockBox.high()[0],
                                 bxGhosted.low()[1],bxGhosted.low()[2]);
        Point source_high = Point(a_blockBox.high()[0],bxGhosted.high()[1],
                                  bxGhosted.high()[2]);
        Box sourceBox(source_low,source_high);
        State U_ghost(bdryBoxHigh);
        Scalar radius_ghost(bdryBoxHigh);
        Stencil<double> m_exterp_f_2nd;
        for (int i = 1; i <= bdryBoxHigh.sizes()[0]; i++ ) {
          a_USphStage.copyTo(U_ghost,sourceBox,Point::Basis(0)*(i));
          // Using shifting option of copyTo
          radius.copyTo(radius_ghost,sourceBox,Point::Basis(0)*(i));
        }
        U_ghost.copyTo(a_USphStage,bdryBoxHigh);
        //scale by radius.
        forallInPlace_p([ ] PROTO_LAMBDA
                        (Point a_pt,
                         Var<T, NUMCOMPS, MEM> &a_Usph,
                         Var<T> &a_radius,
                         Var<T> &a_radius_ghost)
                        {  
                          double rad = a_radius(0);
                          double rad_last = a_radius_ghost(0);
                          a_Usph(0) *= pow(rad_last/rad,2.0);
                          a_Usph(1) *= pow(rad_last/rad,2.0);
                          a_Usph(2) *= pow(rad_last/rad,2.0);
                          a_Usph(3) *= pow(rad_last/rad,2.0);
                          a_Usph(4) *= pow(rad_last/rad,2.0);
                          a_Usph(5) *= pow(rad_last/rad,2.0);
                          a_Usph(6) *= pow(rad_last/rad,1.0); 
                          a_Usph(7) *= pow(rad_last/rad,1.0);
                        },bdryBoxHigh,a_USphStage,radius,radius_ghost);
    }

    if (isConstantMagneticFieldTest(init_condition_type))
    {
      // Preserve the cell-averaged multiblock state at radial/angular
      // corners. Reconstructing an analytic point value here is inconsistent
      // with the fourth-order averages in a_USphStage and seeds visible block
      // seams in low-beta magnetic tests.
      Point sourceLow = bxGhosted.low();
      Point sourceHigh = bxGhosted.high();
      sourceLow[0] = a_blockBox.high()[0];
      sourceHigh[0] = a_blockBox.high()[0];
      const Box sourceBox(sourceLow, sourceHigh);
      for (int ii = 1; ii <= bdryBoxHigh.sizes()[0]; ++ii)
      {
        a_USphStage.copyTo(
            a_USphStage, sourceBox, Point::Basis(0)*ii);
      }
    }
  }
  // Inner BCs
  if (a_blockBox.low()[0] > bxGhosted.low()[0])
  {

    BoxData<T,DIM,MEM,DIM> A_matrix_inv;
    if (init_condition_type == 0 && ParseInputs::get_BC_conserved_averages() == 0)
    {
      if (ParseInputs::get_spatial_order() == 2)
      {
        auto basis=pointwiseSphericalBasis(bdryBoxLow,a_block,-1);
        A_matrix_inv=forallOverwrite<T,DIM,MEM,DIM>(f_matinv3by3,basis);
      }
      else
      {
      // Calculate A_matrix
      BoxData<T,DIM,MEM,DIM> amat = CubedSphereShell::detA_A<T,MEM>(bdryBoxLow,a_h,a_block,0);
      BoxData<T,1,MEM> deta = CubedSphereShell::detA<T,MEM>(bdryBoxLow,a_h,0);
      forallInPlace([ ] PROTO_LAMBDA
        (Var<T, DIM, MEM, DIM> &a_amat,
        Var<T, 1, MEM> &a_deta)
        {
          double inv_deta = 1.0 / a_deta(0);
          for (int i = 0; i < DIM; i++){
            for (int j = 0; j < DIM; j++){
              a_amat(i,j) = a_amat(i,j) * inv_deta;
            }
          }
        },amat,deta);
      A_matrix_inv.define(bdryBoxLow, T(0));
      forallInPlace(f_matinv3by3,A_matrix_inv,amat);
      }
    }

    // Find X_cart
    double half = 0.5;
    BoxData<double, DIM, HOST> XCart;
    if (init_condition_type == 1 || !ParseInputs::get_CMEs().empty()
        || (init_condition_type == 0 && ParseInputs::get_BC_conserved_averages() == 0))
    {
      // Only the inner radial ghost cells consume these coordinates.
      XCart = forall_p<double,DIM,HOST>
        (f_cubedSphereMap3,bdryBoxLow,radius,dx,half,half,a_block);
    }

    if (init_condition_type == 2
        || init_condition_type == 4
        || init_condition_type == 6
        || init_condition_type == 8){
      forallInPlace_p(
          f_testProblemHydroBoundary,
          bdryBoxLow,
          a_USphStage,
          m_gamma);
    }
    if (init_condition_type == 1){
      T strength = ParseInputs::get_strength();
      forallInPlace_p(
          f_radialoutflowIBCs,
          bdryBoxLow,
          a_USphStage,
          radius,
          XCart,
          m_gamma,
          thickness,
          a_time,
          strength,
          innerRadius);
    }
    // Legacy point sampling and averaged-basis conversion, retained for
    // the default fast BC path. The optional pressure halo can be wider than NGHOST.
    if (init_condition_type == 0 && ParseInputs::get_BC_conserved_averages() == 0){
      Point source_lo = Point(a_BC.box().low()[0],a_BC.box().low()[1],a_BC.box().low()[2]);
      Point source_hi = Point(a_BC.box().low()[0],a_BC.box().high()[1],a_BC.box().high()[2]);
      Box sourceBox(source_lo,source_hi);
      for (int i = 0; i < bdryBoxLow.sizes()[0]; i++){
        a_BC.copyTo(a_USphStage,sourceBox,Point::Basis(0)*(-i));
      }

      // Convert classical spherical vectors to Cartesian.
      forallInPlace([ ] PROTO_LAMBDA
      (Var<double, NUMCOMPS, HOST> &a_W,
      Var<double, DIM> &a_X_cart)
      {
        double x = a_X_cart(0);
        double y = a_X_cart(1);
        double z = a_X_cart(2);
        double r = sqrt(x*x + y*y + z*z);
        double theta = acos(z/r);
        double phi = atan2(y,x);
        // Convert spherical to Cartesian velocities
        double u_r = a_W(iVX);
        double u_theta = a_W(iVZ); //BC files are written as r,phi,theta
        double u_phi = a_W(iVY);
        a_W(iVX) = sin(theta)*cos(phi)*u_r + cos(theta)*cos(phi)*u_theta - sin(phi)*u_phi;
        a_W(iVY) = sin(theta)*sin(phi)*u_r + cos(theta)*sin(phi)*u_theta + cos(phi)*u_phi;
        a_W(iVZ) = cos(theta)*u_r - sin(theta)*u_theta;
        // Convert spherical to Cartesian magnetic fields
        double B_r = a_W(iBX);
        double B_theta = a_W(iBZ); //BC files are written as r,phi,theta
        double B_phi = a_W(iBY);
        a_W(iBX) = sin(theta)*cos(phi)*B_r + cos(theta)*cos(phi)*B_theta - sin(phi)*B_phi;
        a_W(iBY) = sin(theta)*sin(phi)*B_r + cos(theta)*sin(phi)*B_theta + cos(phi)*B_phi;
        a_W(iBZ) = cos(theta)*B_r - sin(theta)*B_theta;
      },bdryBoxLow,a_USphStage,XCart);

      BoxData<T,NUMCOMPS,MEM> USph_stage_temp(bdryBoxLow);
      a_USphStage.copyTo(USph_stage_temp);

      // Convert from cartesian to cubed-spherical Primitive variables w=A_inv*u
      BoxData<T,DIM,MEM> V_sph = slice<T,NUMCOMPS,DIM,MEM>(a_USphStage,iVX);
      BoxData<T,DIM,MEM> V_cart = slice<T,NUMCOMPS,DIM,MEM>(USph_stage_temp,iVX);
      BoxData<T,DIM,MEM> tmpV(USph_stage_temp.box());
      T one = 1.0;
      forallInPlace(f_matVecProd,tmpV,A_matrix_inv,V_cart,one);
      tmpV.copyTo(V_sph);

      BoxData<T,DIM,MEM> B_sph = slice<T,NUMCOMPS,DIM,MEM>(a_USphStage,iBX);
      BoxData<T,DIM,MEM> B_cart = slice<T,NUMCOMPS,DIM,MEM>(USph_stage_temp,iBX);
      BoxData<T,DIM,MEM> tmpB(USph_stage_temp.box());
      forallInPlace(f_matVecProd,tmpB,A_matrix_inv,B_cart,one);
      tmpB.copyTo(B_sph);

      // Convert to native conserved variables and apply the radial scaling.
      forallInPlace_p([ ] PROTO_LAMBDA
      (Point a_pt,
       Var<T, NUMCOMPS, MEM> &a_USph,
       Var<T, DIM, MEM> &a_V_cart,
       Var<T, DIM, MEM> &a_B_cart,
       Var<T> &a_radius,
       T a_gamma,
       double r_BC)
      {
        double rad = a_radius(0);
        double rho = a_USph(0)*1.67262192e-24*pow(r_BC/rad,2.0);
        double u = a_USph(1);
        double v = a_USph(2);
        double w = a_USph(3);
        double p = a_USph(4)*1.0e-12*pow(r_BC/rad,2.0*a_gamma); // From picodyne to dyne
        double Bx = a_USph(5)*1.0e-6*pow(r_BC/rad,2.0); // MicroGauss to Gauss
        double By = a_USph(6)*1.0e-6*pow(r_BC/rad,1.0);
        double Bz = a_USph(7)*1.0e-6*pow(r_BC/rad,1.0);
        double V_cart_mag2 = a_V_cart(0)*a_V_cart(0) + a_V_cart(1)*a_V_cart(1) + a_V_cart(2)*a_V_cart(2);
        double B_cart_mag2 = a_B_cart(0)*a_B_cart(0) + a_B_cart(1)*a_B_cart(1) + a_B_cart(2)*a_B_cart(2);
        double B_rad_mag2 = a_USph(5)*a_USph(5);
        double B_non_rad_mag2 = B_cart_mag2 - B_rad_mag2;
        double B_mag2_scaled = 1.0e-12*B_non_rad_mag2*pow(r_BC/rad,2.0) + 1.0e-12*B_rad_mag2*pow(r_BC/rad,4.0);
        double e = p/(a_gamma-1.0) + rho*(V_cart_mag2)/2.0 + (B_mag2_scaled)/8.0/M_PI;
        a_USph(0) = rho; //rho
        a_USph(1) = rho*u; //Momentum-x
        a_USph(2) = rho*v; //Momentum-y
        a_USph(3) = rho*w; //Momentum-z
        a_USph(4) = e; //Energy
        a_USph(5) = Bx; //Bx
        a_USph(6) = By; //By
        a_USph(7) = Bz; //Bz
      },bdryBoxLow,a_USphStage,V_cart,B_cart,radius,m_gamma,
        innerRadius);
    }

    if (init_condition_type == 0 && ParseInputs::get_BC_conserved_averages() == 1)
    {
      // The reader supplies native conserved cell averages (center values
      // at order two), including radial/angular corner ghosts.
      PROTO_ASSERT(a_BC.box().containsBox(bdryBoxLow),
          "Inner boundary averages do not cover the radial ghost region");
      a_BC.copyTo(a_USphStage,bdryBoxLow);
    }

    const auto& sources=ParseInputs::get_CMEs();
    const bool activeBoundarySource=std::any_of(sources.begin(),sources.end(),
        [&](const auto& source){return CMEModel::boundaryTaperWeight(source,a_time)>0;});
    if(ParseInputs::get_spatial_order()==2 && activeBoundarySource) {
      // Re-evaluate the analytic source at every RK stage. Ghosts are first
      // reset to the WSA boundary above, so overlapping sources and restarts
      // do not accumulate density or B repeatedly in boundary storage.
      auto basis=pointwiseSphericalBasis(bdryBoxLow,a_block,-1);
      auto inverse=forallOverwrite<T,DIM,MEM,DIM>(f_matinv3by3,basis);
      for(const auto& source:sources) {
        const double weight=CMEModel::boundaryTaperWeight(source,a_time);
        if(weight<=0) continue;
        CMEParameters current;
        if(!CMEModel::expandedSource(source,a_time,current)) continue;
        for(auto p:bdryBoxLow) {
          CMEModel::Sample cme;
          const CMEModel::Vector position={XCart(p,0),XCart(p,1),XCart(p,2)};
          // Gate the ghost footprint on the physical boundary sphere. A
          // coarse ghost center can already be inside the rope when its
          // front has only just touched the boundary at a single point.
          const double radius=std::sqrt(position[0]*position[0]+position[1]*position[1]+position[2]*position[2]);
          CMEModel::Sample faceSample;
          if(!CMEModel::sample(current,{position[0]*innerRadius/radius,
              position[1]*innerRadius/radius,position[2]*innerRadius/radius},faceSample)) continue;
          if(!CMEModel::sample(current,position,cme)) continue;
          // Load the full boundary source first. Its mass, field and excess
          // velocity are faded together below, leaving exactly WSA at expiry.
          cme.density=source.density;
          std::array<double,8> old{};old[0]=a_USphStage(p,iRHO);
          double ke=0,me=0,radialOld=0,radialCME=0;
          for(int d=0;d<3;++d) {
            for(int q=0;q<3;++q) {
              old[1+d]+=basis(p,d,q)*a_USphStage(p,iMOMX+q)/old[0];
              old[5+d]+=basis(p,d,q)*a_USphStage(p,iBX+q);
            }
            ke+=.5*old[0]*old[1+d]*old[1+d];me+=old[5+d]*old[5+d]/(8*M_PI);
            radialOld+=old[1+d]*position[d]/radius;radialCME+=cme.velocity[d]*position[d]/radius;
          }
          old[4]=std::max(0.,(a_USphStage(p,iE)-ke-me)*(m_gamma-1));
          // As in Provornikova et al. (2024), return to the ambient flow
          // speed when the homologous driver becomes slower than the wind.
          for(int d=0;d<3;++d)
            cme.velocity[d]=old[1+d]+std::max(0.,radialCME-radialOld)*position[d]/radius;
          CMEModel::loadDensity(cme, old, std::max(radialOld,radialCME));
          CMEModel::taperBoundarySource(cme,old,weight);
          const auto delta=CMEModel::increment(old,cme,m_gamma,source.plasmaBeta);
          // Prescribing all primitive variables is justified only for a
          // super-fast inflow. Expose violations in this experimental driver
          // rather than silently changing the observed magnetic flux.
          double b2=0,bn=0;
          for(int d=0;d<3;++d) {
            const double b=old[5+d]+cme.field[d];b2+=b*b;bn+=b*position[d]/radius;
          }
          const double rho=old[0]+cme.density;
          const double pressure=CMEModel::thermalPressure(old,cme,source.plasmaBeta);
          const double cs2=m_gamma*pressure/rho,va2=b2/(4*M_PI*rho),van2=bn*bn/(4*M_PI*rho);
          const double fast2=.5*(cs2+va2+std::sqrt(std::max(0.,(cs2+va2)*(cs2+va2)-4*cs2*van2)));
          const double radialInflow=radialOld+weight*std::max(0.,radialCME-radialOld);
          const double mach=radialInflow/std::sqrt(fast2);
          static std::vector<int> warnedSources;
          if(!(mach>1) && std::find(warnedSources.begin(),warnedSources.end(),source.sourceIndex)==warnedSources.end()) {
            warnedSources.push_back(source.sourceIndex);
            std::cerr<<"CME boundary diagnostic: rank="<<procID()<<", source="<<source.sourceIndex
                <<", time="<<a_time<<", fast Mach="<<mach<<", density="<<rho/c_MP
                <<" cm^-3. Prescribed-primitive injection is sub-fast here and needs a characteristic boundary for physical accuracy.\n";
          }
          a_USphStage(p,iRHO)+=delta[0];a_USphStage(p,iE)+=delta[4];
          for(int d=0;d<3;++d) for(int q=0;q<3;++q) {
            a_USphStage(p,iMOMX+d)+=inverse(p,d,q)*delta[1+q];
            a_USphStage(p,iBX+d)+=inverse(p,d,q)*delta[5+q];
          }
        }
      }
    }

    if (isConstantMagneticFieldTest(init_condition_type))
    {
      Point sourceLow = bxGhosted.low();
      Point sourceHigh = bxGhosted.high();
      sourceLow[0] = a_blockBox.low()[0];
      sourceHigh[0] = a_blockBox.low()[0];
      const Box sourceBox(sourceLow, sourceHigh);
      for (int ii = 1; ii <= bdryBoxLow.sizes()[0]; ++ii)
      {
        a_USphStage.copyTo(
            a_USphStage, sourceBox, Point::Basis(0)*(-ii));
      }
    }
  }
  // if (procID() == 0) h5.writePatch(1,a_USphStage,"a_USphStage_withBC");
}


/// Exchange, block boundary conditions, transform from JU to USph. Set state variables in boundary registers.
template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::
preStageLevel(MBLevelBoxData<T,NUMCOMPS,MEM>& a_UStage,
              MBLevelBoxData<T,NUMCOMPS,MEM>& a_k,
              MBLevelBoxData<T,1,MEM>& a_dVolrLev,
              MBInterpOp& a_iop,
              MBBoundaryRegister<T,NUMREG,MEM>& a_blockreg,
              Array<Reduction<T,Operation::Sum,MEM>,NUMCOMPS>& a_consSum,
              T a_time,
              T a_dt,
              T a_weight,
              int a_kstage,
              bool a_flag_viscosity)
{
  
  PR_TIMERS("BoxOp_EulerCubedSphere::PreStageLevel()");
  auto layout = a_UStage.layout();

  auto& geometry = levelGeometry(layout);
  auto& eulerOp = *geometry.op;

  // Stage storage is scratch: Cartesian primitives at O2, native conserved
  // variables for the higher-order and artificial-viscosity paths.
  if (usesSecondOrderPrimitives())
    prepareSecondOrderPrimitives(a_UStage,a_iop,a_dVolrLev,a_time);
  else
    prepareState(a_UStage,a_iop,a_dVolrLev,a_time);

  // initialize a_k to zero. 
  a_k.setVal(0.);
  for (auto dit : a_UStage.layout())
  {
    unsigned int block = layout.block(dit);
    Box blockBox = layout.getBlock(block).domain().box();
    auto &rhs_i = a_k[dit];
    auto &USph_i =a_UStage[dit];
    
    auto dx = eulerOp[dit].dx();    
    // Compute viscous operators here, and increment a_rhsTotal.
    const bool viscosityEnabled =
      (ParseInputs::get_linear_viscosity() == 1 ||
       ParseInputs::get_nonlinear_viscosity() == 1);
    if (a_flag_viscosity)
      {
        PR_TIMERS("artificial viscosity");
        unsigned int block = layout.block(dit);
        Box blockBox = layout.getBlock(block).domain().box();
        // auto mbdata = a_blockreg.bounds(dit);
        if (viscosityEnabled)
        {
          Array<BoxData<double, NUMCOMPS>, DIM> fluxes;
          fluxes[0].define(rhs_i.box().extrude(0), 0.0);
          fluxes[1].define(rhs_i.box().extrude(1), 0.0);
          fluxes[2].define(rhs_i.box().extrude(2), 0.0);
          eulerOp[dit].viscFlux(rhs_i, fluxes, USph_i,a_dt);
          // viscFlux updates the state with +div(Fvisc). The register and
          // radial balance use the conservative -div(F) convention.
          const T sourceWeight = ParseInputs::get_temporal_order() == 2 ? a_weight : T(1);
          a_blockreg.increment(fluxes, dit, -a_dt*sourceWeight);
          CubedSphereShell::radialCons<T,NUMCOMPS,MEM>(fluxes[0],a_consSum,dx[0],blockBox,-a_dt*sourceWeight);
        }
        if (ParseInputs::get_Sun_gravity() == 1)
        {
          if (usesSecondOrderPrimitives())
            eulerOp[dit].gravitySecondOrderPrimitives(rhs_i,USph_i);
          else eulerOp[dit].Gravity_Euler(rhs_i,USph_i);
        }
      }
  }
}
