#pragma once
#ifndef _BOX_OP_EULER_
#define _BOX_OP_EULER_

#include "Proto.H"
#include "Inputs_Parsing.H"
#include "MHD_StateConversion.H"
#include "MHD_Reconstruction.H"
#include "MHD_SecondOrderFlux.H"
#include <array>
#include <vector>

#define NUMCOMPS 2*DIM+2
#define NUMREG NUMCOMPS
#define NGHOST 5
#if DIM == 2
#define iRHO 0
#define iVX 1
#define iMOMX 1
#define iVY 2
#define iMOMY 2
#define iE 3
#define iP 3
#define iBX 4
#define iBY 5
#if TURB == 1
	#define iRHOZ2 6
	#define iZ2 6
	#define iRHOZ2SIGMA 7
	#define iSIGMA 7
	#define iRHOLAMBDA 8
	#define iLAMBDA 8
#endif
#endif
#define VEL00 1.E5
#define RHO00  1.67262192e-24/100.
#define P00 1.E12
#define B00 1.E6
#if DIM == 3
#define iRHO 0
#define iVX 1
#define iMOMX 1
#define iVY 2
#define iMOMY 2
#define iVZ 3
#define iMOMZ 3
#define iE 4
#define iP 4
#define iBX 5
#define iBY 6
#define iBZ 7
#if TURB == 1
	#define iRHOZ2 8
	#define iZ2 8
	#define iRHOZ2SIGMA 9
	#define iSIGMA 9
	#define iRHOLAMBDA 10
	#define iLAMBDA 10
#endif
#endif

using namespace Proto;
typedef BoxData<double> Scalar;
typedef BoxData<double, NUMCOMPS> State;
typedef BoxData<double, DIM> Vector;
#define BLOCKTEST -1

template<typename T,MemType MEM>
PROTO_KERNEL_START
void f_sphToPrim_F(
                   Var<T,NUMCOMPS,MEM>& a_wpoint,
                   Var<T,NUMCOMPS,MEM>& a_upoint)                           
{
  Var<T,NUMCOMPS,MEM> wpoint;
  
  a_wpoint(iRHO) = max(a_upoint(iRHO),RHO00);
  a_wpoint(iVX) =
    a_upoint(iMOMX)/a_wpoint(iRHO);
  a_wpoint(iVY) =
    a_upoint(iMOMY)/a_wpoint(iRHO);
  a_wpoint(iVZ) =
    a_upoint(iMOMZ)/a_wpoint(iRHO);
  a_wpoint(iE) = a_upoint(iP);  
  a_wpoint(iBX) = a_upoint(iBX);
  a_wpoint(iBY) = a_upoint(iBY);
  a_wpoint(iBZ) = a_upoint(iBZ);
  
}
PROTO_KERNEL_END(f_sphToPrim_F,f_sphToPrim)
PROTO_KERNEL_START
void
f_thresholdF(
             Var<short>& a_tags,
             Var<double, NUMCOMPS>& a_U)
{
  double thresh = 1.001;
  if (a_U(0) > thresh) {a_tags(0) = 1;}
  else {a_tags(0) = 0;};
}

PROTO_KERNEL_END(f_thresholdF, f_threshold);
template<typename T,MemType MEM>
PROTO_KERNEL_START
void f_dx_F
(
 Var<T,1,MEM>& a_dx,
 const Var<T,DIM,MEM>& a_adjA,
 const Var<T,1,MEM>& a_adjDrDir,
 const Var<T,1,MEM>& a_detA,
 const Var<T,1,MEM>& a_dVolr,
 const Var<T,DIM,MEM>& a_Dr,
 const T a_dxiPerp,
 const T a_dxiR,
 const int a_d)
{ 
  double dx_d;
  if (a_d > 0){
    T adjA_mag = sqrt(a_adjA(0)*a_adjA(0) + a_adjA(1)*a_adjA(1) + a_adjA(2)*a_adjA(2));
    dx_d = a_dVolr(0) * a_detA(0) * a_dxiPerp / a_adjDrDir(0) / adjA_mag;
  } 
  else{
    dx_d = a_Dr(0) * a_dxiR;
  }
  a_dx(0) = dx_d;
}
PROTO_KERNEL_END(f_dx_F, f_dx);
template<typename T,MemType MEM>
PROTO_KERNEL_START
void f_speedmax_F
(
 Var<T,1,MEM>& a_speedmax,
 const Var<T,NUMCOMPS,MEM>& a_W,
 const Var<T,DIM,MEM,DIM>& a_amat,
 const Var<T,DIM,MEM>& a_adjA,
 const Var<T,1,MEM>& a_adjDrDir,
 const Var<T,1,MEM>& a_detA,
 const Var<T,1,MEM>& a_dVolr,
 const Var<T,DIM,MEM>& a_Dr,
 const T a_dxiPerp,
 const T a_dxiR,
 const int a_d,
 const T a_gamma)
{ 
  double dx_d, dt_d;
  if (a_d > 0){
    T adjA_mag = sqrt(a_adjA(0)*a_adjA(0) + a_adjA(1)*a_adjA(1) + a_adjA(2)*a_adjA(2));
    dx_d = a_dVolr(0) * a_detA(0) * a_dxiPerp / a_adjDrDir(0) / adjA_mag;
  } 
  if (a_d == 0){
    dx_d = a_Dr(0) * a_dxiR;
  }   
  vecType velSph;
  vecType velCart;
  vecType bSph;
  vecType bCart;

  velSph[0] = a_W(iVX);       
  velSph[1] = a_W(iVY);
  velSph[2] = a_W(iVZ);
  velCart[0] = 0.;       
  velCart[1] = 0.;
  velCart[2] = 0.;
      
  bSph[0] = a_W(iBX);       
  bSph[1] = a_W(iBY);
  bSph[2] = a_W(iBZ);
  bCart[0] = 0.;       
  bCart[1] = 0.;
  bCart[2] = 0.;
  T half = .5;
  T one = 1.0;
  T v2 = 0.;
  T v1 = 0.;
  T b2 = 0.;
  for (int dir0 = 0; dir0 < DIM; dir0++)
    {
      for (int dir1 = 0; dir1 < DIM; dir1++)
        {
          velCart[dir0] += a_amat(dir0,dir1)*velSph[dir1];
          bCart[dir0] += a_amat(dir0,dir1)*bSph[dir1];
        }
      v2 += velCart[dir0]*velCart[dir0];
      v1 += abs(velCart[dir0]);
      b2 += bCart[dir0]*bCart[dir0];
    }
  T rho = a_W(iRHO);
  T B_mag = sqrt(b2);
  T p = a_W(iP);
  if (p < 0.0) p = 0.0;
  rho = max(rho,RHO00);
  T af = sqrt(a_gamma*p/rho + b2/4.0/M_PI/rho);  
  a_speedmax(0) += (af + sqrt(v2))/dx_d;
}
PROTO_KERNEL_END(f_speedmax_F, f_speedmax);
template<typename T,MemType MEM>
PROTO_KERNEL_START
void f_primflux_F
     (
      Var<T,NUMCOMPS,MEM>& a_flux,
      const Var<T,NUMCOMPS,MEM>& a_wface,      
      const Var<T,1,MEM>& a_ke,
      const Var<T,1,MEM>& a_me,
      const Var<T,1,MEM>& a_vb,
      const Var<T,1,MEM>& a_adjddir,
      const T a_gamma,
      uint a_block,
      uint a_centering)
     {
       T velnorm = a_wface(a_centering+iVX);
       T veltan1 = a_wface((a_centering+1)%DIM + iVX);
       T veltan2 = a_wface((a_centering+2)%DIM + iVX);
       T bnorm = a_wface(a_centering+iBX);
       T btan1 = a_wface((a_centering+1)%DIM + iBX);
       T btan2 = a_wface((a_centering+2)%DIM + iBX);
       a_flux(iRHO) = a_wface(iRHO)*velnorm;
       // The pressure term in momentum fluxes will be added later in mapped flux calculation.
       a_flux(a_centering+iMOMX) = a_flux(iRHO)*velnorm - bnorm*bnorm/4.0/M_PI;
       a_flux((a_centering+1)%DIM + iMOMX) = a_flux(iRHO)*veltan1 - bnorm*btan1/4.0/M_PI;
       a_flux((a_centering+2)%DIM + iMOMX) = a_flux(iRHO)*veltan2 - bnorm*btan2/4.0/M_PI;
       a_flux(iE) = (2.0*a_me(0) + a_ke(0) + a_wface(iP)*a_gamma/(a_gamma - 1.0))*velnorm - bnorm*a_vb(0)/4.0/M_PI;
       a_flux(iBX) = velnorm*a_wface(iBX) - a_wface(iVX)*bnorm;
       a_flux(iBY) = velnorm*a_wface(iBY) - a_wface(iVY)*bnorm;
       a_flux(iBZ) = velnorm*a_wface(iBZ) - a_wface(iVZ)*bnorm;
     }
PROTO_KERNEL_END(f_primflux_F, f_primflux);
template<typename T,MemType MEM>
PROTO_KERNEL_START
void f_csq_F
(
 Var<T,1,MEM>& a_csq,
 const Var<T,NUMCOMPS,MEM>& a_W,
 const Var<T,DIM,MEM,DIM>& a_amat,
 const T a_gamma)
{ 
  vecType velSph;
  vecType velCart;
  vecType bSph;
  vecType bCart;

  velSph[0] = a_W(iVX);       
  velSph[1] = a_W(iVY);
  velSph[2] = a_W(iVZ);
  velCart[0] = 0.;       
  velCart[1] = 0.;
  velCart[2] = 0.;
      
  bSph[0] = a_W(iBX);       
  bSph[1] = a_W(iBY);
  bSph[2] = a_W(iBZ);
  bCart[0] = 0.;       
  bCart[1] = 0.;
  bCart[2] = 0.;
  T half = .5;
  T one = 1.0;
  T v2 = 0.;
  T v1 = 0.;
  T b2 = 0.;
  for (int dir0 = 0; dir0 < DIM; dir0++)
    {
      for (int dir1 = 0; dir1 < DIM; dir1++)
        {
          velCart[dir0] += a_amat(dir0,dir1)*velSph[dir1];
          bCart[dir0] += a_amat(dir0,dir1)*bSph[dir1];
        }
      v2 += velCart[dir0]*velCart[dir0];
      v1 += abs(velCart[dir0]);
      b2 += bCart[dir0]*bCart[dir0];
    }
  T rho = a_W(iRHO);
  T B_mag = sqrt(b2);
  T p = a_W(iP);
  if (p < 0.0) p = 0.0;
  rho = max(rho,RHO00);
  a_csq(0) = a_gamma*p/rho + b2/4.0/M_PI/rho;  
}
PROTO_KERNEL_END(f_csq_F, f_csq);
template<
    typename T,
    typename MAP,
    MemType MEM = MEMTYPE_DEFAULT>
class BoxOp_EulerCubedSphere : public MBBoxOp<T, MAP, NUMCOMPS, 0, MEM>
{
public:

  using LayoutT = MBDisjointBoxLayout;
  using MapT = decltype(CubedSphereShell::Map(std::declval<const LayoutT&>(),Point()));
  using OpT = decltype(CubedSphereShell::Operator<BoxOp_EulerCubedSphere, double, HOST>(std::declval<MapT&>()));
  using SecondOrderFaceMask = Array<BoxData<unsigned char,1,HOST>,DIM>;
  using PanelStates = std::array<BoxData<T,NUMCOMPS,MEM>,2*DIM>;

  // Short-lived derived values from one final face state. The caller owns
  // these inside the current patch/direction/stage scratch scope.
  struct FaceEnergies
  {
    BoxData<T,1,MEM> kinetic, magnetic, velocityDotMagnetic;
  };

  inline BoxData<T,DIM,MEM,DIM> pointwiseSphericalBasis(
      const Box& a_box, uint a_block, uint a_centering) const;

  inline FaceEnergies faceEnergies(
      const State& a_point, uint a_block, uint a_dir,
      const BoxData<T,DIM,MEM,DIM>* a_basis = nullptr) const
  {
    FaceEnergies values;
    // EnergyW initializes and assigns all three outputs over the full state box.
    values.kinetic.defineForOverwrite(a_point.box());
    values.magnetic.defineForOverwrite(a_point.box());
    values.velocityDotMagnetic.defineForOverwrite(a_point.box());
    EnergyW(values.kinetic,values.magnetic,values.velocityDotMagnetic,a_point,a_block,a_dir,a_basis);
    return values;
  }

  struct SecondOrderCFLCache;
  struct SecondOrderBoundaryCache;
  struct SecondOrderPanelStateCache;
  struct LevelGeometry
  {
    std::unique_ptr<MapT> map;
    std::unique_ptr<OpT> op;
    // AMR alternates layouts; each level retains its own immutable CFL metrics.
    std::shared_ptr<SecondOrderCFLCache> secondOrderCFL;
    std::shared_ptr<SecondOrderBoundaryCache> secondOrderBoundary;
    std::shared_ptr<SecondOrderPanelStateCache> secondOrderPanelStates;
  };

  // An AMR hierarchy owns several geometries simultaneously. Bind its level
  // while using shared boundary/CFL helpers, without replacing the single-
  // level driver's retained geometry or rebuilding maps at every stage.
  static LevelGeometry*& boundLevelGeometry()
  {
    static thread_local LevelGeometry* bound = nullptr;
    return bound;
  }
  class GeometryScope
  {
    LevelGeometry* previous;
  public:
    explicit GeometryScope(LevelGeometry& geometry)
      :previous(boundLevelGeometry()) {boundLevelGeometry()=&geometry;}
    ~GeometryScope() {boundLevelGeometry()=previous;}
    GeometryScope(const GeometryScope&)=delete;
    GeometryScope& operator=(const GeometryScope&)=delete;
  };

  // All stages use the same immutable grid geometry. The partition remains
  // owned by the cached map; its identity also detects a new convergence grid.
  static LevelGeometry& levelGeometry(const LayoutT& layout)
  {
    if (auto* bound=boundLevelGeometry())
    {
      PROTO_ASSERT(&bound->map->layout().partition()==&layout.partition(),
          "The bound AMR geometry does not match the requested level");
      return *bound;
    }
    static LevelGeometry cached;
    if (!cached.map || &cached.map->layout().partition() != &layout.partition())
    {
      cached.secondOrderCFL.reset();
      cached.secondOrderBoundary.reset();
      cached.secondOrderPanelStates.reset();
      cached.op.reset();
      cached.map = std::make_unique<MapT>(CubedSphereShell::Map(layout, ghost()));
      cached.op = std::make_unique<OpT>(
          CubedSphereShell::Operator<BoxOp_EulerCubedSphere, double, HOST>(*cached.map));
    }
    return cached;
  }

  using MBBoxOp<T, MAP, NUMCOMPS,0,MEM>::MBBoxOp;

  T m_gamma = ParseInputs::get_gamma();
  mutable T umax;

  // How many ghost cells does the operator need from the state variables
  inline static Point ghost()
  {
    if (ParseInputs::get_spatial_order() == 2)
    {
      // PLM and nonlinear viscosity need two layers. D5 linear viscosity
      // reaches three cells beyond the owned box at its boundary faces.
      const bool fifthDerivative = ParseInputs::get_linear_viscosity() == 1
                               && ParseInputs::get_viscosity_order() == 5;
      return Point::Ones(fifthDerivative ? 3 : 2);
    }
    // The derivative-ratio detector consumes one more layer than Jameson.
    const bool extraDetectorGhost = ParseInputs::get_spatial_operator() == 2
                                && ParseInputs::get_Berta_detector_type() == 1;
    return Point::Ones(NGHOST + int(extraDetectorGhost));
  }
    
  // How many ghost cells does the operator need from the auxiliary variables
  inline static Point auxGhost() { return Point::Zeros();}
    
  // What is the intended order of accuracy of the operator
  inline static constexpr int order() { return 4; }

  // Initialization
  inline void initialize( State& a_W,
                        const State& a_BC,
                        const Scalar& a_radius,
                        const Vector& a_Xcart,
                        T a_gamma,
                        int a_nradius,
                        T a_h,
                        uint a_block) const;

  // New factorization of the operator starts here.
  // O2 stages may supply Cartesian primitives and a mask of faces needed by
  // the composite AMR update. Other callers evaluate every face.

  inline void operator()
  (
   State&                          a_Rhs,
   Array<State, DIM>&              a_fluxes,
   const State&                    a_USph,
   T                               a_dt,
   T                               a_scale = 1.0,
   int                             a_counter = -1,
   bool                            a_cartesianPrimitives = false,
   const SecondOrderFaceMask*       a_activeFaces = nullptr) const;

  inline void BertaOperator
  (
   State&                          a_Rhs,
   Array<State, DIM>&              a_fluxes,
   const State&                    a_USph,
   T                               a_dt,
   T                               a_scale = 1.0,
   int                             a_counter = -1) const;

  inline void sphPointToPrim(
                         State& a_WPoint,
                         const State& a_UPoint,
                         uint a_block,
                         const BoxData<T,DIM,MEM,DIM>* a_basis = nullptr) const;

  inline void sphAvgToPrim(
                         State& a_WAvg,
                         const State& a_UAvg,
                         uint a_block,
                         const BoxData<T,DIM,MEM,DIM>* a_basis = nullptr) const;

  inline void BertaComputeThetaJameson(
                         BoxData<T,1,MEM>& a_theta,
                         const State& a_WAvgForDetector,
                         T a_eta_d) const;

  inline void BertaComputeThetaDerivativeRatio(
                         BoxData<T,1,MEM>& a_theta,
                         const State& a_WAvgForDetector,
                         T a_eta_d) const;

  inline void BertaAverageToPoint(
                         State& a_UPoint,
                         const State& a_UAvg,
                         const BoxData<T,1,MEM>& a_theta) const;

  inline void BertaReconstructWithFallback(
                         State& a_WFace_L,
                         State& a_WFace_R,
                         BoxData<T,1,MEM>& a_thetaFluxFace,
                         BoxData<T,1,MEM>& a_thetaReconFace,
                         const State& a_WPoint,
                         const BoxData<T,1,MEM>& a_thetaCell,
                         uint a_dir,
                         int a_fallback_type) const;

  inline void BertaPrimPointToFluxAverage(
                         State& a_physFlux,
                         const State& a_WFacePoint,
                         const BoxData<T,1,MEM>& a_thetaFluxFace,
                         const BoxData<T,1,MEM>& a_adjDdir,
                         uint a_block,
                         uint a_dir,
                         const FaceEnergies* a_energy = nullptr,
                         const BoxData<T,1,MEM>* a_radialPoint = nullptr) const;

  inline void BertaPrimPointToMappedFlux(
                         State& a_fluxdir,
                         const State& a_physFluxAvg,
                         const State& a_WFacePoint,
                         const BoxData<T,1,MEM>& a_thetaFluxFace,
                         const BoxData<T,1,MEM>& a_adjDrDir,
                         const BoxData<T,DIM,MEM,DIM>& a_deta_a,
                         const BoxData<T,1,MEM>& a_deta,
                         const BoxData<T,DIM,MEM>& a_ntrow,
                         uint a_block,
                         uint a_dir,
                         const std::array<BoxData<T,DIM,MEM,DIM>,2>* a_metricDifferences = nullptr,
                         const BoxData<T,1,MEM>* a_magneticPoint = nullptr,
                         const BoxData<T,1,MEM>* a_totalPressurePoint = nullptr,
                         bool a_separableMetrics = false) const;

  inline void BertaPrimPointToMappedRusanovFlux(
                         State& a_fluxdir,
                         const State& a_WFacePoint_L,
                         const State& a_WFacePoint_R,
                         const BoxData<T,1,MEM>& a_thetaFluxFace,
                         const BoxData<T,1,MEM>& a_adjDdir,
                         const BoxData<T,DIM,MEM,DIM>& a_deta_a,
                         const BoxData<T,1,MEM>& a_deta,
                         const BoxData<T,DIM,MEM,DIM>& a_A,
                         const BoxData<T,DIM,MEM>& a_ntrow,
                         uint a_block,
                         uint a_dir,
                         const FaceEnergies* a_leftEnergy = nullptr,
                         const FaceEnergies* a_rightEnergy = nullptr,
                         const BoxData<T,1,MEM>* a_radialPoint = nullptr) const;

  // Optional point data must be prepared from the final face states in this stage.
  inline void primToFlux(
                         State& a_physFlux,
                         const State& a_WFace,
                         const State& a_WFaceBar,
                         const BoxData<T,1,MEM>& a_adjDdir,
                         uint a_block,
                         uint a_centering,
                         const State* a_point = nullptr,
                         const BoxData<T,1,MEM>* a_radialPoint = nullptr,
                         const FaceEnergies* a_pointEnergy = nullptr,
                         const State* a_fluxBar = nullptr) const;
  // Implemented.
  inline void sphToPrim(
                         State& a_W,
                         State& a_WBar,
                         const State& a_USph,
                         uint a_block,
                         bool a_restoreTestBackground = true,
                         BoxData<T,1,MEM>* a_magneticBar = nullptr) const;

  // Implemented.
  inline void sphToPrim_2O(
                         State& a_W,
                         const State& a_USph,
                         uint a_block,
                         bool a_restoreTestBackground = true) const;

  inline void secondOrderOperator(State& rhs, Array<State,DIM>& fluxes,
      const State& state, T scale, bool cartesianPrimitives = false,
      const SecondOrderFaceMask* activeFaces = nullptr) const;
  inline void reconstructPLM(State& left, State& right,
      const State& primitive, const Box& faces, int dir) const;
  inline static void prepareSharedPanelStates(
      const MBLevelBoxData<T,NUMCOMPS,MEM>& primitive);

  /// Restore the analytic Cartesian background used by magnetized tests.
  inline void addTestProblemBackgroundB(
                         State& a_W,
                         uint a_block) const;

  /// Compute (|B0 + deltaB|^2 - |B0|^2)/(8*pi) without subtracting
  /// two low-beta magnetic energies of nearly equal magnitude.
  inline void testProblemDeltaMagneticEnergy(
                         BoxData<T,1,MEM>& a_deltaMe,
                         const State& a_WDelta,
                         uint a_block) const;

  /// Compute conserved quantities only on valid cells, given point values
  /// of primitive variables. Typically used for initialization. 
  inline void primToCons(
                         State& a_JU,
                         const State& a_WPoint,
                         const BoxData<T>& a_dVolr,
                         T a_gamma,
                         T a_h,
                         uint a_block) const;

  inline void dtInv(Reduction<T,Operation::Max,MEM>& a_dtInv,const State& a_W) const;

  inline static bool validConservedState(
                         const MBLevelBoxData<T,NUMCOMPS,MEM>& a_JU);

   inline static T dtCFL(
                         MBLevelBoxData<T,NUMCOMPS,MEM>& a_JU,
                         MBInterpOp& a_iop,
                         MBLevelBoxData<T,1,MEM>& a_dVolr);

  inline static T dtCFLSecondOrderHost(
      const MBLevelBoxData<T,NUMCOMPS,MEM>& state,
      const MBLevelBoxData<T,1,MEM>& volume,
      const MBLevelBoxData<T,1,MEM>* mappedScale=nullptr);
  
  inline BoxData<T,1,MEM> kineticEnergyW(
                                         const State& a_WPoint,
                                         uint a_block,
                                         uint a_centering,
                                         const BoxData<T,DIM,MEM,DIM>* a_basis = nullptr) const;

  inline BoxData<T,1,MEM> magneticEnergyW(
                                         const State& a_WPoint,
                                         uint a_block,
                                         uint a_centering,
                                         const BoxData<T,DIM,MEM,DIM>* a_basis = nullptr) const;

  inline BoxData<T,1,MEM> vdotbW(
                                const State& a_WPoint,
                                uint a_block,
                                uint a_centering) const;    
                                
  inline void EnergyW(
                                BoxData<T,1,MEM>& a_ke,
                                BoxData<T,1,MEM>& a_me,
                                BoxData<T,1,MEM>& a_vb,
                                const State& a_WPoint,
                                uint a_block,
                                uint a_centering,
                                const BoxData<T,DIM,MEM,DIM>* a_basis = nullptr) const;
                                
  inline void EnergyW(
                                BoxData<T,1,MEM>& a_ke,
                                BoxData<T,1,MEM>& a_me,
                                const State& a_WPoint,
                                uint a_block,
                                uint a_centering) const;

  inline void radialMetrics(
                            BoxData<T, 1, MEM>& a_radius,
                            BoxData<T,DIM,MEM>& a_Diagr,
                            BoxData<T,DIM,MEM>& a_adjDr,
                            BoxData<T, 1, MEM>& a_dVolr,
                            Box a_bx) const;

  inline void primToMappedFlux(
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
                              const State* a_point = nullptr,
                              const std::array<BoxData<T,DIM,MEM,DIM>,2>* a_metricDifferences = nullptr,
                              const BoxData<T,1,MEM>* a_magneticPoint = nullptr,
                              const BoxData<T,1,MEM>* a_magneticBar = nullptr,
                              const BoxData<T,1,MEM>* a_pressureBar = nullptr,
                              const BoxData<T,1,MEM>* a_radialPressureBar = nullptr) const;

  inline void primToMappedRusanovFlux(
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
                              const State* a_leftPoint = nullptr,
                              const State* a_rightPoint = nullptr,
                              const BoxData<T,1,MEM>* a_radialPoint = nullptr,
                              const FaceEnergies* a_leftEnergy = nullptr,
                              const FaceEnergies* a_rightEnergy = nullptr) const;

  /// Repair invalid reconstructed states before either part of the Roe flux.
  inline void repairRoeFaceStates(
                              State& a_WFace_L,
                              State& a_WFace_R,
                              const State& a_WFaceBar,
                              const State& a_WBar,
                              uint a_dir) const;

  /// Mapped Roe dissipation; optional point states and shock flags support Berta.
  inline void primToMappedRoeFlux(
                              State& a_fluxdir,
                              const State& a_WFace_L,
                              const State& a_WFace_R,
                              const State& a_WFaceBar,
                              const BoxData<T,1,MEM>& a_adjDdir,
                              const BoxData<T,DIM,MEM,DIM>& a_detA_A,
                              const BoxData<T,1,MEM>& a_detA,
                              const BoxData<T,DIM,MEM,DIM>& a_A,
                              uint a_dir,
                              const State* a_leftPoint = nullptr,
                              const State* a_rightPoint = nullptr,
                              const BoxData<T,1,MEM>* a_radialPoint = nullptr,
                              const std::array<BoxData<T,DIM,MEM,DIM>,2>* a_metricDifferences = nullptr,
                              const BoxData<T,1,MEM>* a_thetaFluxFace = nullptr) const;

  // End new factorization of operator().
  // Viscosity starts here.

  inline State viscosityFaceState(const State& a_W, uint a_dir,
                                 const Box& a_faceBox = Box()) const;

  inline void
  viscFlux(
           State& a_Rhs,
           Array<BoxData<double, NUMCOMPS>, DIM>& a_fluxes,
           const State& a_USph,
           double a_dt) const;

  inline void
  LinearViscCoef(
                 BoxData<T,1,MEM>& a_LNViscCoef,
                 const State& a_WFace,
                 const BoxData<T,1,MEM>& a_deta,
                 const BoxData<T,DIM,MEM,DIM>& a_A,
                 const BoxData<T,DIM,MEM>& a_ntrow,
                 uint a_block,
                 uint a_dir) const;
  inline void
  NonLinearViscCoef(
                    BoxData<T,1,MEM>& a_NLViscCoef,
                    const State& a_W,
                    const BoxData<T,DIM,MEM>& a_Dr,
                    const BoxData<T,DIM,MEM>& a_adjDr,
                    const BoxData<T,DIM,MEM,DIM>& a_A,
                    const BoxData<T,DIM,MEM>& a_ntrow,
                    const BoxData<T,1,MEM>& a_dVolr,
                    const BoxData<T,1,MEM>& a_deta,
                    Array<T,DIM> a_dxi,
                    uint a_block,
                    uint a_dir) const;

  inline void
  primToMappedViscFlux(
                       State& a_fluxdir,
                       const State& a_physFlux,
                       const State& a_WFace,
                       const BoxData<T,1,MEM>& a_adjDdir,
                       const BoxData<T,DIM,MEM,DIM>& a_deta_a,
                       const BoxData<T,1,MEM>& a_deta,
                       const BoxData<T,DIM,MEM,DIM>& a_A,
                       const BoxData<T,DIM,MEM>& a_ntrow,
                       uint a_block,
                       uint a_dir) const;

  // End Viscosity.

  // DivB cleaning starts here.
  // Restore fourth-order magnetic averages for tests that evolve delta-B.
  inline void BertaPowellMagneticAverages(BoxData<T,DIM,MEM>& a_BAvg,
                                        const State& a_USph,
                                        uint a_block) const;

  // Accumulate -J*div(B), where J is the cell mapping Jacobian.
  // Both operators supply fourth-order magnetic face averages. Berta can
  // retain its local point flux on troubled faces using the optional switch.
  inline void DivBPowell1(BoxData<T,1,MEM>& a_minusJDivB,
                        const Scalar& a_Bnormal,
                        const BoxData<T,1,MEM>& a_deta,
                        const BoxData<T,1,MEM>& a_adjDrDir,
                        int a_dir,
                        const Scalar* a_BnormalPoint = nullptr,
                        const Scalar* a_thetaFluxFace = nullptr) const;
  
  // Fourth-order cell average of the Powell source. The divergence average
  // must include one ghost layer; A_matrix0 contains pointwise basis matrices.
  inline void DivBPowell2(State& a_Rhs,
                        const BoxData<T,1,MEM>& a_minusJDivB,
                        const State& a_W,
                        const BoxData<T,DIM,MEM,DIM>& a_A_matrix0,
                        bool a_cellAverages = true) const;

  // Compute physical div(B) from the same discrete operator used by Powell cleaning.
  // O2 AMR can supply its already prepared Cartesian primitives.
  inline void computeDivB(BoxData<T,1,MEM>& a_divB,
                          const State& a_USph,
                          bool a_cartesianPrimitives = false) const;

  // End DivB cleaning.

  // Gravity starts here.
  // Inputs are cubed-spherical conserved cell averages and averaged metrics.
  // The radial r^2 Jacobian factor cancels the inverse-square gravity.
  inline void Gravity(State& a_Rhs,
                      const State& a_USph,
                      const BoxData<T,1,MEM>& a_drDxi,
                      const BoxData<T,DIM,MEM,DIM>& a_detA_A,
                      const BoxData<T,1,MEM>& a_detA) const;

  inline void Gravity_Euler(State& a_Rhs,
                            const State& a_USph) const;

  inline void gravitySecondOrderPrimitives(State& rhs, const State& w) const;

  // End Gravity.

  // Begin Limiter.
  inline void Limiters_4O(BoxData<double,NUMCOMPS>& a_W_ave_L_lim_flat,
                              BoxData<double,NUMCOMPS>& a_W_ave_R_lim_flat,
                              BoxData<double,NUMCOMPS>& a_W_ave_L,
                              BoxData<double,NUMCOMPS>& a_W_ave_R,
                              BoxData<double,NUMCOMPS>& a_W_ave,
                              BoxData<double,NUMCOMPS>& a_W_bar,
                              BoxData<double,1>& a_me,
                              const int a_d,
                              const double a_dx,
                              const double a_dy,
                              const double a_dz) const;

  inline void Limiters_minmod(BoxData<double,NUMCOMPS>& a_W_low_lim,
                  BoxData<double,NUMCOMPS>& a_W_high_lim,
                  BoxData<double,NUMCOMPS>& a_W,
                  BoxData<double,DIM>& a_dx_sph,
                  const int a_d) const;                   

  // End Limiter.

// Begin CME
  static inline void
  Insert_CME(MBLevelBoxData<T, NUMCOMPS, MEM> &a_JU,
    MBLevelBoxData<T, 1, MEM> &a_dVolrLev,
    MBInterpOp &a_iop,
    MBDisjointBoxLayout &a_layout,
    double a_time,
    double a_dt,
    double a_gamma,
    const CMEParameters& a_cme);
// End CME

// Begin Floor P
static inline void
P_floor(MBLevelBoxData<T, NUMCOMPS, MEM> &a_JU,
  MBLevelBoxData<T, 1, MEM> &a_dVolrLev,
  MBInterpOp &a_iop,
  MBDisjointBoxLayout &a_layout,
  double a_time,
  double a_dt,
  double a_gamma);
// End Floor P

  /// Local boundary conditions here. 
  inline void
  PreStagePatch(BoxData<T,NUMCOMPS,MEM>& a_UStage,
                BoxData<T,NUMCOMPS,MEM>& a_BC,
                BoxData<T,1,MEM>& a_dVolr,
                Box a_blockBox,
                T a_time);
  
  /// Exchange, block boundary conditions, transform from JU to USph. 
  static inline void
  prepareState(MBLevelBoxData<T,NUMCOMPS,MEM>& a_state,
               MBInterpOp& a_iop,
               MBLevelBoxData<T,1,MEM>& a_dVolr,
               T a_time);

  // Unlimited O2 runs with artificial viscosity retain the native conserved
  // path because the viscosity stencil consumes those conserved components.
  static bool usesSecondOrderPrimitives()
  {
    return ParseInputs::get_spatial_order() == 2
        && ParseInputs::get_linear_viscosity() == 0
        && ParseInputs::get_nonlinear_viscosity() == 0;
  }

  static inline void prepareSecondOrderPrimitives(
      MBLevelBoxData<T,NUMCOMPS,MEM>& state, MBInterpOp& interp,
      MBLevelBoxData<T,1,MEM>& volume, T time);

  static inline void fillRadialPrimitiveBoundaries(
      MBLevelBoxData<T,NUMCOMPS,MEM>& state,
      MBLevelBoxData<T,1,MEM>& volume, T time);

  /// Fill physical radial ghosts of an already converted native state.
  static inline void
  fillRadialBoundaries(MBLevelBoxData<T,NUMCOMPS,MEM>& a_state,
                       MBLevelBoxData<T,1,MEM>& a_dVolr,
                       T a_time);

  /// Requires radial BCs and one extra native-state halo for pressure recovery.
  static inline void
  correctSeamPressure(MBLevelBoxData<T,NUMCOMPS,MEM>& a_state,
                       MBInterpOp& a_iop);

  /// Prepare stage primitives/native state and accumulate viscosity/gravity.
  // Implemented.
  static inline void
  preStageLevel(MBLevelBoxData<T,NUMCOMPS,MEM>& a_UStage,
                MBLevelBoxData<T,NUMCOMPS,MEM>& a_rhsTotal,               
                MBLevelBoxData<T,1,MEM>& a_dVolrLev,
                MBInterpOp& a_iop,
                MBBoundaryRegister<T,NUMREG,MEM>& a_blockreg,
                Array<Reduction<T,Operation::Sum,MEM>,NUMCOMPS>& a_consSum,
                T a_time,
                T a_dt,
                T a_weight,
                int a_kstage,
                bool flag_viscosity);

  /// Compute fluxes, and RHS.
  static inline void
  StageLevel(MBLevelBoxData<T,NUMCOMPS,MEM>& a_rhs,
             MBLevelBoxData<T,NUMCOMPS,MEM>& a_UStage,
             MBBoundaryRegister<T,NUMREG,MEM>& a_blockreg,
             Array<Reduction<T,Operation::Sum,MEM>,NUMCOMPS>& a_consSum,
             T a_time,
             T a_dt,
             T a_weight,
             int a_kstage);

private:
  int m_count = 0;
  // The owning patch operator is recreated with the level geometry. Store
  // only one angular plane and one radial line for the O2 gravity source.
  struct SecondOrderGravityGeometry
  {
    Box box;
    std::vector<std::array<T,7>> angular;
    std::vector<T> radial;
  };
  mutable std::shared_ptr<SecondOrderGravityGeometry> m_gravityGeometry;
  struct SecondOrderFaceGeometry
  {
    Box faces;
    std::vector<HelioCubed::SecondOrderFrame> angular;
    std::vector<T> radial;
  };
  mutable std::array<std::shared_ptr<SecondOrderFaceGeometry>,DIM> m_faceGeometry;
  // Set only while StageLevel evaluates this patch. Individual AMR patch
  // operators and the analytic test equilibrium use their existing paths.
  mutable const PanelStates* m_panelStates = nullptr;
  struct SecondOrderBoundaryGeometry
  {
    Box box;
    std::vector<T> basis,inverse;
  };
  mutable std::array<std::shared_ptr<SecondOrderBoundaryGeometry>,2> m_boundaryGeometry;
  // The exact magnetic-test equilibrium and geometry are time-independent.
  // Cache its discrete operator so well balancing adds only one operator
  // evaluation per patch per run, rather than one at every RK stage.
  mutable bool m_testBackgroundReady = false;
  mutable T m_testBackgroundScale = 0.0;
  mutable State m_testBackgroundRhs;
  mutable Array<State,DIM> m_testBackgroundFluxes;
  
};
#include "BoxOp_Utilities.H"
#include "BoxOp_Viscocity.H"
#include "BoxOp_DivBcleaning.H"
#include "BoxOp_Gravity.H"
#include "BoxOp_Limiters.H"
#include "BoxOp_TestProblems.H"
#include "BoxOp_PreStage.H"
#include "BoxOp_SeamPressure.H"
#include "BoxOp_SecondOrderPreStage.H"
#include "BoxOp_RoeCubedSphere.H"
#include "BoxOp_SecondOrder.H"
#include "BoxOp_SecondOrderCFL.H"
#include "BoxOp_EulerCubedSphereImplem.H"
#include "BoxOp_BertaCubedSphereImplem.H"
#include "BoxOp_Initialize.H"

#ifdef PR_AMR
/* TODO: this doesn't belong here and it doesn't compile
static inline void generateTags(
                                TagData& a_tags,
                                BoxData<T, NUMCOMPS>& a_state)
{
  forallInPlace(f_threshold, a_tags, a_state);
}
*/
#endif


#endif //end include guard
