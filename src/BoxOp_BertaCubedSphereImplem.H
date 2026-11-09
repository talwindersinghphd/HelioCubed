#include <limits>

// Separate component lines expose independent output storage to CPU compilers.
// Keep each cell's correction and matrix-column accumulation order.
template<unsigned N, bool BOTH, typename T>
inline void bertaMappedLine(
    T* __restrict__ a_out,
    const T* const* a_matrix,
    const T* const* a_firstMatrix,
    const T* const* a_secondMatrix,
    const T* const* a_flux,
    const T* const* a_firstFlux,
    const T* const* a_secondFlux,
    int a_size)
{
#ifdef __clang__
#pragma clang loop vectorize(enable)
#endif
  for (int cell = 0; cell < a_size; ++cell)
  {
    T sum = T(0);
    for (unsigned column = 0; column < N; ++column)
    {
      T term = T(0);
      if constexpr (BOTH)
        term += a_firstMatrix[column][cell]*a_firstFlux[column][cell]*T(1.0/12.0);
      term += a_secondMatrix[column][cell]*a_secondFlux[column][cell]*T(1.0/12.0);
      term += a_matrix[column][cell]*a_flux[column][cell]*T(1);
      sum += term;
    }
    a_out[cell] = sum;
  }
}

template<typename T>
inline void bertaMapHostBox(
    BoxData<T,NUMCOMPS,HOST>& a_out,
    const BoxData<T,NUMCOMPS,HOST>& a_flux,
    const BoxData<T,NUMCOMPS,HOST>& a_firstFlux,
    const BoxData<T,NUMCOMPS,HOST>& a_secondFlux,
    const BoxData<T,DIM,HOST,DIM>& a_matrix,
    const BoxData<T,DIM,HOST,DIM>& a_firstMatrix,
    const BoxData<T,DIM,HOST,DIM>& a_secondMatrix,
    const BoxData<T,1,HOST>& a_det,
    const BoxData<T,1,HOST>& a_firstDet,
    const BoxData<T,1,HOST>& a_secondDet,
    const Box& a_box,bool a_skipFirst)
{
  PROTO_ASSERT(a_out.aliasData() != a_flux.aliasData(),
      "Berta mapping requires separate input and output storage");
  const Point low = a_box.low(),high = a_box.high();
  const int count = a_box.size(0);
  for (int k = low[2]; k <= high[2]; ++k)
    for (int j = low[1]; j <= high[1]; ++j)
    {
      const Point start(low[0],j,k);
      for (int component : {iRHO,iE})
      {
        const T* matrix[1] = {a_det.data(start)};
        const T* firstMatrix[1] = {a_firstDet.data(start)};
        const T* secondMatrix[1] = {a_secondDet.data(start)};
        const T* flux[1] = {a_flux.data(start,component)};
        const T* firstFlux[1] = {a_firstFlux.data(start,component)};
        const T* secondFlux[1] = {a_secondFlux.data(start,component)};
        if (a_skipFirst)
          bertaMappedLine<1,false>(a_out.data(start,component),matrix,
              firstMatrix,secondMatrix,flux,firstFlux,secondFlux,count);
        else
          bertaMappedLine<1,true>(a_out.data(start,component),matrix,
              firstMatrix,secondMatrix,flux,firstFlux,secondFlux,count);
      }
      for (int component : {iMOMX,iBX})
        for (unsigned row = 0; row < DIM; ++row)
        {
          const T* matrix[DIM];
          const T* firstMatrix[DIM];
          const T* secondMatrix[DIM];
          const T* flux[DIM];
          const T* firstFlux[DIM];
          const T* secondFlux[DIM];
          for (unsigned column = 0; column < DIM; ++column)
          {
            matrix[column] = a_matrix.data(start,row,column);
            firstMatrix[column] = a_firstMatrix.data(start,row,column);
            secondMatrix[column] = a_secondMatrix.data(start,row,column);
            flux[column] = a_flux.data(start,component+column);
            firstFlux[column] = a_firstFlux.data(start,component+column);
            secondFlux[column] = a_secondFlux.data(start,component+column);
          }
          if (a_skipFirst)
            bertaMappedLine<DIM,false>(a_out.data(start,component+row),matrix,
                firstMatrix,secondMatrix,flux,firstFlux,secondFlux,count);
          else
            bertaMappedLine<DIM,true>(a_out.data(start,component+row),matrix,
                firstMatrix,secondMatrix,flux,firstFlux,secondFlux,count);
        }
    }
}

// A shifted component view shares storage; const callers never modify it.
template<typename T, unsigned C, MemType MEM>
inline const BoxData<T,C,MEM> bertaReadView(
    const BoxData<T,C,MEM>& a_data,
    const Point& a_shift = Point::Zeros())
{
  auto view = slice<T,C,C,MEM>(a_data,0);
  view.shift(a_shift);
  return view;
}

// Share both transverse derivatives and accumulate every component in one pass.
// The scalar product and correction order match Operator::_faceTensorProduct.
template<typename T, unsigned C, MemType MEM>
inline BoxData<T,C,MEM> bertaFaceTensorProduct(
    const BoxData<T,C,MEM>& a_vector,
    const BoxData<T,1,MEM>& a_scalar,
    int a_dir,
    int a_correctionDir = -2)
{
  PR_TIMERS("Berta::faceTensorProduct()");
  if constexpr (MEM == HOST && DIM == 3)
  {
    Box productBox = (a_vector.box() & a_scalar.box()).grow(-1).grow(a_dir,1);
    // -2 keeps both corrections; -1 means a factor is constant in both
    // transverse directions; a coordinate index selects the sole nonzero one.
    if (a_correctionDir == -1)
      return forallOverwrite<T,C,MEM>([] PROTO_LAMBDA(
          Var<T,C,MEM>& out, const Var<T,C,MEM>& vector,
          const Var<T,1,MEM>& scalar)
      {
        for (int comp = 0; comp < C; ++comp)
        {
          T value = T(0);
          value += vector(comp)*scalar(0)*T(1);
          out(comp) = value;
        }
      },productBox,a_vector,a_scalar);
    if (a_correctionDir >= 0)
    {
      const Stencil<T> difference =
          0.5*Shift::Basis(a_correctionDir,1)
        - 0.5*Shift::Basis(a_correctionDir,-1);
      BoxData<T,C,MEM> vectorDifference = difference(a_vector);
      BoxData<T,1,MEM> scalarDifference = difference(a_scalar);
      return forallOverwrite<T,C,MEM>([] PROTO_LAMBDA(
          Var<T,C,MEM>& out, const Var<T,C,MEM>& vector,
          const Var<T,C,MEM>& dv, const Var<T,1,MEM>& scalar,
          const Var<T,1,MEM>& ds)
      {
        for (int comp = 0; comp < C; ++comp)
        {
          T value = T(0);
          value += dv(comp)*ds(0)*T(1.0/12.0);
          value += vector(comp)*scalar(0)*T(1);
          out(comp) = value;
        }
      },productBox,a_vector,vectorDifference,a_scalar,scalarDifference);
    }
    const int firstDir = a_dir == 0 ? 1 : 0;
    const int secondDir = a_dir == 2 ? 1 : 2;
    const Stencil<T> firstDiff =
        0.5*Shift::Basis(firstDir,1) - 0.5*Shift::Basis(firstDir,-1);
    const Stencil<T> secondDiff =
        0.5*Shift::Basis(secondDir,1) - 0.5*Shift::Basis(secondDir,-1);
    BoxData<T,C,MEM> vectorFirst = firstDiff(a_vector);
    BoxData<T,C,MEM> vectorSecond = secondDiff(a_vector);
    BoxData<T,1,MEM> scalarFirst = firstDiff(a_scalar);
    BoxData<T,1,MEM> scalarSecond = secondDiff(a_scalar);
    BoxData<T,C,MEM> product;
    product.defineForOverwrite(productBox);
    forallInPlace([] PROTO_LAMBDA(
        Var<T,C,MEM>& out, const Var<T,C,MEM>& vector,
        const Var<T,C,MEM>& dv0, const Var<T,C,MEM>& dv1,
        const Var<T,1,MEM>& scalar,
        const Var<T,1,MEM>& ds0, const Var<T,1,MEM>& ds1)
    {
      for (int comp = 0; comp < C; ++comp)
      {
        T value = T(0);
        value += dv0(comp)*ds0(0)*T(1.0/12.0);
        value += dv1(comp)*ds1(0)*T(1.0/12.0);
        value += vector(comp)*scalar(0)*T(1);
        out(comp) = value;
      }
    },product, a_vector,vectorFirst,vectorSecond,
        a_scalar,scalarFirst,scalarSecond);
    return product;
  }
  else
    return Operator::_faceTensorProduct(
        a_vector,a_scalar,a_vector,a_scalar,a_dir);
}

// ====================================================================================
// Berta-style 4th-order FV pointwise-WENOZ implementation.
// Corrected for face indexing convention:
//   face index i = lower face of cell i = x_{i-1/2}
// so the face lies between cells i-1 and i.
//
// Fixes included here:
//   1. thetaCell is defined on the full cell box and defaults to 0 at boundaries.
//   2. thetaReconFace and thetaFluxFace are separated:
//        thetaReconFace: full WENOZ stencil flag, used to choose WENOZ vs fallback.
//        thetaFluxFace : adjacent-cell flag, used for flux point-to-average correction.
//   3. Default fallback is WENO3, not PLM.
// ====================================================================================

// ====================================================================================
// 1. Primitive conversion helpers.
// ====================================================================================

template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::sphPointToPrim(
                                          State& a_WPoint,
                                          const State& a_UPoint,
                                          uint a_block,
                                          const BoxData<T,DIM,MEM,DIM>* a_basis) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::sphPointToPrim()");

  // Pure pointwise conserved -> pointwise primitive conversion.
  // No deconvolution here.
  // No primitive-average correction here.
  a_WPoint = forall<T,NUMCOMPS,MEM>(f_sphToPrim,a_UPoint);
  const bool magneticTest =
      isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
  BoxData<T,1,MEM> deltaMe(a_WPoint.box());
  deltaMe.setVal(0.0);
  if (magneticTest)
  {
    testProblemDeltaMagneticEnergy(deltaMe,a_WPoint,a_block);
    addTestProblemBackgroundB(a_WPoint,a_block);
  }

  BoxData<T,DIM,MEM,DIM> ownedBasis;
  if (!a_basis)
  {
    ownedBasis = pointwiseSphericalBasis(a_WPoint.box(),a_block,-1);
    a_basis = &ownedBasis;
  }
  BoxData<T,1,MEM> kePoint = kineticEnergyW(a_WPoint,a_block,-1,a_basis);
  BoxData<T,1,MEM> mePoint = magneticEnergyW(a_WPoint,a_block,-1,a_basis);

  forallInPlace(
      [ ] PROTO_LAMBDA
      (
        Var<T,NUMCOMPS>& a_w,
        const Var<T,1,MEM>& a_ke,
        const Var<T,1,MEM>& a_me,
        const Var<T,1,MEM>& a_deltaMe,
        bool a_magneticTest,
        T a_gamma)
      {
        if (a_magneticTest)
        {
          a_w(iP) = c_TEST_P0 + (a_gamma - 1.0)*
              (a_w(iP) - a_ke(0) - a_deltaMe(0));
        }
        else
        {
          a_w(iP) = (a_w(iP) - a_ke(0) - a_me(0))*(a_gamma - 1.0);
        }
        if (a_w(iP) < 0.0) a_w(iP) = 0.0;
      },
      a_WPoint,
      kePoint,
      mePoint,
      deltaMe,
      magneticTest,
      m_gamma);
}


template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::sphAvgToPrim(
                                          State& a_WAvg,
                                          const State& a_UAvg,
                                          uint a_block,
                                          const BoxData<T,DIM,MEM,DIM>* a_basis) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::sphAvgToPrim()");

  // Direct V(<U>) conversion.
  // Used only for Berta detector, not for reconstruction.
  a_WAvg = forall<T,NUMCOMPS,MEM>(f_sphToPrim,a_UAvg);
  const bool magneticTest =
      isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
  BoxData<T,1,MEM> deltaMe(a_WAvg.box());
  deltaMe.setVal(0.0);
  if (magneticTest)
  {
    testProblemDeltaMagneticEnergy(deltaMe,a_WAvg,a_block);
    addTestProblemBackgroundB(a_WAvg,a_block);
  }

  BoxData<T,DIM,MEM,DIM> ownedBasis;
  if (!a_basis)
  {
    ownedBasis = pointwiseSphericalBasis(a_WAvg.box(),a_block,-1);
    a_basis = &ownedBasis;
  }
  BoxData<T,1,MEM> keAvg = kineticEnergyW(a_WAvg,a_block,-1,a_basis);
  BoxData<T,1,MEM> meAvg = magneticEnergyW(a_WAvg,a_block,-1,a_basis);

  forallInPlace(
      [ ] PROTO_LAMBDA
      (
        Var<T,NUMCOMPS>& a_w,
        const Var<T,1,MEM>& a_ke,
        const Var<T,1,MEM>& a_me,
        const Var<T,1,MEM>& a_deltaMe,
        bool a_magneticTest,
        T a_gamma)
      {
        if (a_magneticTest)
        {
          a_w(iP) = c_TEST_P0 + (a_gamma - 1.0)*
              (a_w(iP) - a_ke(0) - a_deltaMe(0));
        }
        else
        {
          a_w(iP) = (a_w(iP) - a_ke(0) - a_me(0))*(a_gamma - 1.0);
        }
        if (a_w(iP) < 0.0) a_w(iP) = 0.0;
      },
      a_WAvg,
      keAvg,
      meAvg,
      deltaMe,
      magneticTest,
      m_gamma);
}


// ====================================================================================
// 2. Berta Jameson-type discontinuity detector.
// ====================================================================================

template<typename T, MemType MEM>
PROTO_KERNEL_START
void f_berta_jameson_theta_rawF(
    Var<T,1,MEM>&              a_theta,
    const Var<T,NUMCOMPS,MEM>& a_Wm_x,
    const Var<T,NUMCOMPS,MEM>& a_W0,
    const Var<T,NUMCOMPS,MEM>& a_Wp_x,
    const Var<T,NUMCOMPS,MEM>& a_Wm_y,
    const Var<T,NUMCOMPS,MEM>& a_Wp_y,
    const Var<T,NUMCOMPS,MEM>& a_Wm_z,
    const Var<T,NUMCOMPS,MEM>& a_Wp_z,
    const T                    a_eta_d,
    const T                    a_eps)
{
  auto eta_one_dir = [&] PROTO_LAMBDA
    (
      const T qm,
      const T q0,
      const T qp)
    {
      T num = abs(qp - 2.0*q0 + qm);
      T den = abs(qp) + 2.0*abs(q0) + abs(qm) + a_eps;
      return num/den;
    };

  auto magp = [&] PROTO_LAMBDA
    (
      const Var<T,NUMCOMPS,MEM>& W)
    {
      T bx = W(iBX);
      T by = W(iBY);
      T bz = W(iBZ);
      return 0.5*(bx*bx + by*by + bz*bz);
    };

  T rho0 = a_W0(iRHO);
  T p0   = a_W0(iP);
  T mp0  = magp(a_W0);

  T etax_rho = eta_one_dir(a_Wm_x(iRHO),rho0,a_Wp_x(iRHO));
  T etax_p   = eta_one_dir(a_Wm_x(iP),  p0,  a_Wp_x(iP));
  T etax_mp  = eta_one_dir(magp(a_Wm_x),mp0, magp(a_Wp_x));
  T etax     = max(etax_rho,max(etax_p,etax_mp));

  T etay_rho = eta_one_dir(a_Wm_y(iRHO),rho0,a_Wp_y(iRHO));
  T etay_p   = eta_one_dir(a_Wm_y(iP),  p0,  a_Wp_y(iP));
  T etay_mp  = eta_one_dir(magp(a_Wm_y),mp0, magp(a_Wp_y));
  T etay     = max(etay_rho,max(etay_p,etay_mp));

  T etaz_rho = eta_one_dir(a_Wm_z(iRHO),rho0,a_Wp_z(iRHO));
  T etaz_p   = eta_one_dir(a_Wm_z(iP),  p0,  a_Wp_z(iP));
  T etaz_mp  = eta_one_dir(magp(a_Wm_z),mp0, magp(a_Wp_z));
  T etaz     = max(etaz_rho,max(etaz_p,etaz_mp));

  T eta = sqrt(etax*etax + etay*etay + etaz*etaz);

  // theta = 1: smooth, use 4th-order correction.
  // theta = 0: troubled, reduce order.
  a_theta(0) = (eta < a_eta_d) ? 1.0 : 0.0;
}
PROTO_KERNEL_END(f_berta_jameson_theta_rawF, f_berta_jameson_theta_raw)


template<typename T, MemType MEM>
PROTO_KERNEL_START
void f_berta_dilate_thetaF(
    Var<T,1,MEM>&       a_theta,
    const Var<T,1,MEM>& a_t0,
    const Var<T,1,MEM>& a_tm_x,
    const Var<T,1,MEM>& a_tp_x,
    const Var<T,1,MEM>& a_tm_y,
    const Var<T,1,MEM>& a_tp_y,
    const Var<T,1,MEM>& a_tm_z,
    const Var<T,1,MEM>& a_tp_z)
{
  T tmin = a_t0(0);

  tmin = min(tmin,a_tm_x(0));
  tmin = min(tmin,a_tp_x(0));
  tmin = min(tmin,a_tm_y(0));
  tmin = min(tmin,a_tp_y(0));
  tmin = min(tmin,a_tm_z(0));
  tmin = min(tmin,a_tp_z(0));

  // If this cell or any first neighbor is troubled, theta=0.
  a_theta(0) = tmin;
}
PROTO_KERNEL_END(f_berta_dilate_thetaF, f_berta_dilate_theta)


template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::BertaComputeThetaJameson(
    BoxData<T,1,MEM>& a_theta,
    const State&      a_WAvgForDetector,
    T                 a_eta_d) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::BertaComputeThetaJameson()");

  Point ex = Point::Basis(0);
  Point ey = Point::Basis(1);
  Point ez = Point::Basis(2);

  // Raw detector needs first neighbors.
  Box bxRaw = a_WAvgForDetector.box().grow(-1);
  BoxData<T,1,MEM> thetaRaw(bxRaw);

  const T eps = 1.0e-40;

  const auto Wm_x = bertaReadView(a_WAvgForDetector,-(-ex));
  const auto W0 = bertaReadView(a_WAvgForDetector,-(Point::Zeros()));
  const auto Wp_x = bertaReadView(a_WAvgForDetector,-( ex));
  const auto Wm_y = bertaReadView(a_WAvgForDetector,-(-ey));
  const auto Wp_y = bertaReadView(a_WAvgForDetector,-( ey));
  const auto Wm_z = bertaReadView(a_WAvgForDetector,-(-ez));
  const auto Wp_z = bertaReadView(a_WAvgForDetector,-( ez));

  forallInPlace(
      f_berta_jameson_theta_raw,
      thetaRaw,
      Wm_x,
      W0,
      Wp_x,
      Wm_y,
      Wp_y,
      Wm_z,
      Wp_z,
      a_eta_d,
      eps);

  // Dilation needs first neighbors of thetaRaw.
  Box bxInterior = thetaRaw.box().grow(-1);
  BoxData<T,1,MEM> thetaInterior(bxInterior);

  const auto theta0 = bertaReadView(thetaRaw,-(Point::Zeros()));
  const auto thetam_x = bertaReadView(thetaRaw,-(-ex));
  const auto thetap_x = bertaReadView(thetaRaw,-( ex));
  const auto thetam_y = bertaReadView(thetaRaw,-(-ey));
  const auto thetap_y = bertaReadView(thetaRaw,-( ey));
  const auto thetam_z = bertaReadView(thetaRaw,-(-ez));
  const auto thetap_z = bertaReadView(thetaRaw,-( ez));

  forallInPlace(
      f_berta_dilate_theta,
      thetaInterior,
      theta0,
      thetam_x,
      thetap_x,
      thetam_y,
      thetap_y,
      thetam_z,
      thetap_z);

  // Full-box theta. Cells without enough detector stencil default to fallback.
  a_theta.define(a_WAvgForDetector.box());
  a_theta.setVal(0.0);
  thetaInterior.copyTo(a_theta);
}


// ====================================================================================
// 3. Berta derivative-ratio discontinuity detector.
// ====================================================================================

template<typename T, MemType MEM>
PROTO_KERNEL_START
void f_berta_derivative_ratio_theta_rawF(
    Var<T,1,MEM>&              a_theta,
    const Var<T,NUMCOMPS,MEM>& a_Wmm_x,
    const Var<T,NUMCOMPS,MEM>& a_Wm_x,
    const Var<T,NUMCOMPS,MEM>& a_W0,
    const Var<T,NUMCOMPS,MEM>& a_Wp_x,
    const Var<T,NUMCOMPS,MEM>& a_Wpp_x,
    const Var<T,NUMCOMPS,MEM>& a_Wmm_y,
    const Var<T,NUMCOMPS,MEM>& a_Wm_y,
    const Var<T,NUMCOMPS,MEM>& a_Wp_y,
    const Var<T,NUMCOMPS,MEM>& a_Wpp_y,
    const Var<T,NUMCOMPS,MEM>& a_Wmm_z,
    const Var<T,NUMCOMPS,MEM>& a_Wm_z,
    const Var<T,NUMCOMPS,MEM>& a_Wp_z,
    const Var<T,NUMCOMPS,MEM>& a_Wpp_z,
    const T                    a_eta_d,
    const T                    a_gamma,
    const T                    a_eps)
{
  auto mag = [&] PROTO_LAMBDA
    (
      const Var<T,NUMCOMPS,MEM>& W)
    {
      T bx = W(iBX);
      T by = W(iBY);
      T bz = W(iBZ);
      return sqrt(bx*bx + by*by + bz*bz);
    };

  auto ref_value = [&] PROTO_LAMBDA
    (
      const int comp)
    {
      if (comp == iRHO || comp == iP)
      {
        return abs(a_W0(comp));
      }
      if (comp == iVX || comp == iVY || comp == iVZ)
      {
        T rho = max(abs(a_W0(iRHO)),a_eps);
        T p   = max(abs(a_W0(iP)),a_eps);
        return sqrt(abs(a_gamma)*p/rho);
      }
      return mag(a_W0);
    };

  auto eta_comp = [&] PROTO_LAMBDA
    (
      const T qmm,
      const T qm,
      const T q0,
      const T qp,
      const T qpp,
      const T qref)
    {
      T d1 = 0.5*(qp - qm);
      T d2 = qp - 2.0*q0 + qm;
      T d3 = 0.5*(qpp - 2.0*qp + 2.0*qm - qmm);
      T d4 = qpp - 4.0*qp + 6.0*q0 - 4.0*qm + qmm;

      T eta_odd  = abs(d3)/(abs(qref) + abs(d1) + abs(d3) + a_eps);
      T eta_even = abs(d4)/(abs(qref) + abs(d2) + abs(d4) + a_eps);
      return max(eta_odd,eta_even);
    };

  auto eta_dir = [&] PROTO_LAMBDA
    (
      const Var<T,NUMCOMPS,MEM>& Wmm,
      const Var<T,NUMCOMPS,MEM>& Wm,
      const Var<T,NUMCOMPS,MEM>& Wp,
      const Var<T,NUMCOMPS,MEM>& Wpp)
    {
      T eta = 0.0;
      for (int comp = 0; comp < NUMCOMPS; comp++)
      {
        eta = max(eta,eta_comp(Wmm(comp),Wm(comp),a_W0(comp),Wp(comp),Wpp(comp),ref_value(comp)));
      }
      return eta;
    };

  T etax = eta_dir(a_Wmm_x,a_Wm_x,a_Wp_x,a_Wpp_x);
  T etay = eta_dir(a_Wmm_y,a_Wm_y,a_Wp_y,a_Wpp_y);
  T etaz = eta_dir(a_Wmm_z,a_Wm_z,a_Wp_z,a_Wpp_z);
  T eta  = sqrt(etax*etax + etay*etay + etaz*etaz);

  a_theta(0) = (eta < a_eta_d) ? 1.0 : 0.0;
}
PROTO_KERNEL_END(f_berta_derivative_ratio_theta_rawF, f_berta_derivative_ratio_theta_raw)


template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::BertaComputeThetaDerivativeRatio(
    BoxData<T,1,MEM>& a_theta,
    const State&      a_WAvgForDetector,
    T                 a_eta_d) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::BertaComputeThetaDerivativeRatio()");

  Point ex = Point::Basis(0);
  Point ey = Point::Basis(1);
  Point ez = Point::Basis(2);

  Box bxRaw = a_WAvgForDetector.box().grow(-2);
  BoxData<T,1,MEM> thetaRaw(bxRaw);

  const T eps = 1.0e-40;

  const auto Wmm_x = bertaReadView(a_WAvgForDetector,-(-2*ex));
  const auto Wm_x = bertaReadView(a_WAvgForDetector,-(-ex));
  const auto W0 = bertaReadView(a_WAvgForDetector,-(Point::Zeros()));
  const auto Wp_x = bertaReadView(a_WAvgForDetector,-( ex));
  const auto Wpp_x = bertaReadView(a_WAvgForDetector,-( 2*ex));

  const auto Wmm_y = bertaReadView(a_WAvgForDetector,-(-2*ey));
  const auto Wm_y = bertaReadView(a_WAvgForDetector,-(-ey));
  const auto Wp_y = bertaReadView(a_WAvgForDetector,-( ey));
  const auto Wpp_y = bertaReadView(a_WAvgForDetector,-( 2*ey));

  const auto Wmm_z = bertaReadView(a_WAvgForDetector,-(-2*ez));
  const auto Wm_z = bertaReadView(a_WAvgForDetector,-(-ez));
  const auto Wp_z = bertaReadView(a_WAvgForDetector,-( ez));
  const auto Wpp_z = bertaReadView(a_WAvgForDetector,-( 2*ez));

  forallInPlace(
      f_berta_derivative_ratio_theta_raw,
      thetaRaw,
      Wmm_x,
      Wm_x,
      W0,
      Wp_x,
      Wpp_x,
      Wmm_y,
      Wm_y,
      Wp_y,
      Wpp_y,
      Wmm_z,
      Wm_z,
      Wp_z,
      Wpp_z,
      a_eta_d,
      m_gamma,
      eps);

  // Flag the first neighbors too, as in the paper's troubled-cell propagation.
  Box bxInterior = thetaRaw.box().grow(-1);
  BoxData<T,1,MEM> thetaInterior(bxInterior);

  const auto theta0 = bertaReadView(thetaRaw,-(Point::Zeros()));
  const auto thetam_x = bertaReadView(thetaRaw,-(-ex));
  const auto thetap_x = bertaReadView(thetaRaw,-( ex));
  const auto thetam_y = bertaReadView(thetaRaw,-(-ey));
  const auto thetap_y = bertaReadView(thetaRaw,-( ey));
  const auto thetam_z = bertaReadView(thetaRaw,-(-ez));
  const auto thetap_z = bertaReadView(thetaRaw,-( ez));

  forallInPlace(
      f_berta_dilate_theta,
      thetaInterior,
      theta0,
      thetam_x,
      thetap_x,
      thetam_y,
      thetap_y,
      thetam_z,
      thetap_z);

  a_theta.define(a_WAvgForDetector.box());
  a_theta.setVal(0.0);
  thetaInterior.copyTo(a_theta);
}


// ====================================================================================
// 4. Theta-aware average-to-point recovery.
// ====================================================================================

template<typename T, MemType MEM>
PROTO_KERNEL_START
void f_berta_avg_to_point_thetaF(
    Var<T,NUMCOMPS,MEM>&       a_UPoint,
    const Var<T,NUMCOMPS,MEM>& a_UAvg,
    const Var<T,NUMCOMPS,MEM>& a_LapUAvg,
    const Var<T,1,MEM>&        a_theta)
{
  for (int comp = 0; comp < NUMCOMPS; comp++)
  {
    if (a_theta(0) > 0.5)
    {
      a_UPoint(comp) = a_UAvg(comp) - a_LapUAvg(comp)/24.0;
    }
    else
    {
      a_UPoint(comp) = a_UAvg(comp);
    }
  }
}
PROTO_KERNEL_END(f_berta_avg_to_point_thetaF, f_berta_avg_to_point_theta)


template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::BertaAverageToPoint(
    State&                  a_UPoint,
    const State&            a_UAvg,
    const BoxData<T,1,MEM>& a_theta) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::BertaAverageToPoint(theta)");

  State lapUAvg = Stencil<T>::Laplacian()(a_UAvg);

  a_UPoint.define(a_UAvg.box());

  forallInPlace(
      f_berta_avg_to_point_theta,
      a_UPoint,
      a_UAvg,
      lapUAvg,
      a_theta);
}


// Normalize each primitive component on the combined L/R face stencil before
// forming squared smoothness indicators. This makes epsilon dimensionless and
// prevents CGS density/pressure from losing their nonlinear WENO weights.
template<typename T, unsigned N>
inline ACCEL_DECORATION T bertaNormalizeWenoStencil(T (&a_values)[N])
{
  T scale = T(0);
  for (unsigned i = 0; i < N; ++i) scale = max(scale,abs(a_values[i]));
  if (scale == T(0))
  {
    // A max reduction can discard NaNs; do not turn an invalid stencil into zero.
    for (unsigned i = 0; i < N; ++i)
      if (a_values[i] != T(0)) return a_values[i];
    return scale;
  }
  if (scale > T(0))
  {
    if (scale >= std::numeric_limits<T>::min())
    {
      const T inverse = T(1)/scale;
      for (unsigned i = 0; i < N; ++i) a_values[i] *= inverse;
    }
    else
    {
      // Direct division also handles subnormal data without overflowing 1/scale.
      for (unsigned i = 0; i < N; ++i) a_values[i] /= scale;
    }
  }
  return scale;
}

// ====================================================================================
// 4. WENOZ, WENO3, and PLM face reconstruction.
// Corrected convention:
//   face index i = lower face of cell i = x_{i-1/2}
//   adjacent cells are i-1 and i.
// ====================================================================================

template<typename T, MemType MEM, unsigned C = NUMCOMPS>
PROTO_KERNEL_START
void f_wenoz_pointwise_faceF(
    Var<T,C,MEM>&       a_WL,
    Var<T,C,MEM>&       a_WR,
    const Var<T,C,MEM>& a_Wm2,
    const Var<T,C,MEM>& a_Wm1,
    const Var<T,C,MEM>& a_W0,
    const Var<T,C,MEM>& a_Wp1,
    const Var<T,C,MEM>& a_Wp2,
    const Var<T,C,MEM>& a_Wp3)
{
  const T eps = 1.0e-40;
  const T d0  = 1.0/16.0;
  const T d1  = 5.0/8.0;
  const T d2  = 5.0/16.0;

  for (int comp = 0; comp < C; comp++)
  {
    T values[6] = {a_Wm2(comp),a_Wm1(comp),a_W0(comp),
                   a_Wp1(comp),a_Wp2(comp),a_Wp3(comp)};
    const T scale = bertaNormalizeWenoStencil(values);
    if (scale == T(0))
    {
      a_WL(comp) = a_WR(comp) = T(0);
      continue;
    }
    // Left state at target face, using a_W0 as the left cell.
    T vm2 = values[0];
    T vm1 = values[1];
    T v0  = values[2];
    T vp1 = values[3];
    T vp2 = values[4];

    T q0 = ( 3.0*vm2 - 10.0*vm1 + 15.0*v0  ) / 8.0;
    T q1 = (    -vm1 +  6.0*v0  +  3.0*vp1 ) / 8.0;
    T q2 = ( 3.0*v0  +  6.0*vp1 -      vp2 ) / 8.0;

    T b0a = vm2 - 2.0*vm1 + v0;
    T b0b = vm2 - 4.0*vm1 + 3.0*v0;

    T b1a = vm1 - 2.0*v0 + vp1;
    T b1b = vm1 - vp1;

    T b2a = v0 - 2.0*vp1 + vp2;
    T b2b = 3.0*v0 - 4.0*vp1 + vp2;

    T beta0 = (13.0/12.0)*b0a*b0a + 0.25*b0b*b0b;
    T beta1 = (13.0/12.0)*b1a*b1a + 0.25*b1b*b1b;
    T beta2 = (13.0/12.0)*b2a*b2a + 0.25*b2b*b2b;

    T tau5 = abs(beta0 - beta2);

    T alpha0 = d0*(1.0 + tau5/(beta0 + eps));
    T alpha1 = d1*(1.0 + tau5/(beta1 + eps));
    T alpha2 = d2*(1.0 + tau5/(beta2 + eps));

    T asum = alpha0 + alpha1 + alpha2;

    a_WL(comp) = scale*((alpha0*q0 + alpha1*q1 + alpha2*q2)/asum);

    // Right state at target face: reversed stencil, using a_Wp1 as the right cell.
    T rm2 = values[5];
    T rm1 = values[4];
    T r0  = values[3];
    T rp1 = values[2];
    T rp2 = values[1];

    T rq0 = ( 3.0*rm2 - 10.0*rm1 + 15.0*r0  ) / 8.0;
    T rq1 = (    -rm1 +  6.0*r0  +  3.0*rp1 ) / 8.0;
    T rq2 = ( 3.0*r0  +  6.0*rp1 -      rp2 ) / 8.0;

    T rb0a = rm2 - 2.0*rm1 + r0;
    T rb0b = rm2 - 4.0*rm1 + 3.0*r0;

    T rb1a = rm1 - 2.0*r0 + rp1;
    T rb1b = rm1 - rp1;

    T rb2a = r0 - 2.0*rp1 + rp2;
    T rb2b = 3.0*r0 - 4.0*rp1 + rp2;

    T rbeta0 = (13.0/12.0)*rb0a*rb0a + 0.25*rb0b*rb0b;
    T rbeta1 = (13.0/12.0)*rb1a*rb1a + 0.25*rb1b*rb1b;
    T rbeta2 = (13.0/12.0)*rb2a*rb2a + 0.25*rb2b*rb2b;

    T rtau5 = abs(rbeta0 - rbeta2);

    T ralpha0 = d0*(1.0 + rtau5/(rbeta0 + eps));
    T ralpha1 = d1*(1.0 + rtau5/(rbeta1 + eps));
    T ralpha2 = d2*(1.0 + rtau5/(rbeta2 + eps));

    T rasum = ralpha0 + ralpha1 + ralpha2;

    a_WR(comp) = scale*((ralpha0*rq0 + ralpha1*rq1 + ralpha2*rq2)/rasum);
  }
}
PROTO_KERNEL_END(f_wenoz_pointwise_faceF, f_wenoz_pointwise_face)


template<typename T, MemType MEM, unsigned C = NUMCOMPS>
PROTO_KERNEL_START
void f_weno3_pointwise_faceF(
    Var<T,C,MEM>&       a_WL,
    Var<T,C,MEM>&       a_WR,
    const Var<T,C,MEM>& a_Wm1,
    const Var<T,C,MEM>& a_W0,
    const Var<T,C,MEM>& a_Wp1,
    const Var<T,C,MEM>& a_Wp2,
    const T                    a_dx2)
{
  const T d0  = 3.0/4.0;
  const T d1  = 1.0/4.0;
  const T eps = a_dx2;

  for (int comp = 0; comp < C; comp++)
  {
    T values[4] = {a_Wm1(comp),a_W0(comp),a_Wp1(comp),a_Wp2(comp)};
    const T scale = bertaNormalizeWenoStencil(values);
    if (scale == T(0))
    {
      a_WL(comp) = a_WR(comp) = T(0);
      continue;
    }
    // Left state using a_W0 as the left cell of the face.
    T vm1 = values[0];
    T v0  = values[1];
    T vp1 = values[2];

    T Dm = v0  - vm1;
    T Dp = vp1 - v0;

    T beta0 = Dp*Dp;
    T beta1 = Dm*Dm;
    T tau   = (Dp - Dm)*(Dp - Dm);

    T a0 = d0*(1.0 + tau/(beta0 + eps));
    T a1 = d1*(1.0 + tau/(beta1 + eps));

    T q0 = 0.5*(vp1 + v0);
    T q1 = 0.5*(-vm1 + 3.0*v0);

    a_WL(comp) = scale*((a0*q0 + a1*q1)/(a0 + a1));

    // Right state using a_Wp1 as the right cell of the face.
    T rm1 = values[3];
    T r0  = values[2];
    T rp1 = values[1];

    T rDm = r0  - rm1;
    T rDp = rp1 - r0;

    T rbeta0 = rDp*rDp;
    T rbeta1 = rDm*rDm;
    T rtau   = (rDp - rDm)*(rDp - rDm);

    T ra0 = d0*(1.0 + rtau/(rbeta0 + eps));
    T ra1 = d1*(1.0 + rtau/(rbeta1 + eps));

    T rq0 = 0.5*(rp1 + r0);
    T rq1 = 0.5*(-rm1 + 3.0*r0);

    a_WR(comp) = scale*((ra0*rq0 + ra1*rq1)/(ra0 + ra1));
  }
}
PROTO_KERNEL_END(f_weno3_pointwise_faceF, f_weno3_pointwise_face)


template<typename T>
inline T berta_minmod3_scalar(T a, T b, T c)
{
  if (a > 0.0 && b > 0.0 && c > 0.0) return min(a,min(b,c));
  if (a < 0.0 && b < 0.0 && c < 0.0) return max(a,max(b,c));
  return 0.0;
}


template<typename T, MemType MEM, unsigned C = NUMCOMPS>
PROTO_KERNEL_START
void f_plm_pointwise_faceF(
    Var<T,C,MEM>&       a_WL,
    Var<T,C,MEM>&       a_WR,
    const Var<T,C,MEM>& a_Wm1,
    const Var<T,C,MEM>& a_W0,
    const Var<T,C,MEM>& a_Wp1,
    const Var<T,C,MEM>& a_Wp2)
{
  const T theta = 1.5;

  for (int comp = 0; comp < C; comp++)
  {
    // Left state from cell represented by a_W0.
    T dm = a_W0(comp)  - a_Wm1(comp);
    T dp = a_Wp1(comp) - a_W0(comp);
    T dc = 0.5*(a_Wp1(comp) - a_Wm1(comp));

    T slopeL = berta_minmod3_scalar(theta*dm,dc,theta*dp);

    a_WL(comp) = a_W0(comp) + 0.5*slopeL;

    // Right state from cell represented by a_Wp1.
    T rdm = a_Wp1(comp) - a_W0(comp);
    T rdp = a_Wp2(comp) - a_Wp1(comp);
    T rdc = 0.5*(a_Wp2(comp) - a_W0(comp));

    T slopeR = berta_minmod3_scalar(theta*rdm,rdc,theta*rdp);

    a_WR(comp) = a_Wp1(comp) - 0.5*slopeR;
  }
}
PROTO_KERNEL_END(f_plm_pointwise_faceF, f_plm_pointwise_face)


template<typename T, MemType MEM>
PROTO_KERNEL_START
void f_face_theta_adjacentF(
    Var<T,1,MEM>&       a_thetaFace,
    const Var<T,1,MEM>& a_thetaLeft,
    const Var<T,1,MEM>& a_thetaRight)
{
  // Face i lies between cells i-1 and i.
  a_thetaFace(0) = min(a_thetaLeft(0),a_thetaRight(0));
}
PROTO_KERNEL_END(f_face_theta_adjacentF, f_face_theta_adjacent)


template<typename T, MemType MEM>
PROTO_KERNEL_START
void f_face_theta_recon_stencilF(
    Var<T,1,MEM>&       a_thetaFace,
    const Var<T,1,MEM>& a_thetam3,
    const Var<T,1,MEM>& a_thetam2,
    const Var<T,1,MEM>& a_thetam1,
    const Var<T,1,MEM>& a_theta0,
    const Var<T,1,MEM>& a_thetap1,
    const Var<T,1,MEM>& a_thetap2)
{
  // WENOZ at lower face of cell i uses cells i-3 ... i+2.
  // If any cell in that stencil is troubled, use fallback reconstruction.
  T tmin = a_thetam3(0);
  tmin = min(tmin,a_thetam2(0));
  tmin = min(tmin,a_thetam1(0));
  tmin = min(tmin,a_theta0(0));
  tmin = min(tmin,a_thetap1(0));
  tmin = min(tmin,a_thetap2(0));
  a_thetaFace(0) = tmin;
}
PROTO_KERNEL_END(f_face_theta_recon_stencilF, f_face_theta_recon_stencil)


template<typename T, MemType MEM>
PROTO_KERNEL_START
void f_select_reconstructionF(
    Var<T,NUMCOMPS,MEM>&       a_WL,
    Var<T,NUMCOMPS,MEM>&       a_WR,
    const Var<T,NUMCOMPS,MEM>& a_WL4,
    const Var<T,NUMCOMPS,MEM>& a_WR4,
    const Var<T,NUMCOMPS,MEM>& a_WLF,
    const Var<T,NUMCOMPS,MEM>& a_WRF,
    const Var<T,1,MEM>&        a_thetaReconFace)
{
  T th = a_thetaReconFace(0);

  if (th > 0.5)
  {
    for (int comp = 0; comp < NUMCOMPS; comp++)
    {
      a_WL(comp) = a_WL4(comp);
      a_WR(comp) = a_WR4(comp);
    }
  }
  else
  {
    for (int comp = 0; comp < NUMCOMPS; comp++)
    {
      a_WL(comp) = a_WLF(comp);
      a_WR(comp) = a_WRF(comp);
    }
  }

  // Positivity safety.
  a_WL(iRHO) = max(a_WL(iRHO),RHO00);
  a_WR(iRHO) = max(a_WR(iRHO),RHO00);

  a_WL(iP) = max(a_WL(iP),1.0e-12);
  a_WR(iP) = max(a_WR(iP),1.0e-12);
}
PROTO_KERNEL_END(f_select_reconstructionF, f_select_reconstruction)


// Split contiguous flag runs so CPU vectorization need not evaluate an
// unselected fallback. The scalar kernels are the same kernels used on devices.
template<unsigned METHOD,typename T>
inline void bertaReconstructLine(
    T* __restrict__ a_left,T* __restrict__ a_right,
    const T* const* a_source,int a_size,T a_dx2,bool a_clamp,T a_floor)
{
  Var<T,1,HOST> left{},right{},source[6]{};
  left.m_ptrs[0] = a_left;
  right.m_ptrs[0] = a_right;
  // Var stores mutable pointers even for const inputs. These source Vars are
  // passed only to the reconstruction kernels' const input parameters.
  for (unsigned i = 0; i < 6; ++i)
    source[i].m_ptrs[0] = const_cast<T*>(a_source[i]);
  // Let the optimizer choose vectorization for each reconstruction method.
  // Clang cannot honor a forced request for every instantiation of this loop.
  for (int cell = 0; cell < a_size; ++cell)
  {
    if constexpr (METHOD == 0)
      f_wenoz_pointwise_faceF(left,right,source[0],source[1],source[2],
          source[3],source[4],source[5]);
    else if constexpr (METHOD == 1)
      f_weno3_pointwise_faceF(left,right,source[1],source[2],source[3],
          source[4],a_dx2);
    else
      f_plm_pointwise_faceF(left,right,source[1],source[2],source[3],source[4]);
    if (a_clamp)
    {
      left(0) = max(left(0),a_floor);
      right(0) = max(right(0),a_floor);
    }
    ++left;
    ++right;
    for (unsigned i = 0; i < 6; ++i) ++source[i];
  }
}

template<typename T>
inline void bertaReconstructHostBox(
    BoxData<T,NUMCOMPS,HOST>& a_left,BoxData<T,NUMCOMPS,HOST>& a_right,
    const BoxData<T,NUMCOMPS,HOST>& a_wm3,
    const BoxData<T,NUMCOMPS,HOST>& a_wm2,
    const BoxData<T,NUMCOMPS,HOST>& a_wm1,
    const BoxData<T,NUMCOMPS,HOST>& a_w0,
    const BoxData<T,NUMCOMPS,HOST>& a_wp1,
    const BoxData<T,NUMCOMPS,HOST>& a_wp2,
    const BoxData<T,1,HOST>& a_theta,T a_dx2,int a_fallback)
{
  const Box& box = a_left.box();
  const Point low = box.low(),high = box.high();
  const int count = box.size(0);
  for (int k = low[2]; k <= high[2]; ++k)
    for (int j = low[1]; j <= high[1]; ++j)
    {
      const T* theta = a_theta.data(Point(low[0],j,k));
      int begin = 0;
      while (begin < count)
      {
        const bool smooth = theta[begin] > T(0.5);
        int end = begin+1;
        while (end < count && (theta[end] > T(0.5)) == smooth) ++end;
        const Point start(low[0]+begin,j,k);
        for (unsigned component = 0; component < NUMCOMPS; ++component)
        {
          const T* source[6] = {a_wm3.data(start,component),
              a_wm2.data(start,component),a_wm1.data(start,component),
              a_w0.data(start,component),a_wp1.data(start,component),
              a_wp2.data(start,component)};
          const bool clamp = component == iRHO || component == iP;
          const T floor = component == iRHO ? T(RHO00) : T(1.0e-12);
          if (smooth)
            bertaReconstructLine<0>(a_left.data(start,component),
                a_right.data(start,component),source,end-begin,a_dx2,clamp,floor);
          else if (a_fallback == 0)
            bertaReconstructLine<1>(a_left.data(start,component),
                a_right.data(start,component),source,end-begin,a_dx2,clamp,floor);
          else
            bertaReconstructLine<2>(a_left.data(start,component),
                a_right.data(start,component),source,end-begin,a_dx2,clamp,floor);
        }
        begin = end;
      }
    }
}

template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::BertaReconstructWithFallback(
    State&                  a_WFace_L,
    State&                  a_WFace_R,
    BoxData<T,1,MEM>&       a_thetaFluxFace,
    BoxData<T,1,MEM>&       a_thetaReconFace,
    const State&            a_WPoint,
    const BoxData<T,1,MEM>& a_thetaCell,
    uint                    a_dir,
    int                     a_fallback_type) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::BertaReconstructWithFallback()");

  Point e = Point::Basis(a_dir);

  // Face index i = lower face of cell i = x_{i-1/2}.
  // WENOZ needs cells i-3 ... i+2.
  // grow(-3) is conservative. You may later optimize asymmetrically.
  Box faceBox = a_WPoint.box().extrude(e).grow(-3);

  a_WFace_L.defineForOverwrite(faceBox);
  a_WFace_R.defineForOverwrite(faceBox);
  a_thetaFluxFace.defineForOverwrite(faceBox);
  a_thetaReconFace.defineForOverwrite(faceBox);

  const auto tm3 = bertaReadView(a_thetaCell,3*e);
  const auto tm2 = bertaReadView(a_thetaCell,2*e);
  const auto tm1 = bertaReadView(a_thetaCell,e);
  const auto t0 = bertaReadView(a_thetaCell);
  const auto tp1 = bertaReadView(a_thetaCell,-e);
  const auto tp2 = bertaReadView(a_thetaCell,-2*e);
  forallInPlace([] PROTO_LAMBDA(
      Var<T,1,MEM>& flux, Var<T,1,MEM>& recon,
      const Var<T,1,MEM>& tm3, const Var<T,1,MEM>& tm2,
      const Var<T,1,MEM>& tm1, const Var<T,1,MEM>& t0,
      const Var<T,1,MEM>& tp1, const Var<T,1,MEM>& tp2)
  {
    flux(0) = min(tm1(0),t0(0));
    T value = tm3(0);
    value = min(value,tm2(0));
    value = min(value,tm1(0));
    value = min(value,t0(0));
    value = min(value,tp1(0));
    recon(0) = min(value,tp2(0));
  },a_thetaFluxFace,a_thetaReconFace,tm3,tm2,tm1,t0,tp1,tp2);

  const auto wm3 = bertaReadView(a_WPoint,3*e);
  const auto wm2 = bertaReadView(a_WPoint,2*e);
  const auto wm1 = bertaReadView(a_WPoint,e);
  const auto w0 = bertaReadView(a_WPoint);
  const auto wp1 = bertaReadView(a_WPoint,-e);
  const auto wp2 = bertaReadView(a_WPoint,-2*e);
  T dx2 = this->dx()[a_dir]*this->dx()[a_dir];
  if constexpr (MEM == HOST)
  {
    bertaReconstructHostBox(a_WFace_L,a_WFace_R,wm3,wm2,wm1,w0,wp1,wp2,
        a_thetaReconFace,dx2,a_fallback_type);
    return;
  }
  // Evaluate only the selected reconstruction; both read the same point data.
  forallInPlace([] PROTO_LAMBDA(
      Var<T,NUMCOMPS,MEM>& left, Var<T,NUMCOMPS,MEM>& right,
      const Var<T,NUMCOMPS,MEM>& wm3, const Var<T,NUMCOMPS,MEM>& wm2,
      const Var<T,NUMCOMPS,MEM>& wm1, const Var<T,NUMCOMPS,MEM>& w0,
      const Var<T,NUMCOMPS,MEM>& wp1, const Var<T,NUMCOMPS,MEM>& wp2,
      const Var<T,1,MEM>& recon, T dx2, int fallback)
  {
    if (recon(0) > T(0.5))
      f_wenoz_pointwise_faceF(left,right,wm3,wm2,wm1,w0,wp1,wp2);
    else if (fallback == 0)
      f_weno3_pointwise_faceF(left,right,wm2,wm1,w0,wp1,dx2);
    else
      f_plm_pointwise_faceF(left,right,wm2,wm1,w0,wp1);
    left(iRHO) = max(left(iRHO),T(RHO00));
    right(iRHO) = max(right(iRHO),T(RHO00));
    left(iP) = max(left(iP),T(1.0e-12));
    right(iP) = max(right(iP),T(1.0e-12));
  },a_WFace_L,a_WFace_R,wm3,wm2,wm1,w0,wp1,wp2,
      a_thetaReconFace,dx2,a_fallback_type);
}


// ====================================================================================
// 5. Theta-aware point-to-average flux correction.
// ====================================================================================

template<typename T, MemType MEM>
PROTO_KERNEL_START
void f_flux_point_to_avg_thetaF(
    Var<T,NUMCOMPS,MEM>&       a_fluxAvg,
    const Var<T,NUMCOMPS,MEM>& a_fluxPoint,
    const Var<T,NUMCOMPS,MEM>& a_lapFluxPoint,
    const Var<T,1,MEM>&        a_thetaFluxFace)
{
  for (int comp = 0; comp < NUMCOMPS; comp++)
  {
    if (a_thetaFluxFace(0) > 0.5)
    {
      a_fluxAvg(comp) = a_fluxPoint(comp) + a_lapFluxPoint(comp)/24.0;
    }
    else
    {
      a_fluxAvg(comp) = a_fluxPoint(comp);
    }
  }
}
PROTO_KERNEL_END(f_flux_point_to_avg_thetaF, f_flux_point_to_avg_theta)


template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::BertaPrimPointToFluxAverage(
           State&                  a_physFlux,
           const State&            a_WFacePoint,
           const BoxData<T,1,MEM>& a_thetaFluxFace,
           const BoxData<T,1,MEM>& a_adjDdir,
           uint                    a_block,
           uint                    a_dir,
           const FaceEnergies* a_energy,
           const BoxData<T,1,MEM>* a_radialPoint) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::BertaPrimPointToFluxAverage()");

  FaceEnergies ownedEnergy;
  if (!a_energy)
  {
    ownedEnergy = faceEnergies(a_WFacePoint,a_block,a_dir);
    a_energy = &ownedEnergy;
  }
  BoxData<T,1,MEM> ownedRadial;
  if (!a_radialPoint)
  {
    if (a_dir != 0)
      ownedRadial = (1.0*Shift(Point::Zeros())
          - (1.0/24.0)*Stencil<T>::Derivative(2,0,2))(a_adjDdir);
    else
    {
      ownedRadial.define(a_adjDdir.box());
      a_adjDdir.copyTo(ownedRadial);
    }
    a_radialPoint = &ownedRadial;
  }
  State fluxPoint =
      forall<T,NUMCOMPS>(
          f_primflux,
          a_WFacePoint,
          a_energy->kinetic,
          a_energy->magnetic,
          a_energy->velocityDotMagnetic,
          *a_radialPoint,
          m_gamma,
          a_block,
          a_dir);

  State lapFluxPoint = Stencil<T>::LaplacianFace(a_dir)(fluxPoint);

  a_physFlux.define(fluxPoint.box());

  forallInPlace(
      f_flux_point_to_avg_theta,
      a_physFlux,
      fluxPoint,
      lapFluxPoint,
      a_thetaFluxFace);
}


// ====================================================================================
// 6. Mapped flux from pointwise face primitive state.
// ====================================================================================

template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::BertaPrimPointToMappedFlux(
    State&                         a_fluxdir,
    const State&                   a_physFluxAvg,
    const State&                   a_WFacePoint,
    const BoxData<T,1,MEM>&        a_thetaFluxFace,
    const BoxData<T,1,MEM>&        a_adjDrDir,
    const BoxData<T,DIM,MEM,DIM>&  a_deta_a,
    const BoxData<T,1,MEM>&        a_deta,
    const BoxData<T,DIM,MEM>&      a_ntrow,
    uint                           a_block,
    uint                           a_dir,
    const std::array<BoxData<T,DIM,MEM,DIM>,2>* a_metricDifferences,
    const BoxData<T,1,MEM>* a_magneticPoint,
    const BoxData<T,1,MEM>* a_totalPressurePoint,
    bool a_separableMetrics) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::BertaPrimPointToMappedFlux()");

  BoxData<T,DIM,MEM> fluxdirMom =
      slice<T,NUMCOMPS,DIM,MEM>(a_fluxdir,iMOMX);
  if constexpr (MEM == HOST && DIM == 3)
  {
    // Map all eight components in one pass. Preserve each product's original
    // transverse-correction and row-accumulation order without scalar buffers.
    const int firstDir = a_dir == 0 ? 1 : 0;
    const int secondDir = a_dir == 2 ? 1 : 2;
    const Stencil<T> firstDiff =
        0.5*Shift::Basis(firstDir,1) - 0.5*Shift::Basis(firstDir,-1);
    const Stencil<T> secondDiff =
        0.5*Shift::Basis(secondDir,1) - 0.5*Shift::Basis(secondDir,-1);
    const bool skipFirst = a_separableMetrics && firstDir == 0;
    State physicalFirst;
    State physicalSecond = secondDiff(a_physFluxAvg);
    Scalar detFirst;
    Scalar detSecond = secondDiff(a_deta);
    if (!skipFirst)
    {
      physicalFirst = firstDiff(a_physFluxAvg);
      detFirst = firstDiff(a_deta);
    }
    std::array<BoxData<T,DIM,MEM,DIM>,2> ownedDifferences;
    if (!a_metricDifferences)
    {
      ownedDifferences[0] = firstDiff(a_deta_a);
      ownedDifferences[1] = secondDiff(a_deta_a);
      a_metricDifferences = &ownedDifferences;
    }
    Box productBox = (a_physFluxAvg.box() & a_deta_a.box() & a_deta.box())
        .grow(-1).grow(a_dir,1);
    if (a_separableMetrics)
      bertaMapHostBox(a_fluxdir,a_physFluxAvg,
          skipFirst ? physicalSecond : physicalFirst,physicalSecond,
          a_deta_a,(*a_metricDifferences)[0],(*a_metricDifferences)[1],
          a_deta,skipFirst ? detSecond : detFirst,detSecond,productBox,skipFirst);
    else
    {
    forallInPlace([] PROTO_LAMBDA(
        Var<T,NUMCOMPS,MEM>& out, const Var<T,NUMCOMPS,MEM>& flux,
        const Var<T,NUMCOMPS,MEM>& df0, const Var<T,NUMCOMPS,MEM>& df1,
        const Var<T,DIM,MEM,DIM>& matrix,
        const Var<T,DIM,MEM,DIM>& dm0, const Var<T,DIM,MEM,DIM>& dm1,
        const Var<T,1,MEM>& det, const Var<T,1,MEM>& dd0,
        const Var<T,1,MEM>& dd1,bool skipFirst)
    {
      for (int comp : {iRHO,iE})
      {
        T value = T(0);
        if (!skipFirst) value += dd0(0)*df0(comp)*T(1.0/12.0);
        value += dd1(0)*df1(comp)*T(1.0/12.0);
        value += det(0)*flux(comp)*T(1);
        out(comp) = value;
      }
      for (int row = 0; row < DIM; ++row)
      {
        T momentum = T(0), magnetic = T(0);
        for (int col = 0; col < DIM; ++col)
        {
          T termMomentum = T(0), termMagnetic = T(0);
          if (!skipFirst)
            termMomentum += dm0(row,col)*df0(iMOMX+col)*T(1.0/12.0);
          termMomentum += dm1(row,col)*df1(iMOMX+col)*T(1.0/12.0);
          termMomentum += matrix(row,col)*flux(iMOMX+col)*T(1);
          if (!skipFirst)
            termMagnetic += dm0(row,col)*df0(iBX+col)*T(1.0/12.0);
          termMagnetic += dm1(row,col)*df1(iBX+col)*T(1.0/12.0);
          termMagnetic += matrix(row,col)*flux(iBX+col)*T(1);
          momentum += termMomentum;
          magnetic += termMagnetic;
        }
        out(iMOMX+row) = momentum;
        out(iBX+row) = magnetic;
      }
    },productBox,a_fluxdir,a_physFluxAvg,
        skipFirst ? physicalSecond : physicalFirst,physicalSecond,
        a_deta_a,(*a_metricDifferences)[0],(*a_metricDifferences)[1],
        a_deta,skipFirst ? detSecond : detFirst,detSecond,skipFirst);
    }
  }
  else
  {

    BoxData<T,DIM,MEM> physFluxMom =
        slice<T,NUMCOMPS,DIM,MEM>(a_physFluxAvg,iMOMX);

    BoxData<T,DIM,MEM> tmp =
        Operator::_faceMatrixProductAB(
            a_deta_a,physFluxMom,
            a_deta_a,physFluxMom,
            a_dir,a_metricDifferences);

    tmp.copyTo(fluxdirMom);

    BoxData<T,DIM,MEM> fluxdirB =
        slice<T,NUMCOMPS,DIM,MEM>(a_fluxdir,iBX);

    BoxData<T,DIM,MEM> physFluxB =
        slice<T,NUMCOMPS,DIM,MEM>(a_physFluxAvg,iBX);

    BoxData<T,DIM,MEM> tmp2 =
        Operator::_faceMatrixProductAB(
            a_deta_a,physFluxB,
            a_deta_a,physFluxB,
            a_dir,a_metricDifferences);

    tmp2.copyTo(fluxdirB);

    for (int comp : {iRHO,iE})
    {
      auto physical = slice(a_physFluxAvg,comp);
      auto mapped = faceProduct(a_deta,physical,a_dir);
      auto destination = slice(a_fluxdir,comp);
      mapped.copyTo(destination);
    }

  }

  const int radialCorrection = a_separableMetrics ? (a_dir == 0 ? -1 : 0) : -2;
  auto radialFlux = bertaFaceTensorProduct(
      a_fluxdir,a_adjDrDir,a_dir,radialCorrection);
  radialFlux.copyTo(a_fluxdir);

  // Pressure force from point total pressure, then theta-aware face averaging.
  BoxData<T,1,MEM> ownedMagnetic,ownedPressure;
  if (!a_totalPressurePoint)
  {
    if (!a_magneticPoint)
    {
      ownedMagnetic = magneticEnergyW(a_WFacePoint,a_block,a_dir);
      a_magneticPoint = &ownedMagnetic;
    }
    auto pPoint = slice(a_WFacePoint,iP);
    ownedPressure = pPoint + *a_magneticPoint;
    a_totalPressurePoint = &ownedPressure;
  }
  const auto& pTotPoint = *a_totalPressurePoint;

  BoxData<T,1,MEM> lapPTotPoint =
      Stencil<T>::LaplacianFace(a_dir)(pTotPoint);

  BoxData<T,1,MEM> pTotAvg(pTotPoint.box());

  forallInPlace(
      [ ] PROTO_LAMBDA
      (
        Var<T,1,MEM>&       a_pAvg,
        const Var<T,1,MEM>& a_pPoint,
        const Var<T,1,MEM>& a_lapP,
        const Var<T,1,MEM>& a_theta)
      {
        if (a_theta(0) > 0.5)
        {
          a_pAvg(0) = a_pPoint(0) + a_lapP(0)/24.0;
        }
        else
        {
          a_pAvg(0) = a_pPoint(0);
        }
      },
      pTotAvg,
      pTotPoint,
      lapPTotPoint,
      a_thetaFluxFace);

  BoxData<T,1,MEM> tmp3 = bertaFaceTensorProduct(
      pTotAvg,a_adjDrDir,a_dir,radialCorrection);

  BoxData<T,DIM,MEM> pforce(a_ntrow.box());
  pforce.setVal(0.0);

  const int angularCorrection = a_separableMetrics && a_dir != 0
      ? (a_dir == 1 ? 2 : 1) : -2;
  auto tmpPF = bertaFaceTensorProduct(a_ntrow,tmp3,a_dir,angularCorrection);
  tmpPF.copyTo(pforce);

  fluxdirMom += pforce;
}


// ====================================================================================
// 7. Rusanov dissipation from pointwise face states.
// ====================================================================================

template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::BertaPrimPointToMappedRusanovFlux(
    State&                         a_fluxdir,
    const State&                   a_WFacePoint_L,
    const State&                   a_WFacePoint_R,
    const BoxData<T,1,MEM>&        a_thetaFluxFace,
    const BoxData<T,1,MEM>&        a_adjDdir,
    const BoxData<T,DIM,MEM,DIM>&  a_deta_a,
    const BoxData<T,1,MEM>&        a_deta,
    const BoxData<T,DIM,MEM,DIM>&  a_A,
    const BoxData<T,DIM,MEM>&      a_ntrow,
    uint                           a_block,
    uint                           a_dir,
    const FaceEnergies* a_leftEnergy,
    const FaceEnergies* a_rightEnergy,
    const BoxData<T,1,MEM>* a_radialPoint) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::BertaPrimPointToMappedRusanovFlux()");

  BoxData<T,1,MEM> ownedRadial;
  if (!a_radialPoint)
  {
    if (a_dir != 0)
      ownedRadial = (1.0*Shift(Point::Zeros())
          - (1.0/24.0)*Stencil<T>::Derivative(2,0,2))(a_adjDdir);
    else
    {
      ownedRadial.define(a_adjDdir.box());
      a_adjDdir.copyTo(ownedRadial);
    }
    a_radialPoint = &ownedRadial;
  }
  FaceEnergies ownedLeft,ownedRight;
  if (!a_leftEnergy)
  {
    ownedLeft = faceEnergies(a_WFacePoint_L,a_block,a_dir);
    a_leftEnergy = &ownedLeft;
  }
  if (!a_rightEnergy)
  {
    ownedRight = faceEnergies(a_WFacePoint_R,a_block,a_dir);
    a_rightEnergy = &ownedRight;
  }
  // Face averaging is linear with the common theta. Average the pointwise
  // conserved jump once, rather than averaging each side and subtracting.
  State qPointDiff = forallOverwrite<T,NUMCOMPS,MEM>([] PROTO_LAMBDA(
      Var<T,NUMCOMPS,MEM>& out,
      const Var<T,NUMCOMPS,MEM>& left,
      const Var<T,NUMCOMPS,MEM>& right,
      const Var<T,1,MEM>& leftKinetic,
      const Var<T,1,MEM>& leftMagnetic,
      const Var<T,1,MEM>& rightKinetic,
      const Var<T,1,MEM>& rightMagnetic,
      const Var<T,1,MEM>& radial,
      T gamma)
  {
    out(iRHO) = radial(0)*right(iRHO) - radial(0)*left(iRHO);
    for (int dir = 0; dir < DIM; ++dir)
    {
      out(iMOMX+dir) = radial(0)*right(iRHO)*right(iVX+dir)
                    - radial(0)*left(iRHO)*left(iVX+dir);
      out(iBX+dir) = radial(0)*right(iBX+dir) - radial(0)*left(iBX+dir);
    }
    out(iE) = radial(0)*(rightKinetic(0) + rightMagnetic(0)
                        + right(iP)/(gamma - 1.0))
            - radial(0)*(leftKinetic(0) + leftMagnetic(0)
                        + left(iP)/(gamma - 1.0));
  },a_WFacePoint_L,a_WFacePoint_R,a_leftEnergy->kinetic,
      a_leftEnergy->magnetic,a_rightEnergy->kinetic,
      a_rightEnergy->magnetic,*a_radialPoint,m_gamma);
  State lapQDiff = Stencil<T>::LaplacianFace(a_dir)(qPointDiff);
  State qAvgDiff(qPointDiff.box(),T(0));
  forallInPlace(f_flux_point_to_avg_theta,qAvgDiff,qPointDiff,lapQDiff,
      a_thetaFluxFace);

  T one = 1.0;

  BoxData<T,DIM,MEM> fluxdirMom =
      slice<T,NUMCOMPS,DIM,MEM>(a_fluxdir,iMOMX);

  BoxData<T,DIM,MEM> Con_diff_Mom =
      slice<T,NUMCOMPS,DIM,MEM>(qAvgDiff,iMOMX);

  forallInPlace(f_matVecProd,fluxdirMom,a_deta_a,Con_diff_Mom,one);

  BoxData<T,DIM,MEM> fluxdirB =
      slice<T,NUMCOMPS,DIM,MEM>(a_fluxdir,iBX);

  BoxData<T,DIM,MEM> Con_diff_B =
      slice<T,NUMCOMPS,DIM,MEM>(qAvgDiff,iBX);

  forallInPlace(f_matVecProd,fluxdirB,a_deta_a,Con_diff_B,one);

  forallInPlace(
      [ ] PROTO_LAMBDA
      (
        Var<T,NUMCOMPS,MEM>&       a_Rusflux,
        const Var<T,NUMCOMPS,MEM>& a_fluxin,
        const Var<T,1,MEM>&        a_deta)
      {
        a_Rusflux(iRHO) = a_deta(0)*a_fluxin(iRHO);
        a_Rusflux(iE  ) = a_deta(0)*a_fluxin(iE);
      },
      a_fluxdir,
      qAvgDiff,
      a_deta);

  forallInPlace(
      [ ] PROTO_LAMBDA
      (
        Var<T,NUMCOMPS,MEM>& a_Rusflux,
        const Var<T,NUMCOMPS>& a_wface_L,
        const Var<T,NUMCOMPS>& a_wface_R,
        const Var<T,DIM,MEM,DIM>& a_Amatrix,
        const Var<T,1,MEM>& a_detA,
        const Var<T,DIM,MEM>& a_adjArow,
        const T a_gamma,
        const int a_d)
      {
        double rho = 0.5*(a_wface_L(iRHO) + a_wface_R(iRHO));
        double w_d = 0.5*(a_wface_L(iVX+a_d) + a_wface_R(iVX+a_d));

        double B_x = 0.5*(a_wface_L(iBX) + a_wface_R(iBX));
        double B_y = 0.5*(a_wface_L(iBY) + a_wface_R(iBY));
        double B_z = 0.5*(a_wface_L(iBZ) + a_wface_R(iBZ));

        double p = 0.5*(a_wface_L(iP) + a_wface_R(iP));
        if (p < 0.0) p = 0.0;

        rho = max(rho,RHO00);

        vecType bSph;
        vecType bCart;

        bSph[0] = B_x;
        bSph[1] = B_y;
        bSph[2] = B_z;

        bCart[0] = 0.0;
        bCart[1] = 0.0;
        bCart[2] = 0.0;

        double B_n   = 0.0;
        double B_mag = 0.0;

        for (int dir0 = 0; dir0 < DIM; dir0++)
        {
          for (int dir1 = 0; dir1 < DIM; dir1++)
          {
            bCart[dir0] += a_Amatrix(dir0,dir1)*bSph[dir1];
          }

          B_n   += a_Amatrix(dir0,0)*bCart[dir0];
          B_mag += bCart[dir0]*bCart[dir0];
        }

        B_mag = sqrt(B_mag);

        double ce = sqrt(a_gamma*p/rho);
        double aa = ce*ce + B_mag*B_mag/(4.0*M_PI*rho);
        double bb = abs(B_n)*ce/sqrt(M_PI*rho);

        double af =
            0.5*(sqrt(max(0.0,aa + bb)) + sqrt(max(0.0,aa - bb)));

        double A_d_mag =
            sqrt(a_Amatrix(0,a_d)*a_Amatrix(0,a_d)
               + a_Amatrix(1,a_d)*a_Amatrix(1,a_d)
               + a_Amatrix(2,a_d)*a_Amatrix(2,a_d));

        double speed = abs(w_d) + af*A_d_mag;

        for (int comp = 0; comp < NUMCOMPS; comp++)
        {
          a_Rusflux(comp) *= speed;
        }
      },
      a_fluxdir,
      a_WFacePoint_L,
      a_WFacePoint_R,
      a_A,
      a_deta,
      a_ntrow,
      m_gamma,
      a_dir);
}


// ====================================================================================
// 8. Main Berta-WENOZ operator.
// ====================================================================================

template<
    typename T,
    typename MAP,
    MemType MEM>
inline void
BoxOp_EulerCubedSphere<T,MAP,MEM>::BertaOperator
  (
   State&                          a_Rhs,
   Array<State, DIM>&              a_fluxes,
   const State&                    a_USph,
   T                               a_dt,
   T                               a_scale,
   int                             a_counter) const
{
  PR_TIMERS("BoxOp_EulerCubedSphere::operator()::BertaWENOZ");

  int a_block = this->block();

  Box bxRad = a_USph.box().grow(Point::Basis(0,2));

  BoxData<T,DIM,MEM> Dr(bxRad);
  BoxData<T,1,MEM>   radius(bxRad);
  BoxData<T,DIM,MEM> adjDr(bxRad);
  BoxData<T,1,MEM>   dVolr(bxRad);

  radialMetrics(radius,Dr,adjDr,dVolr,bxRad);

  Box bx = a_USph.box();

  // Detector/fallback settings.
  // Increase eta_d for less fallback. Decrease eta_d for more fallback.
  const T eta_d = ParseInputs::get_Berta_eta_d();
  const int detector_type = ParseInputs::get_Berta_detector_type(); // 0 = Jameson, 1 = derivative ratio.
  const int fallback_type = ParseInputs::get_Berta_fallback_type(); // 0 = WENO3, 1 = PLM.
  const bool useRoe = ParseInputs::get_Riemann_solver_type() == 2;

  // Detector uses V(<U>), not V(UPoint).
  auto cellBasis = pointwiseSphericalBasis(a_USph.box(),a_block,-1);
  State WAvgForDetector;
  sphAvgToPrim(WAvgForDetector,a_USph,a_block,&cellBasis);

  BoxData<T,1,MEM> thetaCell;
  if (detector_type == 1)
  {
    BertaComputeThetaDerivativeRatio(thetaCell,WAvgForDetector,eta_d);
  }
  else
  {
    BertaComputeThetaJameson(thetaCell,WAvgForDetector,eta_d);
  }

  // Berta average-to-point recovery.
  State UPoint;
  BertaAverageToPoint(UPoint,a_USph,thetaCell);

  // Pointwise primitive variables.
  State WPoint;
  sphPointToPrim(WPoint,UPoint,a_block,&cellBasis);

  T dxiPerp = this->dx()[1];
  T dxiR    = this->dx()[0];

  T half = 0.5;

  // The final cell-average source product needs one divergence ghost layer.
  BoxData<T,1,MEM> minusJDivB(a_Rhs.box().grow(1));
  minusJDivB.setVal(0.0);
  const bool magneticTest =
      isConstantMagneticFieldTest(ParseInputs::get_init_condition_type());
  BoxData<T,DIM,MEM> magneticAvg;
  if (ParseInputs::get_divergence_cleaning() == 1 && magneticTest)
    BertaPowellMagneticAverages(magneticAvg,a_USph,a_block);

  for (int dir = 0; dir < DIM; dir++)
  {
    PR_TIME("BoxOp_EulerCubedSphere::operator()::Berta directional flux");
    // Reuse the arena for normal patches. Large patches would retain several
    // gigabytes of short-lived intermediates until the direction is finished;
    // use ordinary allocation there so each helper releases its temporaries.
    std::unique_ptr<ScopedHostStack<MEM>> scratch;
    if (a_USph.box().size() <= (1 << 19))
      scratch = std::make_unique<ScopedHostStack<MEM>>();

    Point e = Point::Basis(dir);

    // Pointwise WENOZ reconstruction with WENO3/PLM fallback.
    State WFace_L;
    State WFace_R;
    BoxData<T,1,MEM> thetaFluxFace;
    BoxData<T,1,MEM> thetaReconFace;

    BertaReconstructWithFallback(
        WFace_L,
        WFace_R,
        thetaFluxFace,
        thetaReconFace,
        WPoint,
        thetaCell,
        dir,
        fallback_type);

    if (useRoe)
    {
      // Repair both final point states together so the centered physical
      // flux, Powell term and Roe dissipation consume the same donor states.
      const auto donorLeft = bertaReadView(WAvgForDetector,e);
      forallInPlace([] PROTO_LAMBDA(
          Var<T,NUMCOMPS,MEM>& left, Var<T,NUMCOMPS,MEM>& right,
          Var<T,1,MEM>& fluxTheta, Var<T,1,MEM>& reconTheta,
          const Var<T,NUMCOMPS,MEM>& cellLeft,
          const Var<T,NUMCOMPS,MEM>& cellRight)
      {
        if (MHD_Reconstruction::usablePrimitiveState<NUMCOMPS>(left,RHO00)
            && MHD_Reconstruction::usablePrimitiveState<NUMCOMPS>(right,RHO00))
          return;
        if (!MHD_Reconstruction::usablePrimitiveState<NUMCOMPS>(cellLeft,RHO00)
            || !MHD_Reconstruction::usablePrimitiveState<NUMCOMPS>(cellRight,RHO00))
          return;
        for (int c = 0; c < NUMCOMPS; ++c)
        {
          left(c) = cellLeft(c);
          right(c) = cellRight(c);
        }
        fluxTheta(0) = T(0);
        reconTheta(0) = T(0);
      },WFace_L,WFace_R,thetaFluxFace,thetaReconFace,
          donorLeft,WAvgForDetector);
    }

    Box bxface = WFace_L.box();

    // Mapping variables.
    BoxData<T,1,MEM> adjDrDir = slice(adjDr,dir);

    BoxData<T,DIM,MEM,DIM> deta_a =
        CubedSphereShell::detA_A<T,MEM>(bxface,dxiPerp,a_block,dir);

    BoxData<T,1,MEM> deta =
        CubedSphereShell::detA<T,MEM>(bxface,dxiPerp,dir);


    Box bxCell = WPoint.box();
    Box bxdir  = bxCell.extrude(Point::Basis(dir));

    BoxData<T,DIM,MEM> ntrow =
        CubedSphereShell::adjA<T,MEM>(bxdir,dxiPerp,a_block,dir);


    // Radial-face basis equals the cell basis. Angular faces need only the
    // reconstructed face box, not the complete five-layer cell halo.
    BoxData<T,DIM,MEM,DIM> ownedBasis;
    if (dir != 0)
      ownedBasis = pointwiseSphericalBasis(bxface,a_block,dir);
    const auto& A_matrix = dir == 0 ? cellBasis : ownedBasis;

    auto leftEnergy = faceEnergies(WFace_L,a_block,dir,&A_matrix);
    auto rightEnergy = faceEnergies(WFace_R,a_block,dir,&A_matrix);
    BoxData<T,1,MEM> radialPoint;
    if (dir != 0)
      radialPoint = (1.0*Shift(Point::Zeros())
          - (1.0/24.0)*Stencil<T>::Derivative(2,0,2))(adjDrDir);
    else
    {
      radialPoint.defineForOverwrite(adjDrDir.box());
      adjDrDir.copyTo(radialPoint);
    }
    std::array<BoxData<T,DIM,MEM,DIM>,2> metricDifferences;
    int differenceIndex = 0;
    for (int transverse = 0; transverse < DIM; ++transverse)
      if (transverse != dir)
        metricDifferences[differenceIndex++] =
            (0.5*Shift::Basis(transverse,1)
             - 0.5*Shift::Basis(transverse,-1))(deta_a);

    // Centered point state for Roe and the troubled-face Powell flux.
    State WFace_C(WFace_L.box());
    WFace_L.copyTo(WFace_C);
    WFace_C += WFace_R;
    WFace_C *= 0.5;

    if (ParseInputs::get_divergence_cleaning() == 1)
    {
      // The Powell flux needs face averages and a full source halo. Centered
      // interpolation of native magnetic averages supplies both without the
      // extra ghost layers needed to average the WENO face points twice.
      auto Bnormal = magneticTest ? slice(magneticAvg,dir) : slice(a_USph,iBX + dir);
      Scalar Bface = Stencil<T>::CellToFace(dir)(Bnormal);
      auto Bpoint = slice(WFace_C,iBX + dir);
      const Box sourceFaceBox = minusJDivB.box().extrude(dir).grow(
          2*(Point::Ones()-Point::Basis(dir)));
      auto sourceDetA = CubedSphereShell::detA<T,MEM>(sourceFaceBox,dxiPerp,dir);
      DivBPowell1(minusJDivB,Bface,sourceDetA,adjDrDir,dir,&Bpoint,&thetaFluxFace);
    }

    // Apply the common linear face average to F(WL)+F(WR) once. On the host,
    // local Vars let the shared point-flux kernel write small stack arrays.
    State physFluxSum;
    BoxData<T,1,MEM> pressureSum;
    if constexpr (MEM == HOST)
    {
      State fluxPointSum;
      fluxPointSum.defineForOverwrite(bxface);
      pressureSum.defineForOverwrite(bxface);
      forallInPlace([] PROTO_LAMBDA(
          Var<T,NUMCOMPS,MEM>& out, Var<T,1,MEM>& pressure,
          const Var<T,NUMCOMPS,MEM>& left,
          const Var<T,NUMCOMPS,MEM>& right,
          const Var<T,1,MEM>& leftKinetic,
          const Var<T,1,MEM>& leftMagnetic,
          const Var<T,1,MEM>& leftDot,
          const Var<T,1,MEM>& rightKinetic,
          const Var<T,1,MEM>& rightMagnetic,
          const Var<T,1,MEM>& rightDot,
          const Var<T,1,MEM>& radial,T gamma,uint block,uint direction)
      {
        T leftFlux[NUMCOMPS],rightFlux[NUMCOMPS];
        Var<T,NUMCOMPS,HOST> localLeft{},localRight{};
        for (int comp = 0; comp < NUMCOMPS; ++comp)
        {
          localLeft.m_ptrs[comp] = &leftFlux[comp];
          localRight.m_ptrs[comp] = &rightFlux[comp];
        }
        f_primflux_F(localLeft,left,leftKinetic,leftMagnetic,leftDot,
            radial,gamma,block,direction);
        f_primflux_F(localRight,right,rightKinetic,rightMagnetic,rightDot,
            radial,gamma,block,direction);
        for (int comp = 0; comp < NUMCOMPS; ++comp)
          out(comp) = leftFlux[comp] + rightFlux[comp];
        pressure(0) = (left(iP) + leftMagnetic(0))
                    + (right(iP) + rightMagnetic(0));
      },fluxPointSum,pressureSum,WFace_L,WFace_R,
          leftEnergy.kinetic,leftEnergy.magnetic,leftEnergy.velocityDotMagnetic,
          rightEnergy.kinetic,rightEnergy.magnetic,rightEnergy.velocityDotMagnetic,
          radialPoint,m_gamma,a_block,dir);
      State lapFluxSum = Stencil<T>::LaplacianFace(dir)(fluxPointSum);
      physFluxSum.define(fluxPointSum.box());
      forallInPlace(f_flux_point_to_avg_theta,physFluxSum,fluxPointSum,
          lapFluxSum,thetaFluxFace);
    }
    else
    {
      State physFluxLeft,physFluxRight;
      BertaPrimPointToFluxAverage(physFluxLeft,WFace_L,thetaFluxFace,
          adjDrDir,a_block,dir,&leftEnergy,&radialPoint);
      BertaPrimPointToFluxAverage(physFluxRight,WFace_R,thetaFluxFace,
          adjDrDir,a_block,dir,&rightEnergy,&radialPoint);
      physFluxSum = physFluxLeft + physFluxRight;
      pressureSum = forallOverwrite<T,1,MEM>([] PROTO_LAMBDA(
          Var<T,1,MEM>& out, const Var<T,NUMCOMPS,MEM>& left,
          const Var<T,NUMCOMPS,MEM>& right,
          const Var<T,1,MEM>& leftMagnetic,
          const Var<T,1,MEM>& rightMagnetic)
      {
        out(0) = (left(iP) + leftMagnetic(0))
               + (right(iP) + rightMagnetic(0));
      },WFace_L,WFace_R,leftEnergy.magnetic,rightEnergy.magnetic);
    }

    Box bxface0 = a_fluxes[dir].box();
    State fluxdirSum(bxface);
    State fluxdir_Diss(useRoe ? bxface0 : bxface);
    State fluxdir(bxface0);
    fluxdirSum.setVal(0.0);
    fluxdir_Diss.setVal(0.0);
    fluxdir.setVal(0.0);

    BertaPrimPointToMappedFlux(
        fluxdirSum,
        physFluxSum,
        WFace_L,
        thetaFluxFace,
        adjDrDir,
        deta_a,
        deta,
        ntrow,
        a_block,
        dir,&metricDifferences,nullptr,&pressureSum,true);

    if (useRoe)
      primToMappedRoeFlux(fluxdir_Diss,WFace_L,WFace_R,WFace_C,
          adjDrDir,deta_a,deta,A_matrix,dir,&WFace_L,&WFace_R,
          &radialPoint,&metricDifferences,&thetaFluxFace);
    else
      BertaPrimPointToMappedRusanovFlux(
        fluxdir_Diss,
        WFace_L,
        WFace_R,
        thetaFluxFace,
        adjDrDir,
        deta_a,
        deta,
        A_matrix,
        ntrow,
        a_block,
        dir,&leftEnergy,&rightEnergy,&radialPoint);

    forallInPlace(
      [ ] PROTO_LAMBDA
      (
        Var<T,NUMCOMPS,MEM>&       a_fluxout,
        const Var<T,NUMCOMPS,MEM>& a_fluxSum,
        const Var<T,NUMCOMPS,MEM>& a_fluxin_Diss)
      {
        for (int comp = 0; comp < NUMCOMPS; comp++)
        {
          a_fluxout(comp) =
              0.5*(a_fluxSum(comp) - a_fluxin_Diss(comp));
        }
      },
      fluxdir,
      fluxdirSum,
      fluxdir_Diss);

    fluxdir.copyTo(a_fluxes[dir]);

    T dxInv = -a_scale/(this->dx()[dir]);

    BoxData<T,NUMCOMPS,MEM> rhsIncr;
    rhsIncr = Stencil<T>::FluxDivergence(dir)(fluxdir,dxInv);

    forallInPlace(
      [ ] PROTO_LAMBDA
      (
        Var<T,NUMCOMPS,MEM>&       a_rhs,
        const Var<T,NUMCOMPS,MEM>& a_rhsIncr)
      {
        for (int comp = 0; comp < NUMCOMPS; comp++)
        {
          a_rhs(comp) += a_rhsIncr(comp);
        }
      },
      a_Rhs,
      rhsIncr);
  }

  if (ParseInputs::get_divergence_cleaning() == 1)
  {
    DivBPowell2(a_Rhs,minusJDivB,WPoint,cellBasis,false);
  }
}

// ====================================================================================
// Berta-style 4th-order FV pointwise-WENOZ implementation ends here.
// ====================================================================================
