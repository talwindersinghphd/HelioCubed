#include "MHD_Error.H"
#include "Proto.H"
#include "MHDOp.H"
#include "Proto_Timer.H"
// #include "Proto_WriteBoxData.H"
// For Chrono Timer (Talwinder)
#include <chrono>
#include <iostream>
#include <iomanip>

//////////////////////////////
#include "MHD_Limiters.H"
#include "MHD_Mapping.H"
#include "MHD_Riemann_Solvers.H"
#include "MHD_Output_Writer.H"
#include "MHD_Set_Boundary_Values.H"
#include "MHD_Input_Parsing.H"
#include "MHD_Constants.H"
#include "MHD_CFL.H"
#include "MHD_Gravity.H"
#include "MHD_Rotation.H"
#include "MHDLevelDataRK4.H"
#include "MHD_Turbulence.H"
#include "PolarExchangeCopier.H"

extern Parsefrominputs inputs;

typedef BoxData<double,1,HOST> Scalar;
typedef BoxData<double,NUMCOMPS,HOST> Vector;
/// @brief MHDOp namespace
namespace MHDOp {
	/**
	 * @brief Function to covert conserevd variables to primitive variables. 
	 * @param a_W the output primitive variables.
	 * @param a_U the input conserved variables.
	 * @param a_gamma gamma.
	 */ 
	PROTO_KERNEL_START
	void
	consToPrimF(State&         a_W,
	            const State&   a_U,
	            double a_gamma)
	{
		double rho = a_U(0);
		double v2 = 0.0;
		double B2 = 0.0;
		double gamma = a_gamma;
		a_W(0) = rho;

		for (int i = 1; i <= DIM; i++)
		{
			double v, B;
			v = a_U(i) / rho;
			B = a_U(DIM+1+i);
			a_W(i) = v;
			a_W(DIM+1+i) = a_U(DIM+1+i);
			v2 += v*v;
			B2 += B*B;
		}

		a_W(DIM+1) = (a_U(DIM+1) - .5 * rho * v2  - B2/8.0/c_PI) * (gamma - 1.0);

		#if TURB == 1

			a_W(DIM+1) = (a_U(DIM+1) - .5 * rho * v2  - B2/8.0/c_PI - 0.5*a_U(iRHOZ2)) * (gamma - 1.0);

			a_W(iZ2) = a_U(iRHOZ2)/rho;
			a_W(iSIGMA) = a_U(iRHOZ2SIGMA)/a_U(iRHOZ2);
			a_W(iLAMBDA) = a_U(iRHOLAMBDA)/rho;
		#endif

	}
	PROTO_KERNEL_END(consToPrimF, consToPrim)


	/**
	 * @brief Function to covert conserevd variables to primitive variables. 
	 * @param a_W the output primitive variables.
	 * @param a_U the input conserved variables.
	 * @param a_gamma gamma.
	 */ 
	void consToPrimcalc(BoxData<double,NUMCOMPS>& a_W,
	                    const BoxData<double,NUMCOMPS>& a_U,
	                    const double gamma)
	{
		a_W = forall<double,NUMCOMPS>(consToPrim,a_U, gamma);
	}


	/**
	 * @brief Function to covert dimensional variables to non-dimensional variables. 
	 * @param a_U will get converted from dimensional to non-dimensional.
	 */  
	PROTO_KERNEL_START
	void
	DimToNonDimF(const Point& a_pt,
				 State& a_U)
	{
		// When going from cgs to non-dimensional
		// density is divided by a scale density density_scale
		// velocity is divided by a scale velocity velocity_scale
		// energy is divided by density_scale*velocity_scale*velocity_scale
		// magnetic field is divided by sqrt(density_scale*velocity_scale*velocity_scale)
		// pref = (lismN*eos_mp)*lismV*lismV	
		// CMEData(CHF_IX[i;j;k],0) = (mCME)/(lismN*eos_mp) !rhoTD
		// CMEData(CHF_IX[i;j;k],1) = B_x_dom/sqrt( pref ) !Bx
		// CMEData(CHF_IX[i;j;k],2) = B_y_dom/sqrt( pref ) !By
		// CMEData(CHF_IX[i;j;k],3) = B_z_dom/sqrt( pref ) !Bz
		// CMEData(CHF_IX[i;j;k],4) = energy_control*e0TD/pref !e0TD
		// CMEData(CHF_IX[i;j;k],5) = V_x_dom*1.0e5/lismV !Vx
		// CMEData(CHF_IX[i;j;k],6) = V_y_dom*1.0e5/lismV !Vy
		// CMEData(CHF_IX[i;j;k],7) = V_z_dom*1.0e5/lismV !Vz
		double pref = (inputs.density_scale*c_MP*inputs.velocity_scale*inputs.velocity_scale);
		
		a_U(0) = a_U(0)/(inputs.density_scale*c_MP);
		a_U(1) = a_U(1)/(inputs.density_scale*c_MP*inputs.velocity_scale);
		a_U(2) = a_U(2)/(inputs.density_scale*c_MP*inputs.velocity_scale);
		a_U(3) = a_U(3)/(inputs.density_scale*c_MP*inputs.velocity_scale);
		a_U(4) = a_U(4)/pref;
		a_U(5) = a_U(5)/sqrt(pref);
		a_U(6) = a_U(6)/sqrt(pref);
		a_U(7) = a_U(7)/sqrt(pref);
		#if TURB == 1
		a_U(8) = a_U(8)/pref;
		a_U(9) = a_U(9)/pref;
		a_U(10) = a_U(10)/(inputs.density_scale*c_MP);
		#endif
		
	}
	PROTO_KERNEL_END(DimToNonDimF, DimToNonDim)


	/**
	 * @brief Function to covert dimensional variables to non-dimensional variables. 
	 * @param a_U will get converted from dimensional to non-dimensional.
	 */ 
	void DimToNonDimcalc(BoxData<double,NUMCOMPS>& a_U)
	{
		forallInPlace_p(DimToNonDim, a_U);
	}

	/**
	 * @brief Function to covert non-dimensional variables to dimensional variables. 
	 * @param a_U will get converted from non-dimensional to dimensional.
	 */  
	PROTO_KERNEL_START
	void
	NonDimToDimF(const Point& a_pt,
				 State& a_U)
	{
		// When going from non-dimensional to cgs
		// density is multiplied by a scale density density_scale
		// velocity is multiplied by a scale velocity velocity_scale
		// energy is multiplied by density_scale*velocity_scale*velocity_scale
		// magnetic field is multiplied by sqrt(density_scale*velocity_scale*velocity_scale)

		double pref = (inputs.density_scale*c_MP*inputs.velocity_scale*inputs.velocity_scale);
		
		a_U(0) = a_U(0)*(inputs.density_scale*c_MP);
		a_U(1) = a_U(1)*(inputs.density_scale*c_MP*inputs.velocity_scale);
		a_U(2) = a_U(2)*(inputs.density_scale*c_MP*inputs.velocity_scale);
		a_U(3) = a_U(3)*(inputs.density_scale*c_MP*inputs.velocity_scale);
		a_U(4) = a_U(4)*pref;
		a_U(5) = a_U(5)*sqrt(pref);
		a_U(6) = a_U(6)*sqrt(pref);
		a_U(7) = a_U(7)*sqrt(pref);
		#if TURB == 1
		a_U(8) = a_U(8)*pref;
		a_U(9) = a_U(9)*pref;
		a_U(10) = a_U(10)*(inputs.density_scale*c_MP);
		#endif
		
	}
	PROTO_KERNEL_END(NonDimToDimF, NonDimToDim)


	/**
	 * @brief Function to covert non-dimensional variables to dimensional variables. 
	 * @param a_U will get converted from non-dimensional to dimensional.
	 */ 
	void NonDimToDimcalc(BoxData<double,NUMCOMPS>& a_U)
	{
		forallInPlace_p(NonDimToDim, a_U);
	}


	/**
	 * @brief Function to covert conserevd variables to primitive variables, used in spherical mapping. 
	 * @param a_W_sph the output primitive variables.
	 * @param a_U_sph the input conserved variables, but the v and B vectors are scaled with row magnitudes of A matrix (see overleaf document).
	 * @param a_U_sph_actual the input conserved variables.
	 * @param a_gamma gamma.
	 */
	PROTO_KERNEL_START
	void
	consToPrimSphF(State&         a_W_sph,
	            const State&   a_U_sph,
	            const State&   a_U_sph_actual,
	            double a_gamma)
	{
		double rho = a_U_sph(0);
		double v2 = 0.0;
		double B2 = 0.0;
		double gamma = a_gamma;
		a_W_sph(0) = rho;

		for (int i = 1; i <= DIM; i++)
		{
			double v, v_actual, B, B_actual;
			v = a_U_sph(i) / rho;
			v_actual = a_U_sph_actual(i) / rho;
			B = a_U_sph(DIM+1+i);
			B_actual = a_U_sph_actual(DIM+1+i);
			a_W_sph(i) = v;
			a_W_sph(DIM+1+i) = a_U_sph(DIM+1+i);
			v2 += v_actual*v_actual;
			B2 += B_actual*B_actual;
		}

		a_W_sph(DIM+1) = (a_U_sph(DIM+1) - .5 * rho * v2  - B2/8.0/c_PI) * (gamma - 1.0);
		// a_W_sph(DIM+1) = a_U_sph(DIM+1);

	}
	PROTO_KERNEL_END(consToPrimSphF, consToPrimSph)

	/**
	 * @brief Function to covert conserevd variables to primitive variables, used in spherical mapping. 
	 * @param a_W_sph the output primitive variables.
	 * @param a_U_sph the input conserved variables, but the v and B vectors are scaled with row magnitudes of A matrix (see overleaf document).
	 * @param a_U_sph_actual the input conserved variables.
	 * @param a_gamma gamma.
	 */
	void consToPrimSphcalc(BoxData<double,NUMCOMPS>& a_W_sph,
	                    const BoxData<double,NUMCOMPS>& a_U_sph,
	                    const BoxData<double,NUMCOMPS>& a_U_sph_actual,
	                    const double gamma)
	{
		a_W_sph = forall<double,NUMCOMPS>(consToPrimSph,a_U_sph,a_U_sph_actual, gamma);
	}

	/**
	 * @brief Function to calculate the Powell term, without the divB multiplied to it. 
	 * @param a_P the output terms.
	 * @param a_W the input primitive variables.
	 */
	PROTO_KERNEL_START
	void PowellF(State&         a_P,
	             const State&   a_W)
	{
		a_P(0) = 0.;
		a_P(1) = a_W(5)/4.0/c_PI;
		a_P(2) = a_W(6)/4.0/c_PI;
		a_P(3) = a_W(7)/4.0/c_PI;
		a_P(4) = a_W(1)*a_W(5)/4.0/c_PI + a_W(2)*a_W(6)/4.0/c_PI + a_W(3)*a_W(7)/4.0/c_PI;
		a_P(5) = a_W(1);
		a_P(6) = a_W(2);
		a_P(7) = a_W(3);
	}
	PROTO_KERNEL_END(PowellF, Powell)


	/**
	 * @brief Function to average cell centered B on both sides of a face. 
	 * @param a_Bavg the output BoxData.
	 * @param a_W_ave the input cell averaged BoxData of primitive variables.
	 * @param a_d direction to average.
	 */ 
	PROTO_KERNEL_START
	void BavgcalcF(State& a_Bavg,
	               const State& a_W_ave,
	               int a_d)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			a_Bavg(i) = a_W_ave(2+DIM+a_d);
		}
	}
	PROTO_KERNEL_END(BavgcalcF, Bavgcalc)
	
	/**
	 * @brief Function to transfer data from lower dimensional BoxData to a higher dimensional BoxData. 
	 * @param a_F the output BoxData.
	 * @param a_F_temp the input BoxData.
	 * @param a_s where to transfer.
	 */ 
	PROTO_KERNEL_START
	void Fill_flux_calcF(const Point& a_pt,
						Var<double,DIM, MEM ,NUMCOMPS>&       a_F,
						const Var<double,NUMCOMPS>&        a_F_temp,
	            		const int a_s)
	{
		for (int i=0; i<NUMCOMPS; i++){
			a_F(a_s,i) = a_F_temp(i);
		}
	}
	PROTO_KERNEL_END(Fill_flux_calcF, Fill_flux_calc)

	/**
	 * @brief Function to transpose BoxData. 
	 * @param a_F the output BoxData.
	 * @param a_F_temp the input BoxData.
	 */ 
	PROTO_KERNEL_START
	void Transpose_calcF(const Point& a_pt,
						Var<double,NUMCOMPS>&       a_F,
						const Var<double,1,MEM,NUMCOMPS>&        a_F_temp)
	{
		for (int i=0; i<NUMCOMPS; i++){
			a_F(i) = a_F_temp(0,i);
		}
	}
	PROTO_KERNEL_END(Transpose_calcF, Transpose_calc)


	/**
	 * @brief Function to make pressure epsilon if it goes to negative for some reason. 
	 * @param a_U the input LevelBoxData containing conserved variables.
	 * @param a_gamma gamma.
	 */ 
	PROTO_KERNEL_START
	void Fix_negative_P_calcF(const Point& a_pt,
						State&         a_U,
	            		double a_gamma)
	{
		
		double rho = a_U(0);
		double v2 = 0.0;
		double B2 = 0.0;
		double gamma = a_gamma;

		for (int i = 1; i <= DIM; i++)
		{
			double v, B;
			v = a_U(i) / rho;
			B = a_U(DIM+1+i);
			v2 += v*v;
			B2 += B*B;
		}

		double p = std::max((a_U(DIM+1) - .5 * rho * v2  - B2/8.0/c_PI) * (gamma - 1.0),1.0e-14);
		a_U(DIM+1) = p/(gamma-1.0) + .5 * rho * v2  + B2/8.0/c_PI;
		
	}
	PROTO_KERNEL_END(Fix_negative_P_calcF, Fix_negative_P_calc)

	/**
	 * @brief Function to make pressure epsilon if it goes to negative for some reason. 
	 * @param a_U the input LevelBoxData containing conserved variables.
	 * @param a_gamma gamma.
	 */ 
	void Fix_negative_P(BoxData<double,NUMCOMPS>& a_U,
	                    const double gamma)
	{
		forallInPlace_p(Fix_negative_P_calc, a_U, gamma);
	}

	/**
	 * @brief Function to scale flux with face area. 
	 * @param a_F_scaled the output scaled flux.
	 * @param a_F the input flux.
	 * @param a_face_area the input face area.
	 * @param a_d direction index.
	 */ 
	PROTO_KERNEL_START
	void Scale_with_A_Ff_calcF(const Point& a_pt,
						Var<double,NUMCOMPS>& a_F_scaled,
						Var<double,NUMCOMPS>& a_F,
	                   	Var<double,DIM>& a_face_area,
	                   	int a_d)
	{
		double area = a_face_area(a_d);

		for (int i = 0; i < NUMCOMPS; i++){
			a_F_scaled(i) = -a_F(i)*area;
		}
	}
	PROTO_KERNEL_END(Scale_with_A_Ff_calcF, Scale_with_A_Ff_calc)


	/**
	 * @brief Function to scale B at face with face area. 
	 * @param a_B_scaled the output scaled B_face.
	 * @param a_W_sph1 the input primitive variables on low side.
	 * @param a_W_sph2 the input primitive variables on high side.
	 * @param a_face_area the input face area.
	 * @param a_d direction index.
	 */ 
	PROTO_KERNEL_START
	void Scale_with_A_Bf_calcF(const Point& a_pt,
						Var<double,1>& a_B_scaled,
						const State& a_W_sph1,
	               		const State& a_W_sph2,
	                   	Var<double,DIM>& a_face_area,
	                   	int a_d)
	{
		double area = a_face_area(a_d);
		double B_face = 0.5*(a_W_sph1(2+DIM+a_d)+a_W_sph2(2+DIM+a_d));
		a_B_scaled(0) = -B_face*area;
		
	}
	PROTO_KERNEL_END(Scale_with_A_Bf_calcF, Scale_with_A_Bf_calc)

	/**
	 * @brief Function to scale V at face with face area. 
	 * @param a_V_scaled the output scaled V_face.
	 * @param a_W_sph1 the input primitive variables on low side.
	 * @param a_W_sph2 the input primitive variables on high side.
	 * @param a_face_area the input face area.
	 * @param a_d direction index.
	 */ 
	PROTO_KERNEL_START
	void Scale_with_A_Vf_calcF(const Point& a_pt,
						Var<double,1>& a_V_scaled,
						const State& a_W_sph1,
	               		const State& a_W_sph2,
	                   	Var<double,DIM>& a_face_area,
	                   	int a_d)
	{
		double area = a_face_area(a_d);
		double V_face = 0.5*(a_W_sph1(iVX+a_d)+a_W_sph2(iVX+a_d));
		a_V_scaled(0) = V_face*area;
	}
	PROTO_KERNEL_END(Scale_with_A_Vf_calcF, Scale_with_A_Vf_calc)

	/**
	 * @brief Function to scale flux at a face with cell volume. 
	 * @param a_F_scaled the output scaled flux.
	 * @param a_F the input flux.
	 * @param a_cell_volume the input cell volume.
	 */ 
	PROTO_KERNEL_START
	void Scale_with_V_calcF(const Point& a_pt,
						Var<double,NUMCOMPS>& a_F_scaled,
						Var<double,NUMCOMPS>& a_F,
	                   	Var<double,1>& a_cell_volume)
	{
		double volume = a_cell_volume(0);

		for (int i = 0; i < NUMCOMPS; i++){
			a_F_scaled(i) = a_F(i)/volume;
		}
	}
	PROTO_KERNEL_END(Scale_with_V_calcF, Scale_with_V_calc)

	/**
	 * @brief Function to scale a scalar flux at a face with cell volume. 
	 * @param a_F_scaled the output scaled flux.
	 * @param a_F the input flux.
	 * @param a_cell_volume the input cell volume.
	 */ 
	PROTO_KERNEL_START
	void Scale_with_V2_calcF(const Point& a_pt,
						Var<double,NUMCOMPS>& a_F_scaled,
						Var<double,1>& a_F,
	                   	Var<double,1>& a_cell_volume)
	{
		double volume = a_cell_volume(0);

		for (int i = 0; i < NUMCOMPS; i++){
			a_F_scaled(i) = a_F(0)/volume;
		}
	}
	PROTO_KERNEL_END(Scale_with_V2_calcF, Scale_with_V2_calc)

	/**
	 * @brief Function to scale a scalar flux at a face with cell volume. 
	 * @param a_F_scaled the output scaled flux.
	 * @param a_F the input flux.
	 * @param a_cell_volume the input cell volume.
	 */ 
	PROTO_KERNEL_START
	void Scale_with_V3_calcF(const Point& a_pt,
						Var<double,1>& a_F_scaled,
						Var<double,1>& a_F,
	                   	Var<double,1>& a_cell_volume)
	{
		double volume = a_cell_volume(0);

		a_F_scaled(0) = a_F(0)/volume;
		
	}
	PROTO_KERNEL_END(Scale_with_V3_calcF, Scale_with_V3_calc)

	/**
	 * @brief Function to scale a scalar flux at a face with cell volume. 
	 * @param a_F_scaled the output scaled flux.
	 * @param a_F the input flux.
	 * @param a_cell_volume the input cell volume.
	 */ 
	PROTO_KERNEL_START
	void Add_Sources_calcF(const Point& a_pt,
						Var<double,NUMCOMPS>& a_Rhs,
						Var<double,NUMCOMPS>& a_S)
	{
		for (int i = 0; i < NUMCOMPS; i++){
			a_Rhs(i) += a_S(i);
		}
	}
	PROTO_KERNEL_END(Add_Sources_calcF, Add_Sources_calc)


	/**
	 * @brief Function to calculate Powell term for div B calculation. 
	 * @param a_P the output powell terms.
	 * @param a_W the input primitive variables.
	 * @param a_F the input divB.
	 * @param a_cell_volume the input cell volume.
	 */
	PROTO_KERNEL_START
	void Powell_Sph_2OF(const Point& a_pt,
				 State&         a_P,
	             const State&   a_W,
				 Var<double,1>& a_F,
				 Var<double,1>& a_cell_volume)
	{
		double volume = a_cell_volume(0);
		a_P(0) = (a_F(0)/volume)*0.;
		a_P(1) = (a_F(0)/volume)*a_W(5)/4.0/c_PI;
		a_P(2) = (a_F(0)/volume)*a_W(6)/4.0/c_PI;
		a_P(3) = (a_F(0)/volume)*a_W(7)/4.0/c_PI;
		a_P(4) = (a_F(0)/volume)*(a_W(1)*a_W(5)/4.0/c_PI + a_W(2)*a_W(6)/4.0/c_PI + a_W(3)*a_W(7)/4.0/c_PI);
		a_P(5) = (a_F(0)/volume)*a_W(1);
		a_P(6) = (a_F(0)/volume)*a_W(2);
		a_P(7) = (a_F(0)/volume)*a_W(3);
		// Powell cleaning has no source for the extra turbulence variables.
		for (int i = 2*DIM+2; i < NUMCOMPS; ++i) a_P(i) = 0.0;
	}
	PROTO_KERNEL_END(Powell_Sph_2OF, Powell_Sph_2O)



	/**
	 * @brief Function to calculate B at face. 
	 * @param a_B_face the output B a face.
	 * @param a_W_sph1 the input primitive variables on low side.
	 * @param a_W_sph2 the input primitive variables on high side.
	 * @param a_d the input direction.
	 */
	PROTO_KERNEL_START
	void B_face_calcF(const Point& a_pt,
                    Var<double,1>& a_B_face,
	               const State& a_W_sph1,
	               const State& a_W_sph2,
	               int a_d)
	{
		a_B_face(0) = 0.5*(a_W_sph1(2+DIM+a_d)+a_W_sph2(2+DIM+a_d));
	}
	PROTO_KERNEL_END(B_face_calcF, B_face_calc)
	

	PROTO_KERNEL_START
	void QuotientF(State&         a_RHSbyJ,
	             const State&   a_RHS,
				 const Var<double,1>& a_J)
	{
		for (int i=0;i<NUMCOMPS;i++){
			a_RHSbyJ(i) = a_RHS(i)/a_J(0);
		}
	}
	PROTO_KERNEL_END(QuotientF, Quotient)

	PROTO_KERNEL_START
	void ProductF(State&         a_a,
	             const State&   a_b,
				 const Var<double,1>& a_c)
	{
		for (int i=0;i<NUMCOMPS;i++){
			a_a(i) = a_b(i)*a_c(0);
		}
	}
	PROTO_KERNEL_END(ProductF, Product)

	/**
	 * @brief Function to calculate the right hand side of the finite volume method, Magnetic field divergence, and min dt according to CFL condition. 
	 * @param a_Rhs the output LevelBoxData.
	 * @param a_U the input LevelBoxData containing conserved variables.
	 */
	void step_spherical_2O(LevelBoxData<double,NUMCOMPS>& a_Rhs,
			  LevelBoxData<double,NUMCOMPS>& a_U,
			  MHDLevelDataState& a_State,
			  double& a_min_dt)
	{
		
		using namespace std;
		double a_dx = a_State.m_dx;
		double a_dy = a_State.m_dy;
		double a_dz = a_State.m_dz;
		double gamma = a_State.m_gamma;
		double dxd[3] = {a_dx, a_dy, a_dz};
		
		static Stencil<double> m_divergence[DIM];
		static bool initialized = false;
		if(!initialized)
		{
			for (int dir = 0; dir < DIM; dir++)
			{
				m_divergence[dir] = Stencil<double>::FluxDivergence(dir);
			}
			initialized =  true;
		}
		double dt_new;
		for (auto dit : a_State.m_U){	
			Box dbx0 = a_U[dit].box();
			// Box dbx1 = a_State.m_U[dit].box();
			Box dbx1 = dbx0.grow(0-NGHOST);
			Vector F_f_sph(dbx0), F_f(dbx0), F_f_scaled(dbx0), Rhs_d(dbx0), RhsV(dbx0);
			Scalar B_f_sph(dbx0), B_f_scaled(dbx0), Rhs_d_divB(dbx0), RhsV_divB(dbx0);
			#if TURB == 1
				Scalar V_f_scaled(dbx0), Rhs_d_divV(dbx0), RhsV_divV(dbx0), divV(dbx0);
				RhsV_divV.setVal(0.0);
				Vector STM(dbx0);
			#endif
			RhsV.setVal(0.0);
			RhsV_divB.setVal(0.0);

			MHDOp::Fix_negative_P(a_U[dit],inputs.gamma);	
			Vector W_low_temp(dbx0), W_high_temp(dbx0), W_low(dbx0), W_high(dbx0);
			BoxData<double,NUMCOMPS> W_sph(dbx0);
			Vector W_cart  = forall<double,NUMCOMPS>(consToPrim, a_U[dit], gamma);
			MHD_Mapping::Cartesian_to_Spherical(W_sph, W_cart, a_State.m_x_sph_cc[dit]);
			MHD_Mapping::Correct_V_theta_phi_at_poles(W_sph, a_dx, a_dy, a_dz);	
			HDF5Handler h5;

			// Vector W_boxed(dbx1);
			// W_sph.copyTo(W_boxed);
			// if (procID() == 0) h5.writePatch({"density","Vx","Vy","Vz", "p","Bx","By","Bz"}, 1, W_boxed, "W_2");

			if (!a_State.m_min_dt_calculated){ 
				MHD_CFL::Min_dt_calc_func(dt_new, W_sph, dbx1, a_dx, a_dy, a_dz, gamma);	
				if (dt_new < a_min_dt) a_min_dt = dt_new;
			}
			// if (procID() == 0) h5.writePatch({"density","Vx","Vy","Vz", "p","Bx","By","Bz"}, 1, W_sph, "W_ave2O");
			for (int d = 0; d < DIM; d++)
			{
				MHD_Limiters::MHD_Limiters_minmod(W_low,W_high,W_sph,a_State.m_x_sph_cc[dit],a_State.m_dx_sph[dit],d);
				// Vector W_low_boxed(dbx1);
				// W_low.copyTo(W_low_boxed);
				// if (procID() == 0) h5.writePatch({"density","Vx","Vy","Vz", "p","Bx","By","Bz"}, 1, W_low_boxed, "W_low_2O_"+to_string(d));
				if (inputs.Riemann_solver_type == 1) MHD_Riemann_Solvers::Rusanov_Solver(F_f_sph,W_low,W_high,d,gamma);
				if (inputs.Riemann_solver_type == 2) MHD_Riemann_Solvers::Roe8Wave_Solver(F_f_sph,W_low,W_high,d,gamma);
				#if TURB == 1
					MHD_Turbulence::Turb_Flux(F_f_sph, W_low, W_high, d);
				#endif
				if (d==0) MHD_Mapping::Spherical_to_Cartesian(F_f, F_f_sph, a_State.m_x_sph_fc_1[dit]);
				if (d==1) MHD_Mapping::Spherical_to_Cartesian(F_f, F_f_sph, a_State.m_x_sph_fc_2[dit]);
				if (d==2) MHD_Mapping::Spherical_to_Cartesian(F_f, F_f_sph, a_State.m_x_sph_fc_3[dit]);
				// Vector F_f_boxed(dbx1);
				// F_f.copyTo(F_f_boxed);
				// if (procID() == 0) h5.writePatch({"density","Vx","Vy","Vz", "p","Bx","By","Bz"}, 1, F_f_boxed, "F_f_2O_"+to_string(d));
				forallInPlace_p(Scale_with_A_Ff_calc, F_f_scaled, F_f, a_State.m_face_area[dit], d);
				Rhs_d = m_divergence[d](F_f_scaled);
				RhsV += Rhs_d;

				if (!a_State.m_divB_calculated && inputs.takedivBstep == 1){
					forallInPlace_p(Scale_with_A_Bf_calc, B_f_scaled, W_low, W_high, a_State.m_face_area[dit], d);
					Rhs_d_divB = m_divergence[d](B_f_scaled);
					RhsV_divB += Rhs_d_divB;
				}

				#if TURB == 1
					forallInPlace_p(Scale_with_A_Vf_calc, V_f_scaled, W_low, W_high, a_State.m_face_area[dit], d);
					Rhs_d_divV = m_divergence[d](V_f_scaled);
					RhsV_divV += Rhs_d_divV;
				#endif
			}
			forallInPlace_p(Scale_with_V_calc, a_Rhs[dit], RhsV, a_State.m_cell_volume[dit]);
			#if TURB == 1
				forallInPlace_p(Scale_with_V3_calc, divV, RhsV_divV, a_State.m_cell_volume[dit]);
				MHD_Turbulence::Turb_Source(STM,W_sph,divV);
				// if (procID() == 0) h5.writePatch({"divV"}, 1, divV, "divV");
				// if (procID() == 0) h5.writePatch({"density","Vx","Vy","Vz", "p","Bx","By","Bz","Z2","Sigma","Lambda"}, 1, STM, "STM");
				forallInPlace_p(Add_Sources_calc, a_Rhs[dit], STM);
			#endif
			if (inputs.Sun_gravity == 1)
			{
				MHD_Gravity::add_gravity_source(
				        a_Rhs[dit], W_sph, a_State.m_x_sph_cc[dit],
				        inputs.velocity_scale);
			}
			if (inputs.corotating_frame == 1)
			{
				MHD_Rotation::add_rotation_source(
				        a_Rhs[dit], W_cart, a_State.m_x_sph_cc[dit],
				        inputs.velocity_scale, inputs.solar_angular_velocity());
			}
			if (!a_State.m_divB_calculated && inputs.takedivBstep == 1) forallInPlace_p(Powell_Sph_2O,a_State.m_divB[dit],W_cart,RhsV_divB,a_State.m_cell_volume[dit]);		
			// Vector RHS(dbx1);
			// a_State.m_divB[dit].copyTo(RHS);
			// a_Rhs[dit].copyTo(RHS);
			// if (procID() == 0) h5.writePatch({"density","Vx","Vy","Vz", "p","Bx","By","Bz"}, 1, RHS, "RHS_2");
		}
	}

	/**
	 * @brief Advance one step with the MUSCL--Hancock scheme used by MS-FLUKSS.
	 *
	 * The predictor evolves each cell from its own reconstructed left and right
	 * physical fluxes. The corrector shifts those same reconstructed face states
	 * by the primitive-variable predictor, solves the Riemann problems once, and
	 * updates the original conserved state with the resulting midpoint fluxes.
	 */
	void step_spherical_2O_hancock(MHDLevelDataState& a_State,
	                              double a_time,
	                              double a_dt)
	{
		PR_TIME("MHDOp::step_spherical_2O_hancock");

		const double half_dt = 0.5*a_dt;
		const double gamma = a_State.m_gamma;

		static Stencil<double> divergence[DIM];
		static bool initialized = false;
		if (!initialized)
		{
			for (int dir = 0; dir < DIM; dir++)
			{
				divergence[dir] = Stencil<double>::FluxDivergence(dir);
			}
			initialized = true;
		}

		// A ghosted copy of U^n is needed by both reconstruction passes. LevelBoxData
		// copyTo does not copy ghosts, so exchange and physical BC filling are explicit.
		LevelBoxData<double,NUMCOMPS> UOld(a_State.m_dbl, Point::Ones(NGHOST));
		for (auto dit : UOld)
		{
			a_State.m_U[dit].copyTo(UOld[dit]);
		}
		UOld.defineExchange<PolarExchangeCopier>(2,1);
		UOld.exchange();
		MHD_Set_Boundary_Values::Set_Boundary_Values_Spherical_2O(UOld, a_State);

		LevelBoxData<double,NUMCOMPS> UPredictor(a_State.m_dbl, Point::Ones(NGHOST));
		for (auto dit : UPredictor)
		{
			UOld[dit].copyTo(UPredictor[dit]);
		}

		double min_dt_local = 1.0e10*inputs.velocity_scale;

		// Predictor: evolve each cell by dt/2 using the physical flux from that
		// cell's own extrapolated state on each of its faces (no Riemann solve).
		for (auto dit : UOld)
		{
			Box ghostBox = UOld[dit].box();
			Box validBox = a_State.m_U[dit].box();
			Vector predictorRhs(ghostBox);
			Vector fluxRhs(ghostBox);
			Scalar divBRhs(ghostBox);
			predictorRhs.setVal(0.0);
			fluxRhs.setVal(0.0);
			divBRhs.setVal(0.0);

			#if TURB == 1
				Scalar divVRhs(ghostBox);
				divVRhs.setVal(0.0);
			#endif

			MHDOp::Fix_negative_P(UOld[dit], gamma);
			Vector WCart = forall<double,NUMCOMPS>(consToPrim, UOld[dit], gamma);
			Vector WSph(ghostBox);
			MHD_Mapping::Cartesian_to_Spherical(WSph, WCart, a_State.m_x_sph_cc[dit]);
			MHD_Mapping::Correct_V_theta_phi_at_poles(
			        WSph, a_State.m_dx, a_State.m_dy, a_State.m_dz);

			double patch_dt;
			MHD_CFL::Min_dt_calc_func(
			        patch_dt, WSph, validBox,
			        a_State.m_dx, a_State.m_dy, a_State.m_dz, gamma);
			min_dt_local = std::min(min_dt_local, patch_dt);

			for (int dir = 0; dir < DIM; dir++)
			{
				Vector WLow(ghostBox), WHigh(ghostBox);
				MHD_Limiters::MHD_Limiters_minmod(
				        WLow, WHigh, WSph, a_State.m_x_sph_cc[dit],
				        a_State.m_dx_sph[dit], dir);

				Vector FPlusSph, FMinusSph;
				MHD_Riemann_Solvers::Physical_Flux(FPlusSph, WLow, dir, gamma);
				MHD_Riemann_Solvers::Physical_Flux(FMinusSph, WHigh, dir, gamma);
				#if TURB == 1
					MHD_Turbulence::Turb_Flux_Hancock(FPlusSph, WLow, dir);
					MHD_Turbulence::Turb_Flux_Hancock(FMinusSph, WHigh, dir);
				#endif

				Vector FPlus(ghostBox), FMinus(ghostBox);
				if (dir == 0)
				{
					MHD_Mapping::Spherical_to_Cartesian(
					        FPlus, FPlusSph, a_State.m_x_sph_fc_1[dit]);
					MHD_Mapping::Spherical_to_Cartesian(
					        FMinus, FMinusSph, a_State.m_x_sph_fc_1[dit]);
				}
				if (dir == 1)
				{
					MHD_Mapping::Spherical_to_Cartesian(
					        FPlus, FPlusSph, a_State.m_x_sph_fc_2[dit]);
					MHD_Mapping::Spherical_to_Cartesian(
					        FMinus, FMinusSph, a_State.m_x_sph_fc_2[dit]);
				}
				if (dir == 2)
				{
					MHD_Mapping::Spherical_to_Cartesian(
					        FPlus, FPlusSph, a_State.m_x_sph_fc_3[dit]);
					MHD_Mapping::Spherical_to_Cartesian(
					        FMinus, FMinusSph, a_State.m_x_sph_fc_3[dit]);
				}

				Vector FPlusScaled(ghostBox), FMinusScaled(ghostBox);
				forallInPlace_p(
				        Scale_with_A_Ff_calc, FPlusScaled, FPlus,
				        a_State.m_face_area[dit], dir);
				forallInPlace_p(
				        Scale_with_A_Ff_calc, FMinusScaled, FMinus,
				        a_State.m_face_area[dit], dir);

				// WLow at face i+1 belongs to cell i; WHigh at face i belongs
				// to cell i. The same indexing applies to their physical fluxes.
				Vector FPlusAtCell = alias(
				        FPlusScaled, -Point::Basis(dir));
				Vector directionalRhs = FPlusAtCell - FMinusScaled;
				fluxRhs += directionalRhs;

				if (inputs.takedivBstep == 1)
				{
					Scalar BFaceScaled(ghostBox);
					forallInPlace_p(
					        Scale_with_A_Bf_calc, BFaceScaled, WLow, WHigh,
					        a_State.m_face_area[dit], dir);
					Scalar directionalDivB = divergence[dir](BFaceScaled);
					divBRhs += directionalDivB;
				}

				#if TURB == 1
					Scalar VPlusScaled(ghostBox), VMinusScaled(ghostBox);
					forallInPlace_p(
					        Scale_with_A_Vf_calc, VPlusScaled, WLow, WLow,
					        a_State.m_face_area[dit], dir);
					forallInPlace_p(
					        Scale_with_A_Vf_calc, VMinusScaled, WHigh, WHigh,
					        a_State.m_face_area[dit], dir);
					Scalar VPlusAtCell = alias(
					        VPlusScaled, -Point::Basis(dir));
					Scalar directionalDivV = VPlusAtCell - VMinusScaled;
					divVRhs += directionalDivV;
				#endif
			}

			forallInPlace_p(
			        Scale_with_V_calc, predictorRhs, fluxRhs,
			        a_State.m_cell_volume[dit]);

			#if TURB == 1
				Scalar divV(ghostBox);
				Vector turbulenceSource(ghostBox);
				forallInPlace_p(
				        Scale_with_V3_calc, divV, divVRhs,
				        a_State.m_cell_volume[dit]);
				MHD_Turbulence::Turb_Source(turbulenceSource, WSph, divV);
				forallInPlace_p(
				        Add_Sources_calc, predictorRhs, turbulenceSource);
			#endif

			if (inputs.Sun_gravity == 1)
			{
				MHD_Gravity::add_gravity_source(
				        predictorRhs, WSph, a_State.m_x_sph_cc[dit],
				        inputs.velocity_scale);
			}
			if (inputs.corotating_frame == 1)
			{
				MHD_Rotation::add_rotation_source(
				        predictorRhs, WCart, a_State.m_x_sph_cc[dit],
				        inputs.velocity_scale, inputs.solar_angular_velocity());
			}

			if (inputs.takedivBstep == 1)
			{
				Vector powellSource(ghostBox);
				forallInPlace_p(
				        Powell_Sph_2O, powellSource, WCart, divBRhs,
				        a_State.m_cell_volume[dit]);
				forallInPlace_p(Add_Sources_calc, predictorRhs, powellSource);
			}

			Vector predictorIncrement(validBox);
			predictorRhs.copyTo(predictorIncrement, validBox);
			predictorIncrement *= half_dt;
			UPredictor[dit] += predictorIncrement;
			MHDOp::Fix_negative_P(UPredictor[dit], gamma);
		}

		double min_dt = min_dt_local;
		#ifdef PR_MPI
			MPI_Allreduce(
			        &min_dt_local, &min_dt, 1, MPI_DOUBLE, MPI_MIN,
			        MPI_COMM_WORLD);
		#endif
		a_State.m_min_dt = min_dt;

		// The corrector needs a globally consistent midpoint state, including
		// inter-patch, polar, and radial physical ghost cells.
		UPredictor.defineExchange<PolarExchangeCopier>(2,1);
		UPredictor.exchange();
		double saved_time = a_State.m_time;
		a_State.m_time = a_time + half_dt;
		MHD_Set_Boundary_Values::Set_Boundary_Values_Spherical_2O(
		        UPredictor, a_State);
		a_State.m_time = saved_time;

		MHDLevelDataDX finalUpdate;
		finalUpdate.init(a_State);

		// Corrector: update the original reconstruction with W^(n+1/2)-W^n,
		// solve the Riemann problem on those predicted faces, and form the full step.
		for (auto dit : UOld)
		{
			Box ghostBox = UOld[dit].box();
			Vector fluxRhs(ghostBox);
			Vector correctorRhs(ghostBox);
			Scalar divBRhs(ghostBox);
			fluxRhs.setVal(0.0);
			correctorRhs.setVal(0.0);
			divBRhs.setVal(0.0);

			#if TURB == 1
				Scalar divVRhs(ghostBox);
				divVRhs.setVal(0.0);
			#endif

			Vector WOldCart = forall<double,NUMCOMPS>(
			        consToPrim, UOld[dit], gamma);
			Vector WPredictorCart = forall<double,NUMCOMPS>(
			        consToPrim, UPredictor[dit], gamma);
			Vector WOldSph(ghostBox), WPredictorSph(ghostBox);
			MHD_Mapping::Cartesian_to_Spherical(
			        WOldSph, WOldCart, a_State.m_x_sph_cc[dit]);
			MHD_Mapping::Cartesian_to_Spherical(
			        WPredictorSph, WPredictorCart, a_State.m_x_sph_cc[dit]);
			MHD_Mapping::Correct_V_theta_phi_at_poles(
			        WOldSph, a_State.m_dx, a_State.m_dy, a_State.m_dz);
			MHD_Mapping::Correct_V_theta_phi_at_poles(
			        WPredictorSph, a_State.m_dx, a_State.m_dy, a_State.m_dz);
			Vector deltaW = WPredictorSph - WOldSph;

			for (int dir = 0; dir < DIM; dir++)
			{
				Vector WLow(ghostBox), WHigh(ghostBox);
				MHD_Limiters::MHD_Limiters_minmod(
				        WLow, WHigh, WOldSph, a_State.m_x_sph_cc[dit],
				        a_State.m_dx_sph[dit], dir);

				// WLow(face i) is extrapolated from cell i-1, whereas
				// WHigh(face i) is extrapolated from cell i.
				Vector deltaWLow = alias(deltaW, Point::Basis(dir));
				WLow += deltaWLow;
				WHigh += deltaW;

				Vector faceFluxSph(ghostBox);
				if (inputs.Riemann_solver_type == 1)
				{
					MHD_Riemann_Solvers::Rusanov_Solver(
					        faceFluxSph, WLow, WHigh, dir, gamma);
				}
				else if (inputs.Riemann_solver_type == 2)
				{
					MHD_Riemann_Solvers::Roe8Wave_Solver(
					        faceFluxSph, WLow, WHigh, dir, gamma);
				}
				else
				{
					MHD::require(false,
					        "Riemann_solver_type must be 1 (Rusanov) or 2 (Roe 8-wave).");
				}
				#if TURB == 1
					MHD_Turbulence::Turb_Flux(faceFluxSph, WLow, WHigh, dir);
				#endif

				Vector faceFlux(ghostBox), faceFluxScaled(ghostBox);
				if (dir == 0)
				{
					MHD_Mapping::Spherical_to_Cartesian(
					        faceFlux, faceFluxSph, a_State.m_x_sph_fc_1[dit]);
				}
				if (dir == 1)
				{
					MHD_Mapping::Spherical_to_Cartesian(
					        faceFlux, faceFluxSph, a_State.m_x_sph_fc_2[dit]);
				}
				if (dir == 2)
				{
					MHD_Mapping::Spherical_to_Cartesian(
					        faceFlux, faceFluxSph, a_State.m_x_sph_fc_3[dit]);
				}
				forallInPlace_p(
				        Scale_with_A_Ff_calc, faceFluxScaled, faceFlux,
				        a_State.m_face_area[dit], dir);
				Vector directionalRhs = divergence[dir](faceFluxScaled);
				fluxRhs += directionalRhs;

				if (inputs.takedivBstep == 1)
				{
					Scalar BFaceScaled(ghostBox);
					forallInPlace_p(
					        Scale_with_A_Bf_calc, BFaceScaled, WLow, WHigh,
					        a_State.m_face_area[dit], dir);
					Scalar directionalDivB = divergence[dir](BFaceScaled);
					divBRhs += directionalDivB;
				}

				#if TURB == 1
					Scalar VFaceScaled(ghostBox);
					forallInPlace_p(
					        Scale_with_A_Vf_calc, VFaceScaled, WLow, WHigh,
					        a_State.m_face_area[dit], dir);
					Scalar directionalDivV = divergence[dir](VFaceScaled);
					divVRhs += directionalDivV;
				#endif
			}

			forallInPlace_p(
			        Scale_with_V_calc, correctorRhs, fluxRhs,
			        a_State.m_cell_volume[dit]);

			#if TURB == 1
				Scalar divV(ghostBox);
				Vector turbulenceSource(ghostBox);
				forallInPlace_p(
				        Scale_with_V3_calc, divV, divVRhs,
				        a_State.m_cell_volume[dit]);
				MHD_Turbulence::Turb_Source(
				        turbulenceSource, WPredictorSph, divV);
				forallInPlace_p(
				        Add_Sources_calc, correctorRhs, turbulenceSource);
			#endif

			if (inputs.Sun_gravity == 1)
			{
				MHD_Gravity::add_gravity_source(
				        correctorRhs, WPredictorSph,
				        a_State.m_x_sph_cc[dit], inputs.velocity_scale);
			}
			if (inputs.corotating_frame == 1)
			{
				MHD_Rotation::add_rotation_source(
				        correctorRhs, WPredictorCart, a_State.m_x_sph_cc[dit],
				        inputs.velocity_scale, inputs.solar_angular_velocity());
			}

			if (inputs.takedivBstep == 1)
			{
				Vector powellSource(ghostBox);
				forallInPlace_p(
				        Powell_Sph_2O, powellSource, WPredictorCart, divBRhs,
				        a_State.m_cell_volume[dit]);
				forallInPlace_p(Add_Sources_calc, correctorRhs, powellSource);
				powellSource.copyTo(a_State.m_divB[dit]);
			}

			correctorRhs.copyTo(finalUpdate.m_DU[dit]);
		}

		finalUpdate *= a_dt;
		a_State.increment(finalUpdate);
		a_State.m_divB_calculated = true;
		a_State.m_divV_calculated = true;
		a_State.m_min_dt_calculated = true;
	}
}
