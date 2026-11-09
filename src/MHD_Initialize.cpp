#include "Proto.H"
#include "MHD_Mapping.H"
#include "MHDOp.H"
#include "MHD_Initialize.H"
#include "MHD_Output_Writer.H"
#include "MHD_Input_Parsing.H"
#include "MHD_Constants.H"
extern Parsefrominputs inputs;

typedef BoxData<double,1,HOST> Scalar;
typedef BoxData<double,NUMCOMPS,HOST> Vector;
/// @brief MHD_Initialize namespace
namespace MHD_Initialize {

	PROTO_KERNEL_START
	void InitializeStateSph2OF(const Point& a_pt,
								  State& a_U,
	                              V& a_x_sph,
	                              const double a_gamma)
	{
		// cout << a_pt[1] << endl;

		double gamma = a_gamma;
		double rho = 0.0;
		double p = 0.0;
		double u = 0.0;
		double v = 0.0;
		double w = 0.0;
		double Bx = 0.0;
		double By = 0.0;
		double Bz = 0.0;
		double rad = a_x_sph(0);
		double theta = a_x_sph(1);
		double phi = a_x_sph(2);
		
		//////Modifying parameters for radially out flow in spherical grid and magnetic field///////
		rho = 700*c_MP*pow(inputs.r_in*c_AU/rad,2.0); // rho at 21.5 c_SR is about 700/cm3
		p = 1.0e-7*pow(inputs.r_in*c_AU/rad,2.0*a_gamma); // p near 21.5 c_SR is about 1e-7 dyne/cm2
		if (inputs.initialize_in_spherical_coords == 1){
			u = 500.0*1e5;  // v at 21.5 c_SR is about 500 km/s
			v = 0.0;
			w = 0.0; 
			Bx = -0.005*(atan(12*(theta-c_PI/2))/atan(12*(100-c_PI/2)))*pow(inputs.r_in*c_AU/rad,2.0); // Br at 21.5 c_SR is about 0.005 G
		} else {
			u = 5.0*sin(theta)*cos(phi);
			v = 5.0*sin(theta)*sin(phi);
			w = 5.0*cos(theta);
		}
		
		// Compute spherical pulse initial data.
		// double x_cart = rad*sin(theta)*cos(phi);
		// double y_cart = rad*sin(theta)*sin(phi);
		// double z_cart = rad*cos(theta);
		// double p0 = 1.0e-7; // p near 21.5 c_SR is about 1e-7 dyne/cm2
		// double rho0 = 700*c_MP; // rho at 21.5 c_SR is about 700/cm3
		// double u0 = 500.0*1e5; // v at 21.5 c_SR is about 500 km/s
		// double eps = 0.1;
		// double amplitude;
		// double arg = sqrt((x_cart+1.0) * (x_cart+1.0) + y_cart * y_cart + z_cart * z_cart);
		// if (abs(arg) < .25)
		// {
		//   amplitude = eps * pow(cos(2 * M_PI * arg), 6);
		// }
		// else
		// {
		//   amplitude = 0.;
		// }
		// arg = sqrt((x_cart) * (x_cart) + (y_cart+1.0) * (y_cart+1.0) + z_cart * z_cart);
		// if (abs(arg) < .25)
		// {
		//   amplitude += eps * pow(cos(2 * M_PI * arg), 6);
		// }
		// else
		// {
		//   amplitude += 0.;
		// }
		// rho = rho0 + amplitude * rho0;
		// p = p0 * pow(rho / rho0, a_gamma);

		// The analytic state specifies inertial velocities, as do the BC files.
		// This kernel also supplies the analytic inner-boundary ghost cells.
		if (inputs.corotating_frame == 1)
		{
			w -= inputs.solar_angular_velocity()*rad*sin(theta);
		}

		double e = p/(gamma-1.0) + rho*(u*u+v*v+w*w)/2.0 + (Bx*Bx+By*By+Bz*Bz)/8.0/c_PI;

#if TURB == 1
		a_U(iRHOZ2) = rho*inputs.Sun_Z2/exp(8.0); //rho*Z^2
		a_U(iRHOZ2SIGMA) = a_U(iRHOZ2)*inputs.Sun_SigmaC; //rho*Z^2*sigma
		a_U(iRHOLAMBDA) = rho*inputs.Sun_Lambda; //rho*lambda
		e += 0.5*a_U(iRHOZ2);
#endif

		a_U(0) = rho; //rho
		a_U(1) = rho*u; //Momentum-x
		a_U(2) = rho*v; //Momentum-y
		a_U(3) = rho*w; //Momentum-z
		a_U(4) = e; //Energy
		a_U(5) = Bx; //Bx
		a_U(6) = By; //By
		a_U(7) = Bz; //Bz

	}
	PROTO_KERNEL_END(InitializeStateSph2OF, InitializeStateSph2O)


	void InitializeState_Spherical_2O(BoxData<double,NUMCOMPS>& a_U,
	                         const BoxData<double,DIM>& a_x,
	                         const double a_gamma)
	{	
		double gamma = a_gamma;
		forallInPlace_p(InitializeStateSph2O,a_U,a_x,a_gamma);
		MHDOp::DimToNonDimcalc(a_U);
	}

	void initializeState_Spherical_2O(MHDLevelDataState& a_State)
	{
		double a_dx = a_State.m_dx;
		double a_dy = a_State.m_dy;
		double a_dz = a_State.m_dz;
		double a_gamma = a_State.m_gamma;
		for (auto dit : a_State.m_U){
			Box dbx0 = a_State.m_U[dit].box();
			Box dbx = dbx0.grow(NGHOST);
			Box dbx1 = dbx.grow(1);
			BoxData<double,NUMCOMPS> UBig_sph(dbx1);
			BoxData<double,DIM> x_sph(dbx1);
			MHD_Mapping::get_sph_coords_cc(x_sph,dbx1,a_dx, a_dy, a_dz);
			forallInPlace_p(InitializeStateSph2O,UBig_sph,x_sph,a_gamma);
			MHD_Mapping::Spherical_to_Cartesian(a_State.m_U[dit],UBig_sph,x_sph);
			MHDOp::DimToNonDimcalc(a_State.m_U[dit]);
		}
	}

}
