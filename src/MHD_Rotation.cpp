#include "MHD_Rotation.H"

namespace MHD_Rotation
{
	PROTO_KERNEL_START
	void AddRotationSourceF(
	        const Point& a_pt,
	        State& a_Rhs,
	        const State& a_W_cart,
	        const V& a_x_sph,
	        double a_scaled_omega)
	{
		const double cylindrical_radius = a_x_sph(0)*sin(a_x_sph(1));
		const double x = cylindrical_radius*cos(a_x_sph(2));
		const double y = cylindrical_radius*sin(a_x_sph(2));
		const double rho = a_W_cart(iRHO);
		const double vx = a_W_cart(iVX);
		const double vy = a_W_cart(iVY);
		const double omega_sq = a_scaled_omega*a_scaled_omega;

		// rho [(Omega x r) x Omega + 2 (v x Omega)].
		// Lengths remain in cm, while the solver advances t' = velocity_scale*t.
		a_Rhs(iMOMX) += rho*(omega_sq*x + 2.0*a_scaled_omega*vy);
		a_Rhs(iMOMY) += rho*(omega_sq*y - 2.0*a_scaled_omega*vx);

		// v . F_c: Coriolis does no work. Only centrifugal work contributes.
		a_Rhs(iE) += rho*omega_sq*(x*vx + y*vy);
	}
	PROTO_KERNEL_END(AddRotationSourceF, AddRotationSource)

	void add_rotation_source(
	        BoxData<double,NUMCOMPS>& a_Rhs,
	        const BoxData<double,NUMCOMPS>& a_W_cart,
	        const BoxData<double,DIM>& a_x_sph,
	        double a_velocity_scale,
	        double a_rotation_rate)
	{
		const double scaled_omega = a_rotation_rate/a_velocity_scale;
		forallInPlace_p(
		        AddRotationSource, a_Rhs, a_W_cart, a_x_sph,
		        scaled_omega);
	}
}
