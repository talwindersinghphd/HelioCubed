#include "MHD_Error.H"
#include "Proto.H"
#include "MHD_Set_Boundary_Values.H"
#include "Proto_Timer.H"
// #include "Proto_WriteBoxData.H"
#include "Proto_LevelBoxData.H"
#include "Proto_ProblemDomain.H"
#include "MHDOp.H"
#include "MHD_Mapping.H"
#include "MHD_Initialize.H"
#include "MHD_Input_Parsing.H"
#include "MHD_Output_Writer.H"
#include "MHD_Constants.H"
#include "PolarExchangeCopier.H"
typedef BoxData<double,1,HOST> Scalar;
typedef BoxData<double,NUMCOMPS,HOST> Vector;

extern Parsefrominputs inputs;
/// @brief MHD_Set_Boundary_Values namespace
namespace MHD_Set_Boundary_Values {

	PROTO_KERNEL_START
	void dot_pro_calcFF(State& a_dot_pro,
	                    const Var<double,1>& a_d_perp_N_s,
	                    const State& a_d_perp_F)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			a_dot_pro(i) = (a_d_perp_N_s(0)*a_d_perp_F(i));
		}
	}
	PROTO_KERNEL_END(dot_pro_calcFF, dot_pro_calcF)


	PROTO_KERNEL_START
	void iotaFuncF(Point           & a_p,
	               V               & a_X,
	               const double a_dx,
	               const double a_dy,
	               const double a_dz)
	{
		for (int ii = 0; ii < DIM; ii++)
		{
			double dxd;
			if (ii == 0) dxd = a_dx;
			if (ii == 1) dxd = a_dy;
			if (ii == 2) dxd = a_dz;
			a_X(ii) = a_p[ii]*dxd + 0.5*dxd;
		}
	}
	PROTO_KERNEL_END(iotaFuncF,iotaFunc)

	PROTO_KERNEL_START
	void scale_with_r_calcF(const Point& a_pt,
							Var<double,NUMCOMPS>& a_U_scaled,
							Var<double,NUMCOMPS>& a_W,
							Var<double,DIM>& a_x_sph,
							const double a_gamma,
							const double a_scaling_factor)
	{	

		

		double r_BC = inputs.r_in*c_AU; //in cm
		double rad = a_x_sph(0);
		double rho = a_W(0)*1.67262192e-24*pow(inputs.r_in*c_AU/rad,2.0);
		double u = a_W(1);
		double v = a_W(2);
		double w = a_W(3);
		double p = a_W(4)*1.0e-12*pow(inputs.r_in*c_AU/rad,2.0*a_gamma); // From picodyne to dyne 
		double Bx = a_W(5)*1.0e-6*pow(inputs.r_in*c_AU/rad,2.0); // MicroGauss to Gauss
		double By = a_W(6)*1.0e-6*pow(inputs.r_in*c_AU/rad,1.0);
		double Bz = a_W(7)*1.0e-6*pow(inputs.r_in*c_AU/rad,1.0);

		// Boundary files contain inertial velocities. Convert at the actual
		// ghost-cell radius before building momentum and kinetic energy.
		if (inputs.corotating_frame == 1)
		{
			w -= inputs.solar_angular_velocity()*rad*sin(a_x_sph(1));
		}

		double e = p/(a_gamma-1.0) + rho*(u*u+v*v+w*w)/2.0 + (Bx*Bx+By*By+Bz*Bz)/8.0/c_PI;

		a_U_scaled(0) = rho; //rho
		a_U_scaled(1) = rho*u; //Momentum-x
		a_U_scaled(2) = rho*v; //Momentum-y
		a_U_scaled(3) = rho*w; //Momentum-z
		a_U_scaled(4) = e; //Energy
		a_U_scaled(5) = Bx; //Bx
		a_U_scaled(6) = By; //By
		a_U_scaled(7) = Bz; //Bz
		#if TURB == 1
			a_U_scaled(iRHOZ2) = rho*(inputs.Sun_Z2)*exp(8.*a_scaling_factor)/exp(8.0); //rho*Z^2
			a_U_scaled(iRHOZ2SIGMA) = a_U_scaled(iRHOZ2)*inputs.Sun_SigmaC; //rho*Z^2*sigma
			a_U_scaled(iRHOLAMBDA) = rho*inputs.Sun_Lambda; //rho*lambda
		#endif


	}
	PROTO_KERNEL_END(scale_with_r_calcF, scale_with_r_calc)

	
	void Set_Boundary_Values_Spherical_2O(LevelBoxData<double,NUMCOMPS>& a_U,
	                         			  MHDLevelDataState& a_State)
	{

		double a_dx = a_State.m_dx;
		double a_dy = a_State.m_dy;
		double a_dz = a_State.m_dz;
		double a_gamma = a_State.m_gamma;
		double a_time = a_State.m_time;
		for (auto dit : a_U){	
			Box dbx0 = a_U[dit].box();
			//Filling ghost cells for low side of dir == 0 here. This will be the inner boundary in r direction once we map to polar, spherical, or cubed sphere grids.
			
			if (dbx0.low()[0] < a_State.m_probDom.box().low()[0]) {
				if (inputs.sph_inner_BC_hdf5 == 1){

					Point source_low = Point(a_State.m_probDom.box().low()[0],a_U[dit].box().low()[1],a_U[dit].box().low()[2]);
					Point source_high = Point(a_State.m_probDom.box().low()[0],a_U[dit].box().high()[1],a_U[dit].box().high()[2]);
					
					Box sourceBox(source_low,source_high);
					Box sourceBox1 = sourceBox.grow(1);
					Point ghost_low = dbx0.low();

					Point ghost_high = Point(a_State.m_probDom.box().low()[0]-1, dbx0.high()[1], dbx0.high()[2]);
					
					Box BoundBox(ghost_low,ghost_high);
					Vector a_U_ghost(BoundBox), a_U_sph(BoundBox), a_U_sph_scaled_r(BoundBox);

					for (int i = 1; i <= NGHOST; i++ ) {
						a_State.m_BC[dit].copyTo(a_U_sph,sourceBox,Point::Basis(0)*(-i));// Using shifting option of copyTo
					}
					BoxData<double,DIM> x_sph(BoundBox);
					MHD_Mapping::get_sph_coords_cc(x_sph,BoundBox,a_dx, a_dy, a_dz);
					double scaling_factor = 0.0;
					#if TURB == 1
					double time_days = a_time/inputs.velocity_scale/24./60./60.;
					scaling_factor = min(time_days,inputs.Scaling_time)/inputs.Scaling_time;
					// round of scaling_factor to 1 decimal places
					scaling_factor = round(scaling_factor*10.0)/10.0;
					#endif
					forallInPlace_p(scale_with_r_calc, BoundBox, a_U_sph_scaled_r, a_U_sph, x_sph, a_gamma, scaling_factor);
					MHD_Mapping::Spherical_to_Cartesian(a_U_ghost, a_U_sph_scaled_r, x_sph);
					MHDOp::DimToNonDimcalc(a_U_ghost);
					a_U_ghost.copyTo(a_U[dit],BoundBox);
				} else {
					Point ghost_low = dbx0.low();
					Point ghost_high = Point(a_State.m_probDom.box().low()[0]-1, dbx0.high()[1], dbx0.high()[2]);
					Box BoundBox(ghost_low,ghost_high);
					Vector a_U_sph(BoundBox), a_U_ghost(BoundBox);
					BoxData<double,DIM> x_sph(BoundBox);
					MHD_Mapping::get_sph_coords_cc(x_sph,BoundBox,a_dx, a_dy, a_dz);
					MHD_Initialize::InitializeState_Spherical_2O(a_U_sph, x_sph, a_gamma);
					MHD_Mapping::Spherical_to_Cartesian(a_U_ghost, a_U_sph, x_sph);
					a_U_ghost.copyTo(a_U[dit],BoundBox);
				}
			}
			

			//Filling ghost cells for high side of dir == 0 here, to make an open boundary. Ghost cells are filled by outermost cells of problem domain. This will be the outer boundary in r direction once we map to polar, spherical, or cubed sphere grids.
			if (dbx0.high()[0] > a_State.m_probDom.box().high()[0]) {

				Point source_low = Point(a_State.m_probDom.box().high()[0],a_U[dit].box().low()[1],a_U[dit].box().low()[2]);
				Point source_high = Point(a_State.m_probDom.box().high()[0],a_U[dit].box().high()[1],a_U[dit].box().high()[2]);

				Box sourceBox(source_low,source_high);
				Box sourceBox1 = sourceBox.grow(1);

				Vector a_U_exterp(sourceBox1);

				Stencil<double> m_exterp_f_2nd;

				Point ghost_low = Point(a_State.m_probDom.box().high()[0]+1, dbx0.low()[1], dbx0.low()[2]);

				Point ghost_high = dbx0.high();
				Box BoundBox(ghost_low,ghost_high);
				// Box BoundBox1 = BoundBox.grow(1);
				Box BoundBox1 = BoundBox.grow(0);
				Vector a_U_ghost(BoundBox);

				for (int i = 1; i <= NGHOST; i++ ) {
					//Using outermost 2 layers to extrapolate to rest.
					// m_exterp_f_2nd = (i+1.0)*Shift(Point::Zeros()) - (i*1.0)*Shift(-Point::Basis(0)); 
					// No extrapolation
					m_exterp_f_2nd = (1.0)*Shift(Point::Zeros()); 
					a_U_exterp = m_exterp_f_2nd(a_U[dit]);
					a_U_exterp.copyTo(a_U_ghost,sourceBox,Point::Basis(0)*(i));// Using shifting option of copyTo
				}
				MHDOp::Fix_negative_P(a_U_ghost,a_gamma); // Extrapolation can make -ve P in ghost cells.
				a_U_ghost.copyTo(a_U[dit],BoundBox);
			}

			// MHD_Output_Writer::WriteBoxData_array_nocoord(a_U[dit], a_dx, a_dy, a_dz, "STATE");
		}

	}


	PROTO_KERNEL_START
	void InterpolateFramesF(const Point& a_pt,
						Var<double,NUMCOMPS>& a_BC_data_interpolated,
						Var<double,NUMCOMPS>& a_BC_data_lo,
						Var<double,NUMCOMPS>& a_BC_data_hi,
	                   	double a_fraction)
	{
		for (int i=0; i<NUMCOMPS; i++){
			a_BC_data_interpolated(i) = a_fraction*a_BC_data_hi(i) + (1.0-a_fraction)*a_BC_data_lo(i);
		}

	}
	PROTO_KERNEL_END(InterpolateFramesF, InterpolateFrames)



	void interpolate_h5_BC(MHDLevelDataState& state,
						const std::vector<BoxData<double,NUMCOMPS>>& BC_data,
						const double time)
	{
		MHD::require(!BC_data.empty(),
				"MHD_Set_Boundary_Values::interpolate_h5_BC | Error: no BC frames available");

		double h5_cadence = inputs.BC_cadence; // hrs
		int low_frame_now = 0;
		if (BC_data.size() > 1){
			MHD::require(h5_cadence > 0.0,
					"MHD_Set_Boundary_Values::interpolate_h5_BC | Error: BC_cadence must be positive");
			low_frame_now = static_cast<int>(time/(h5_cadence*3600.0));
		}

		if (inputs.corotating_frame == 1)
		{
			// Maps are already in the rotating frame. Keep their longitudes fixed,
			// but retain time interpolation and hold the last available frame.
			const int low_frame = std::min(
			        low_frame_now, static_cast<int>(BC_data.size())-1);
			const int next_frame = std::min(
			        low_frame+1, static_cast<int>(BC_data.size())-1);
			const double fraction = (low_frame == next_frame) ? 0.0 :
			        time/(h5_cadence*3600.0)-low_frame;
			for (auto dit : state.m_U)
			{
				const Box boundary_box = state.m_BC[dit].box() &
				        BC_data[low_frame].box() & BC_data[next_frame].box();
				if (boundary_box.empty()) continue;
				forallInPlace_p(
				        InterpolateFrames, boundary_box, state.m_BC[dit],
				        BC_data[low_frame], BC_data[next_frame], fraction);
			}
		}
		// In the inertial frame, rotate a single/final map continuously.
		else if ((BC_data.size() == 1) ||
			(low_frame_now >= static_cast<int>(BC_data.size())-1)){
			int frame_to_rotate = (BC_data.size() == 1) ? 0 : static_cast<int>(BC_data.size())-1;
			BoxData<double, NUMCOMPS> BC_data_rotated;
			const double carr_rot_time = inputs.solar_rotation_period_seconds();
			double angle_to_rotate = fmod(360*time/carr_rot_time,360);
			int cells_to_rotate = angle_to_rotate/(360.0/inputs.domainSizez);
			double needed_fraction = angle_to_rotate/(360.0/inputs.domainSizez) - cells_to_rotate;
			cells_to_rotate = cells_to_rotate % inputs.domainSizez;
			cells_to_rotate = inputs.domainSizez - cells_to_rotate;
			static Stencil<double> m_right_shift;
			m_right_shift = (1.0-needed_fraction)*Shift(Point::Zeros()) + (needed_fraction)*Shift(-Point::Basis(2));
			Box dbx0 = BC_data[frame_to_rotate].box();
			for (auto dit : state.m_U)
			{	
				BC_data[frame_to_rotate].copyTo(state.m_BC[ dit],dbx0,Point::Basis(2)*(-cells_to_rotate));
				BC_data_rotated = m_right_shift(state.m_BC[ dit]);
				BC_data_rotated.copyTo(state.m_BC[ dit]);
			}
		} else {
			LevelBoxData<double,NUMCOMPS> BC1;
			BC1.define(state.m_dbl,Point::Zero());
			LevelBoxData<double,NUMCOMPS> BC2;
			BC2.define(state.m_dbl,Point::Zero());
			
			
			double time_temp = time+0*h5_cadence*3600;

			int low_frame = time_temp/(h5_cadence*3600);
			int next_frame = low_frame+1;
			MHD::require((low_frame >= 0) &&
					(next_frame < static_cast<int>(BC_data.size())),
					"MHD_Set_Boundary_Values::interpolate_h5_BC | Error: BC frames %i and %i not available (num_frames = %i)",
					low_frame, next_frame, static_cast<int>(BC_data.size()));

			double needed_fraction = (time_temp - (low_frame*h5_cadence*3600))/(h5_cadence*3600);
			
			BoxData<double, NUMCOMPS> BC_data_rotated1;
			BoxData<double, NUMCOMPS> BC_data_rotated2;
			const double carr_rot_time = inputs.solar_rotation_period_seconds();

			double effective_time1 = 360*(time_temp+low_frame*h5_cadence*3600)/carr_rot_time;
			double effective_time2 = 360*(time_temp+next_frame*h5_cadence*3600)/carr_rot_time;

			if (inputs.BC_frame_rotate == 0){
				effective_time1 = 360*(time_temp)/carr_rot_time;
				effective_time2 = 360*(time_temp)/carr_rot_time;
			}

			double angle_to_rotate1 = fmod(effective_time1,360);
			double angle_to_rotate2 = fmod(effective_time2,360);
			int cells_to_rotate1 = angle_to_rotate1/(360/inputs.domainSizez);
			int cells_to_rotate2 = angle_to_rotate2/(360/inputs.domainSizez);
			double needed_fraction1 = angle_to_rotate1/(360/inputs.domainSizez) - cells_to_rotate1;
			double needed_fraction2 = angle_to_rotate2/(360/inputs.domainSizez) - cells_to_rotate2;
			cells_to_rotate1 = cells_to_rotate1 % inputs.domainSizez;
			cells_to_rotate2 = cells_to_rotate2 % inputs.domainSizez;
			cells_to_rotate1 = inputs.domainSizez - cells_to_rotate1;
			cells_to_rotate2 = inputs.domainSizez - cells_to_rotate2;

			// if(procID()==0) cout << " low_frame = " << low_frame << " needed_fraction = " << needed_fraction << " cells_to_rotate1 " << cells_to_rotate1 << " cells_to_rotate2 " << cells_to_rotate2 << endl;

			static Stencil<double> m_right_shift1;
			static Stencil<double> m_right_shift2;
			m_right_shift1 = (1.0-needed_fraction1)*Shift(Point::Zeros()) + (needed_fraction1)*Shift(-Point::Basis(2));
			m_right_shift2 = (1.0-needed_fraction2)*Shift(Point::Zeros()) + (needed_fraction2)*Shift(-Point::Basis(2));
			Box dbx0 = BC_data[low_frame].box();
			for (auto dit : state.m_U)
			{	
				BC_data[low_frame].copyTo(state.m_BC[ dit],dbx0,Point::Basis(2)*(-cells_to_rotate1));
				BC1[ dit] = m_right_shift1(state.m_BC[ dit]);
				BC_data[next_frame].copyTo(state.m_BC[ dit],dbx0,Point::Basis(2)*(-cells_to_rotate2));
				BC2[ dit] = m_right_shift2(state.m_BC[ dit]);

				forallInPlace_p(InterpolateFrames, state.m_BC[ dit], BC1[ dit], BC2[ dit], needed_fraction);

			}
		}

		state.m_BC.defineExchange<PolarExchangeCopier>(2,1);
		state.m_BC.exchange(); 
		

	}

}
