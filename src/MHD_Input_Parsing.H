#pragma once

#include "MHD_Error.H"
#include "MHD_Constants.H"

#include "Proto.H"
#include "InputParser.H"
#include <cmath>

using namespace Proto;
/// @brief Parsefrominputs class
class Parsefrominputs {
  public:
    double tstop;
	double CFL;
	double gamma;
	int domainSizex;
	int domainSizey;
	int domainSizez;
	int maxStep;
	int restartStep;
	int outputInterval;
	int CheckpointInterval;
	int MaxCheckpointFiles;
    int BoxSize;
    int sph_inner_BC_hdf5; // 1 is h5 file is used to provide inner BCs
    int limiter_apply;
    int takedivBstep;
    int Sun_gravity = 1; // 1: Include solar gravity as an explicit source term
    int corotating_frame = 0; // 0: Inertial frame, 1: Frame rotating about +z
    double solar_rotation_period = 25.38; // Sidereal solar rotation period in days
    int timeIntegratorType; // 1: Euler, 2: RK2, 3: Hancock, 4: RK4
    int Riemann_solver_type;  //1: Rusanov Solver, 2: Roe Solver
    double entropy_fix_coeff;
    int initialize_in_spherical_coords;
    int output_in_spherical_coords;
    double r_in;
    double r_out;
    double C_rad; // A constant in exponential dr in spherical grid.
	double BC_start_time; // In decimal year
	double BC_cadence; // In hours
	int BC_frame_rotate; // 0: No rotation (SWiG), 1: Rotating frames (Tae WSA)
	double probe_cadence; // In seconds
	string BC_file;
	string Data_file_Prefix;
	string Checkpoint_file_Prefix;
	string Probe_trajectory_file;
	string Probe_data_file;
	double CME_Enter_Time;
	double CME_get_checkpoint_before; // Get a checkpoint file this much time in seconds before CME insertion.
	double CME_r1, CME_r0, CME_halfangle, CME_pol_flux_control, CME_FRiED_n, CME_lat, CME_lon, CME_tilt, CME_Vandas_R0, CME_apex_speed, CME_energy_control, CME_density, CME_helicity_sign, CME_a1, CME_shift;
	//MHD equations can be written in dimensionless variables. We use scaling for this.
	double density_scale, velocity_scale;
	double Sun_Z2 = 3.0e13;
	double Sun_SigmaC = 0.6;
	double Sun_Lambda = 1.2e9;
	double Scaling_time = 20.0; // Days used to ramp Z2 to Sun_Z2 at the inner boundary.
	double SigmaD = -1.0/3.0;
	double alphaTM = 0.8;
	double betaTM = 0.4;
	double smallTM = 1.0e-8;
  public:

	double solar_rotation_period_seconds() const
	{
		return solar_rotation_period*24.0*60.0*60.0;
	}

	double solar_angular_velocity() const
	{
		return 2.0*c_PI/solar_rotation_period_seconds();
	}

	string removeSpaces(string str)
	{
		str.erase(remove(str.begin(), str.end(), ' '), str.end());
		return str;
	}
    
	void parsenow(int argc, char* argv[]){
		MHD::require(argc >= 2, "Error: No input file name provided.");
        std::string inputFileName = argv[1];
		std::ifstream myfile(inputFileName);
		MHD::require(myfile.is_open(), "Error: Unable to open inputs file. Is the file name correct?");

        std::cout << "Input file provided: " << inputFileName << std::endl;

		InputArgs args;
		args.add("tstop",  tstop);
		args.add("CFL",  CFL);
		args.add("gamma",  gamma);
		args.add("domainSizex",  domainSizex);
		args.add("domainSizey",  domainSizey);
		args.add("domainSizez",  domainSizez);
		args.add("maxStep",  maxStep);
		args.add("restartStep",  restartStep);
		args.add("outputInterval",  outputInterval);
		args.add("CheckpointInterval",  CheckpointInterval);
		args.add("MaxCheckpointFiles",  MaxCheckpointFiles);
		args.add("BoxSize",  BoxSize);
		args.add("sph_inner_BC_hdf5",  sph_inner_BC_hdf5);
		args.add("limiter_apply",  limiter_apply);
		args.add("takedivBstep",  takedivBstep);
		args.add("Sun_gravity",  Sun_gravity);
		args.add("corotating_frame",  corotating_frame);
		args.add("solar_rotation_period",  solar_rotation_period);
		args.add("timeIntegratorType",  timeIntegratorType);
		args.add("Riemann_solver_type",  Riemann_solver_type);
		args.add("entropy_fix_coeff",  entropy_fix_coeff);
		args.add("initialize_in_spherical_coords",  initialize_in_spherical_coords);
		args.add("output_in_spherical_coords",  output_in_spherical_coords);
		args.add("r_in",  r_in);
		args.add("r_out",  r_out);
		args.add("C_rad",  C_rad);
		args.add("BC_start_time",  BC_start_time);
		args.add("BC_cadence",  BC_cadence);
		args.add("BC_frame_rotate",  BC_frame_rotate);
		args.add("probe_cadence",  probe_cadence);
		args.add("BC_file",  BC_file);
		args.add("Data_file_Prefix",  Data_file_Prefix);
		args.add("Checkpoint_file_Prefix",  Checkpoint_file_Prefix);
		args.add("Probe_trajectory_file",  Probe_trajectory_file);
		args.add("Probe_data_file",  Probe_data_file);
		args.add("CME_Enter_Time",  CME_Enter_Time);
		args.add("CME_get_checkpoint_before",  CME_get_checkpoint_before);
		args.add("CME_r1",  CME_r1);
		args.add("CME_r0",  CME_r0);
		args.add("CME_halfangle",  CME_halfangle);
		args.add("CME_pol_flux_control",  CME_pol_flux_control);
		args.add("CME_FRiED_n",  CME_FRiED_n);
		args.add("CME_lat",  CME_lat);
		args.add("CME_lon",  CME_lon);
		args.add("CME_tilt",  CME_tilt);
		args.add("CME_Vandas_R0",  CME_Vandas_R0);
		args.add("CME_apex_speed",  CME_apex_speed);
		args.add("CME_energy_control",  CME_energy_control);
		args.add("CME_density",  CME_density);
		args.add("CME_helicity_sign",  CME_helicity_sign);
		args.add("CME_a1",  CME_a1);
		args.add("CME_shift",  CME_shift);
		args.add("density_scale",  density_scale);
		args.add("velocity_scale",  velocity_scale);
		args.add("Sun_Z2",  Sun_Z2);
		args.add("Sun_SigmaC",  Sun_SigmaC);
		args.add("Sun_Lambda",  Sun_Lambda);
		args.add("SigmaD",  SigmaD);
		args.add("alphaTM",  alphaTM);
		args.add("betaTM",  betaTM);
		args.add("smallTM",  smallTM);
		args.add("Scaling_time",  Scaling_time);
		args.parse(argc,argv);
		// These restrictions were previously enforced only by Proto assertions.
		// Keep them active before constructing layouts in a release build.
		MHD::require(domainSizex > 0 && domainSizey > 0 && domainSizez > 0 && BoxSize > 0,
				"Domain sizes and BoxSize must be positive.");
		MHD::require(domainSizex % BoxSize == 0 && domainSizey % BoxSize == 0 && domainSizez % BoxSize == 0,
				"Each domain size must be divisible by BoxSize.");
		MHD::require(domainSizez % 2 == 0 && (domainSizez/BoxSize) % 2 == 0,
				"Polar exchange requires an even number of azimuthal cells and patches.");
		MHD::require(timeIntegratorType >= 1 && timeIntegratorType <= 4,
				"timeIntegratorType must be 1 (Euler), 2 (RK2), 3 (Hancock), or 4 (RK4).");
		MHD::require(Riemann_solver_type == 1 || Riemann_solver_type == 2,
				"Riemann_solver_type must be 1 (Rusanov) or 2 (Roe).");
		MHD::require(corotating_frame == 0 || corotating_frame == 1,
				"corotating_frame must be 0 (inertial) or 1 (co-rotating).");
		MHD::require(std::isfinite(solar_rotation_period) && solar_rotation_period > 0.0,
				"solar_rotation_period must be finite and positive (days).");
		MHD::require(corotating_frame == 0 ||
				(std::isfinite(velocity_scale) && velocity_scale > 0.0),
				"Co-rotating runs require a finite, positive velocity_scale.");
		// args.print(); //It prints, but not in the order I want.

		int pid = procID();

		if (pid == 0){
			cout << "Running with the following inputs (these can be changed in the file 'inputs')" << endl;
			cout << setw(75) << setfill('-') << "-" << endl;
			cout << "tstop: " << tstop << endl;
			cout << "CFL: " << CFL << endl;
			cout << "gamma: " << gamma << endl;
			cout << "domainSizex: "  << domainSizex << endl;
			cout << "domainSizey: "  << domainSizey << endl;
			cout << "domainSizez: "  << domainSizez << endl;
			cout << "maxStep: "    << maxStep << endl;
			cout << "restartStep: "    << restartStep << endl;
			cout << "outputInterval: "    << outputInterval << endl;	
			cout << "CheckpointInterval: "    << CheckpointInterval << endl;	
			cout << "MaxCheckpointFiles: "    << MaxCheckpointFiles << endl;	
			cout << "BoxSize: "    << BoxSize << endl;
			cout << "sph_inner_BC_hdf5: "    << sph_inner_BC_hdf5 << endl;
			cout << "limiter_apply: "    << limiter_apply << endl;
			cout << "takedivBstep: "    << takedivBstep << endl;
			cout << "Sun_gravity: "    << Sun_gravity << endl;
			cout << "corotating_frame: "    << corotating_frame << endl;
			cout << "solar_rotation_period: "    << solar_rotation_period << endl;
			cout << "timeIntegratorType: "    << timeIntegratorType << endl;
			cout << "Riemann_solver_type: "    << Riemann_solver_type << endl;
			cout << "entropy_fix_coeff: "    << entropy_fix_coeff << endl;
			cout << "initialize_in_spherical_coords: "    << initialize_in_spherical_coords << endl;
			cout << "output_in_spherical_coords: "    << output_in_spherical_coords << endl;
			cout << "r_in: "    << r_in << endl;
			cout << "r_out: "    << r_out << endl;
			cout << "C_rad: "    << C_rad << endl;
			cout << "BC_start_time: "    << BC_start_time << endl;
			cout << "BC_cadence: "    << BC_cadence << endl;
			cout << "BC_frame_rotate: "    << BC_frame_rotate << endl;
			cout << "probe_cadence: "    << probe_cadence << endl;
			cout << "BC_file: "    << BC_file << endl;
			cout << "Data_file_Prefix: "    << Data_file_Prefix << endl;
			cout << "Checkpoint_file_Prefix: "    << Checkpoint_file_Prefix << endl;
			cout << "Probe_trajectory_file: "    << Probe_trajectory_file << endl;
			cout << "Probe_data_file: "    << Probe_data_file << endl;
			cout << "CME_Enter_Time: "    << CME_Enter_Time << endl;
			cout << "CME_get_checkpoint_before: "    << CME_get_checkpoint_before << endl;
			cout << "CME_r1: "    << CME_r1 << endl;
			cout << "CME_r0: "    << CME_r0 << endl;
			cout << "CME_halfangle: "    << CME_halfangle << endl;
			cout << "CME_pol_flux_control: "    << CME_pol_flux_control << endl;
			cout << "CME_FRiED_n: "    << CME_FRiED_n << endl;
			cout << "CME_lat: "    << CME_lat << endl;
			cout << "CME_lon: "    << CME_lon << endl;
			cout << "CME_tilt: "    << CME_tilt << endl;
			cout << "CME_Vandas_R0: "    << CME_Vandas_R0 << endl;
			cout << "CME_apex_speed: "    << CME_apex_speed << endl;
			cout << "CME_energy_control: "    << CME_energy_control << endl;
			cout << "CME_density: "    << CME_density << endl;
			cout << "CME_helicity_sign: "    << CME_helicity_sign << endl;
			cout << "CME_a1: "    << CME_a1 << endl;
			cout << "CME_shift: "    << CME_shift << endl;
			cout << "density_scale: "    << density_scale << endl;
			cout << "velocity_scale: "    << velocity_scale << endl;
			cout << "Sun_Z2: "    << Sun_Z2 << endl;
			cout << "Sun_SigmaC: "    << Sun_SigmaC << endl;
			cout << "Sun_Lambda: "    << Sun_Lambda << endl;
			cout << "SigmaD: "    << SigmaD << endl;
			cout << "alphaTM: "    << alphaTM << endl;
			cout << "betaTM: "    << betaTM << endl;
			cout << "smallTM: "    << smallTM << endl;
			cout << "Scaling_time: "    << Scaling_time << endl;
			cout << setw(75) << setfill('-') << "-" << endl;
		}

	}
    //Use the following format if encapsulation is needed (if variables needed to be made private) 
    //Getter
	int getBoxSize() {
      return BoxSize;
    }
	// Setter
    void setBoxSize(int s) {
      BoxSize = s;
    }

};
