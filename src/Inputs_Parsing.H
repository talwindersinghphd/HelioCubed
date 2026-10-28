#pragma once

#include "Proto.H"
#include "InputParser.H"
#include <algorithm> // For std::remove
#include <array>
#include <fstream>
#include <map>
#include <iostream>  // For std::pout(0), etc.
#include <iomanip>   // For std::setw, etc.
#include <vector>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include "MHD_pout.H"
#include "MHD_Constants.H"
#include "MHD_Time.H"
#include "MHD_AMRTagging.H"
#include "MHD_CME.H"
#include "MHD_Thomson.H"

#ifndef GIT_HASH_RAW
#define GIT_HASH_RAW unknown_hash
#endif

#ifndef GIT_MESSAGE_RAW
#define GIT_MESSAGE_RAW unknown_message
#endif

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

#define GIT_HASH    STRINGIFY(GIT_HASH_RAW)
#define GIT_MESSAGE STRINGIFY(GIT_MESSAGE_RAW)

using namespace Proto;
using namespace std;

/// @brief ParseInputs Singleton class
class ParseInputs {
private:
    int convTestType = 0; // 0: No convergence test, 1: Space Convergence test, 2: Space and Time Convergence test
    // -1: run all convergence levels in one MPI launch (legacy behavior);
    // 0, 1, 2: run only that level; 3: compare previously written levels.
    int convergence_level = -1;
    // A positive value overrides the convergence timestep. This lets separately
    // launched levels use the same timestep sequence as the legacy driver.
    // Keep this as text because the shared InputParser currently converts
    // numeric inputs through single precision before assigning to a double.
    string convergence_dt = "-1";
    // 0: file-driven solar wind; 1: analytic solar wind;
    // 2: radial pulse; 3: radial pulse with constant Cartesian B;
    // 4: non-radial pulse; 5: non-radial pulse with constant Cartesian B;
    // 6: strong spherical shock; 7: strong spherical shock with constant B;
    // 8: off-center hydro blast; 9: off-center blast with constant B = 0.1 G.
    // For 8/9 use gamma=1.4 (see exec/inputs_Blast).
    int init_condition_type = 0;
    int domainSize; // Number of cells in non radial direction
    int thickness; // Number of cells in radial direction
    // Required runtime geometry. Parse with stod because InputArgs converts
    // double inputs through single precision.
    string inner_radius;
    string outer_radius;
    string radial_stretch;
    int boxSize_nonrad; // Size of the box for domain decomposition
    int boxSize_rad; // Size of the box for domain decomposition
    int max_iter; 
    double max_time; // Maximum simulation time in seconds
    double CFL;
    int limiter; // 0: unlimited, 1: O4 limiter (minmod at O2), 2: minmod, 3: MC (O2 only)
    // 0: preserve spherical symmetry (default); 1: interpolate Cartesian U.
    int state_conversion = 0;
    // Optional seam treatments; retain the earlier, faster path by default.
    int BC_conserved_averages = 0; // 1: centered initialization and averaged BCs.
    int seam_pressure_correction = 0; // 1: match angular ghost energy to pressure.
    int spatial_order = 4; // 2: common PLM operator, 4: standard/Berta operator
    int spatial_operator = 1; // 1: Existing Euler operator, 2: Berta WENOZ operator
    int Riemann_solver_type = 1; // 1: Rusanov, 2: Roe 8-wave (standard and Berta)
    bool cme_rusanov = false; // Temporary solver override after a CME insertion.
    double entropy_fix_coeff = 0.3; // FullSphere entropy-fix coefficient; 0 disables it
    double Berta_eta_d = 0.02; // Berta detector threshold
    int Berta_detector_type = 0; // 0: Jameson detector, 1: Berta derivative-ratio detector
    int Berta_fallback_type = 0; // 0: WENO3 fallback, 1: PLM fallback
    double strength; // Strength of perturbation for test problems.
    int write_cadence = 10000000000; 
    int slice_cadence = 10000000000;
    double write_time_cadence = 1000000000; // In seconds
    double slice_time_cadence = 1000000000; // In seconds
    int checkpoint_cadence = 1000000000;
    int P_floor_cadence = 1000000000; // Iteration cadence for P_floor
    string data_file_prefix = "data";
    string checkpoint_file_prefix = "checkpoint";
    std::string slices_raw;
    bool slice_X = false;
    bool slice_Y = false;
    bool slice_Z = false;
    bool slice_R = false;
    void parseSlices() {
        slice_X = slice_Y = slice_Z = slice_R = false;
        std::istringstream iss(slices_raw);
        std::string token;
        while (iss >> token) {
            if (token == "X") slice_X = true;
            else if (token == "Y") slice_Y = true;
            else if (token == "Z") slice_Z = true;
            else if (token == "R") slice_R = true;
        }
    }
    int output_coordinates = 0; // Volume/slice vectors: 0=spherical, 1=Cartesian.
    int scale_data_with_r = 0; // 0: No scaling, 1: Scale with r
    int output_divB = 0; // 0: Do not write div(B), 1: Append div(B) to data output
    int max_checkpoint_files = 1;
    string restart_file;
    double gamma = 1.66667; // Ratio of specific heats
    int temporal_order = 4; // Temporal order of accuracy
    int divergence_cleaning = 1; // 0: No divergence cleaning, 1: Divergence cleaning
    int linear_viscosity = 1; // 0: No linear viscosity, 1: Linear viscosity
    int nonlinear_viscosity = 1; // 0: No nonlinear viscosity, 1: Nonlinear viscosity
    int viscosity_order = 5; // Linear viscosity order: 3 or 5
    int scale_viscosity_with_r = 1; // 0: historical scaling, 1: radius-scaled viscosity
    int boundary_reflux = 1; // 0: don't reflux at block boundaries, 1: reflux
    int radial_refinement = 0; // 0: refine in all directions by 2 for convergence studies;
                           // 1 refine only in radial direction.
    int MBInterp_define;   // old define (0) new define (1)
    // Mapped multiblock AMR. Each ratio list has one value (broadcast) or one per
    // level interface. Both angular coordinates share the nonradial ratio.
    int amr_max_level = 0;
    string amr_ref_ratio_rad = "2";
    string amr_ref_ratio_nonrad = "2";
    string amr_boxSize_rad = "0";
    string amr_boxSize_nonrad = "0";
    int amr_subcycling = 1;
    int amr_verbosity = 1; // 0: coarse steps, 1: level summaries, 2: individual events.
    int amr_regrid_interval = 4;
    double amr_regrid_time_cadence = 0;
    string amr_regrid_time_cadence_raw;
    int amr_buffer = 2;
    // Legacy fallback for density/pressure when their individual options are
    // omitted. Negative thresholds disable a criterion; zero tags any change.
    double amr_threshold = 0.05;
    string amr_refine_density_thresh_raw, amr_refine_pressure_thresh_raw;
    string amr_refine_B_thresh_raw, amr_refine_V_thresh_raw;
    double amr_refine_density_thresh = 0.05;
    double amr_refine_pressure_thresh = 0.05;
    double amr_refine_B_thresh = -1;
    double amr_refine_V_thresh = -1;
    // Finest level a criterion may create. -1 uses the hierarchy maximum;
    // zero disables that criterion without changing its threshold.
    int amr_refine_density_maxlevel = -1;
    int amr_refine_pressure_maxlevel = -1;
    int amr_refine_B_maxlevel = -1;
    int amr_refine_V_maxlevel = -1;
    string amr_refine_regions;
    string amr_refine_r_range,amr_refine_colat_range,amr_refine_lon_range;
    HelioCubed::AMRRefinementRegions amr_refine_bounds;
    string BC_file;
    int num_probes = 0;                                
    std::vector<std::string> probe_filenames;           
    std::vector<std::string> probe_coords;  // Accepts HGI coordinates              
    double Probe_cadence = 1000000000; // In seconds
    int num_jmaps=0, Jmap_elongation_bins=201;
    double Jmap_cadence=3600, Jmap_min_elongation=10, Jmap_max_elongation=50;
    double Jmap_los_step=0.001; // Maximum quadrature spacing, AU.
    string Jmap_cadence_raw,Jmap_min_elongation_raw,Jmap_max_elongation_raw,Jmap_los_step_raw;
    std::vector<HelioCubed::JmapParameters> jmaps;
    double BC_start_time = 0.0; // Like 2024.0 for year 2024
    // Epochs must bypass InputArgs' single-precision numeric conversion.
    string BC_start_time_raw = "0";
    CalendarEpoch BC_start_epoch;
    double BC_cadence = 1.0; // In hours
    int BC_frame_rotate = 0; // 0: Frames don't rotate (SWiG), 1: Frames rotate
    int BC_force_single_frame = 0; // 1: Use only one frame for BCs, 0: Use all frames for BCs (if present)
    int Sun_gravity = 0; // 0: No gravity, 1: Include Sun's gravity as source term
    int CME_type = 0; // Derived from the GCS source count.
    int num_CMEs = 0;
    std::vector<CMEParameters> CMEs; // Stable chronological order.

    // Private Constructor
    ParseInputs() {}

    // Private Copy Constructor and Assignment Operator
    ParseInputs(const ParseInputs&) = delete;
    ParseInputs& operator=(const ParseInputs&) = delete;

public:
    // Static method to get the instance of the class
    static ParseInputs& getInstance() {
        static ParseInputs instance; // Guaranteed to be destroyed and instantiated on first use
        return instance;
    }

    // Static Getters
    static int get_convTestType() {return getInstance().convTestType;}
    static int get_convergence_level() {return getInstance().convergence_level;}
    static double get_convergence_dt() {return std::stod(getInstance().convergence_dt);}
    static int get_init_condition_type() {return getInstance().init_condition_type;}
    static int get_domainSize() {return getInstance().domainSize;}
    static int get_thickness() {return getInstance().thickness;}
    static double get_inner_radius() {return cubedSphereShellGeometry().r0;}
    static double get_outer_radius() {return cubedSphereShellGeometry().r1;}
    static double get_radial_stretch() {return cubedSphereShellGeometry().radialStretch;}
    static int get_boxSize_nonrad() {return getInstance().boxSize_nonrad;}
    static int get_boxSize_rad() {return getInstance().boxSize_rad;}
    static int get_max_iter() {return getInstance().max_iter;}
    static double get_max_time() {return getInstance().max_time;}
    static double get_CFL() {return getInstance().CFL;}
    static int get_limiter() {return getInstance().limiter;}
    static int get_state_conversion() {return getInstance().state_conversion;}
    static int get_BC_conserved_averages() {return getInstance().BC_conserved_averages;}
    static int get_seam_pressure_correction() {return getInstance().seam_pressure_correction;}
    static int get_spatial_operator() {return getInstance().spatial_operator;}
    static void set_CME_Rusanov(bool active) {getInstance().cme_rusanov=active;}
    static int get_input_Riemann_solver_type() {return getInstance().Riemann_solver_type;}
    static int get_Riemann_solver_type() {return getInstance().cme_rusanov ? 1 : getInstance().Riemann_solver_type;}
    static double get_entropy_fix_coeff() {return getInstance().entropy_fix_coeff;}
    static double get_Berta_eta_d() {return getInstance().Berta_eta_d;}
    static int get_Berta_detector_type() {return getInstance().Berta_detector_type;}
    static int get_Berta_fallback_type() {return getInstance().Berta_fallback_type;}
    static double get_strength() {return getInstance().strength;}
    static int get_write_cadence() {return getInstance().write_cadence;}
    static int get_slice_cadence() {return getInstance().slice_cadence;}
    static double get_write_time_cadence() {return getInstance().write_time_cadence;}
    static double get_slice_time_cadence() {return getInstance().slice_time_cadence;}
    static int get_checkpoint_cadence() {return getInstance().checkpoint_cadence;}
    static int get_P_floor_cadence() {return getInstance().P_floor_cadence;}
    static string get_data_file_prefix() {return getInstance().data_file_prefix;}
    static string get_checkpoint_file_prefix() {return getInstance().checkpoint_file_prefix;}
    static bool get_slice_X() { return getInstance().slice_X; }
    static bool get_slice_Y() { return getInstance().slice_Y; }
    static bool get_slice_Z() { return getInstance().slice_Z; }
    static bool get_slice_R() { return getInstance().slice_R; }
    static int get_output_coordinates() {return getInstance().output_coordinates;}
    static int get_scale_data_with_r() {return getInstance().scale_data_with_r;}
    static int get_output_divB() {return getInstance().output_divB;}
    static int get_max_checkpoint_files() {return getInstance().max_checkpoint_files;}
    static string get_restart_file() {return getInstance().restart_file;}
    static double get_gamma() {return getInstance().gamma;}
    static int get_spatial_order() {return getInstance().spatial_order;}
    static int get_amr_max_level() {return getInstance().amr_max_level;}
    static const string& get_amr_ref_ratio_rad() {return getInstance().amr_ref_ratio_rad;}
    static const string& get_amr_ref_ratio_nonrad() {return getInstance().amr_ref_ratio_nonrad;}
    static const string& get_amr_boxSize_rad() {return getInstance().amr_boxSize_rad;}
    static const string& get_amr_boxSize_nonrad() {return getInstance().amr_boxSize_nonrad;}
    static int get_amr_subcycling() {return getInstance().amr_subcycling;}
    static int get_amr_verbosity() {return getInstance().amr_verbosity;}
    static int get_amr_regrid_interval() {return getInstance().amr_regrid_interval;}
    static double get_amr_regrid_time_cadence() {return getInstance().amr_regrid_time_cadence;}
    static int get_amr_buffer() {return getInstance().amr_buffer;}
    static double get_amr_threshold() {return getInstance().amr_threshold;}
    static double get_amr_refine_density_thresh() {return getInstance().amr_refine_density_thresh;}
    static double get_amr_refine_pressure_thresh() {return getInstance().amr_refine_pressure_thresh;}
    static double get_amr_refine_B_thresh() {return getInstance().amr_refine_B_thresh;}
    static double get_amr_refine_V_thresh() {return getInstance().amr_refine_V_thresh;}
    static int get_amr_refine_density_maxlevel()
    {return getInstance().amr_refine_density_maxlevel < 0 ? get_amr_max_level() : getInstance().amr_refine_density_maxlevel;}
    static int get_amr_refine_pressure_maxlevel()
    {return getInstance().amr_refine_pressure_maxlevel < 0 ? get_amr_max_level() : getInstance().amr_refine_pressure_maxlevel;}
    static int get_amr_refine_B_maxlevel()
    {return getInstance().amr_refine_B_maxlevel < 0 ? get_amr_max_level() : getInstance().amr_refine_B_maxlevel;}
    static int get_amr_refine_V_maxlevel()
    {return getInstance().amr_refine_V_maxlevel < 0 ? get_amr_max_level() : getInstance().amr_refine_V_maxlevel;}
    static bool uses_amr_tagging()
    {
        return (get_amr_refine_density_thresh() >= 0 && get_amr_refine_density_maxlevel() > 0)
            || (get_amr_refine_pressure_thresh() >= 0 && get_amr_refine_pressure_maxlevel() > 0)
            || (get_amr_refine_B_thresh() >= 0 && get_amr_refine_B_maxlevel() > 0)
            || (get_amr_refine_V_thresh() >= 0 && get_amr_refine_V_maxlevel() > 0);
    }
    static const string& get_amr_refine_regions() {return getInstance().amr_refine_regions;}
    static const HelioCubed::AMRRefinementRegions& get_amr_refine_bounds() {return getInstance().amr_refine_bounds;}
    static int get_temporal_order() {return getInstance().temporal_order;}
    static int get_divergence_cleaning() {return getInstance().divergence_cleaning;}
    static bool uses_second_order_limiter()
    {return get_spatial_order() == 2 && get_limiter() != 0;}
    // The slope limiter supplies dissipation for limited O2. Apply the same switches
    // to fluxes, timestep estimation, and ghost-width selection.
    static int get_linear_viscosity()
    {return uses_second_order_limiter() ? 0 : getInstance().linear_viscosity;}
    static int get_nonlinear_viscosity()
    {return uses_second_order_limiter() ? 0 : getInstance().nonlinear_viscosity;}
    static int get_viscosity_order() {return getInstance().viscosity_order;}
    static int get_scale_viscosity_with_r() {return getInstance().scale_viscosity_with_r;}
    static int get_boundary_reflux() {return getInstance().boundary_reflux;}
    static int get_radial_refinement() {return getInstance().radial_refinement;}
    static int get_MBInterp_define() {return getInstance().MBInterp_define;}
    static string get_BC_file() {return getInstance().BC_file;}
    static int get_num_probes() { return getInstance().num_probes; }                               
    static const std::vector<std::string>& get_probe_filenames() { return getInstance().probe_filenames; } 
    static const std::vector<std::string>& get_probe_coords()     { return getInstance().probe_coords; }     
    static double get_Probe_cadence() {return getInstance().Probe_cadence;}
    static const auto& get_jmaps() {return getInstance().jmaps;}
    static double get_Jmap_cadence() {return getInstance().Jmap_cadence;}
    static double get_Jmap_min_elongation() {return getInstance().Jmap_min_elongation;}
    static double get_Jmap_max_elongation() {return getInstance().Jmap_max_elongation;}
    static int get_Jmap_elongation_bins() {return getInstance().Jmap_elongation_bins;}
    static double get_Jmap_los_step() {return getInstance().Jmap_los_step;}
    static double get_BC_start_time() {return getInstance().BC_start_time;}
    static const CalendarEpoch& get_BC_start_epoch() {return getInstance().BC_start_epoch;}
    static double get_BC_cadence() {return getInstance().BC_cadence;}
    static int get_BC_frame_rotate() {return getInstance().BC_frame_rotate;}
    static int get_BC_force_single_frame() {return getInstance().BC_force_single_frame;}
    static int get_Sun_gravity() {return getInstance().Sun_gravity;}
    static int get_CME_type() {return getInstance().CME_type;}
    static const std::vector<CMEParameters>& get_CMEs() {return getInstance().CMEs;}
    // Parses now with command-line arguments
    void parsenow(int argc, char* argv[]) {
        InputArgs args;
        args.add("convTestType", convTestType);
        args.add("convergence_level", convergence_level);
        args.add("convergence_dt", convergence_dt);
        args.add("init_condition_type", init_condition_type);
        args.add("domainSize", domainSize);
        args.add("thickness", thickness);
        args.add("inner_radius", inner_radius);
        args.add("outer_radius", outer_radius);
        args.add("radial_stretch", radial_stretch);
        args.add("boxSize_nonrad", boxSize_nonrad);
        args.add("boxSize_rad", boxSize_rad);
        args.add("max_iter", max_iter);
        args.add("max_time", max_time);
        args.add("CFL", CFL);
        args.add("limiter", limiter);
        args.add("state_conversion", state_conversion);
        args.add("BC_conserved_averages", BC_conserved_averages);
        args.add("seam_pressure_correction", seam_pressure_correction);
        args.add("spatial_order", spatial_order);
        args.add("spatial_operator", spatial_operator);
        args.add("Riemann_solver_type", Riemann_solver_type);
        args.add("entropy_fix_coeff", entropy_fix_coeff);
        args.add("Berta_eta_d", Berta_eta_d);
        args.add("Berta_detector_type", Berta_detector_type);
        args.add("Berta_fallback_type", Berta_fallback_type);
        args.add("strength", strength);
        args.add("write_cadence", write_cadence);
        args.add("slice_cadence", slice_cadence);
        args.add("write_time_cadence", write_time_cadence);
        args.add("slice_time_cadence", slice_time_cadence);
        args.add("checkpoint_cadence", checkpoint_cadence);
        args.add("P_floor_cadence", P_floor_cadence);
        args.add("data_file_prefix", data_file_prefix);
        args.add("checkpoint_file_prefix", checkpoint_file_prefix);
        args.add("slices", slices_raw);
        args.add("output_coordinates", output_coordinates);
        args.add("scale_data_with_r", scale_data_with_r);
        args.add("output_divB", output_divB);
        args.add("max_checkpoint_files", max_checkpoint_files);
        args.add("restart_file", restart_file);
        args.add("gamma", gamma);
        args.add("temporal_order", temporal_order);
        args.add("divergence_cleaning", divergence_cleaning);
        args.add("linear_viscosity", linear_viscosity);
        args.add("nonlinear_viscosity", nonlinear_viscosity);
        args.add("viscosity_order", viscosity_order);
        args.add("scale_viscosity_with_r", scale_viscosity_with_r);
        args.add("boundary_reflux", boundary_reflux);
        args.add("radial_refinement", radial_refinement);
        args.add("MBInterp_define", MBInterp_define);
        args.add("amr_max_level", amr_max_level);
        args.add("amr_ref_ratio_rad", amr_ref_ratio_rad);
        args.add("amr_ref_ratio_nonrad", amr_ref_ratio_nonrad);
        args.add("amr_boxSize_rad", amr_boxSize_rad);
        args.add("amr_boxSize_nonrad", amr_boxSize_nonrad);
        args.add("amr_subcycling", amr_subcycling);
        args.add("amr_verbosity", amr_verbosity);
        args.add("amr_regrid_interval", amr_regrid_interval);
        args.add("amr_regrid_time_cadence", amr_regrid_time_cadence_raw);
        args.add("amr_buffer", amr_buffer);
        args.add("amr_threshold", amr_threshold);
        args.add("amr_refine_density_thresh", amr_refine_density_thresh_raw);
        args.add("amr_refine_pressure_thresh", amr_refine_pressure_thresh_raw);
        args.add("amr_refine_B_thresh", amr_refine_B_thresh_raw);
        args.add("amr_refine_V_thresh", amr_refine_V_thresh_raw);
        args.add("amr_refine_density_maxlevel", amr_refine_density_maxlevel);
        args.add("amr_refine_pressure_maxlevel", amr_refine_pressure_maxlevel);
        args.add("amr_refine_B_maxlevel", amr_refine_B_maxlevel);
        args.add("amr_refine_V_maxlevel", amr_refine_V_maxlevel);
        args.add("amr_refine_regions", amr_refine_regions);
        args.add("amr_refine_r_range", amr_refine_r_range);
        args.add("amr_refine_colat_range", amr_refine_colat_range);
        args.add("amr_refine_lon_range", amr_refine_lon_range);
        // Register the numbered keys present in this input, without imposing
        // a region-count limit or allocating storage for gaps in the indices.
        struct RegionInput {std::array<string,3> range;std::array<bool,3> present{};};
        std::map<int,RegionInput> numberedBounds;
        const std::array<string,3> rangeNames={"amr_refine_r_range","amr_refine_colat_range","amr_refine_lon_range"};
        auto registerRegion=[&](const string& option)
        {
            for(size_t coordinate=0;coordinate<rangeNames.size();++coordinate)
            {
                const string prefix="-"+rangeNames[coordinate]+".";
                if(option.compare(0,prefix.size(),prefix)!=0) continue;
                const string suffix=option.substr(prefix.size());
                const string message=option+" requires a nonnegative integer region index";
                if(suffix.empty() || !std::all_of(suffix.begin(),suffix.end(),[](char c){return c>='0' && c<='9';}))
                    throw std::invalid_argument(message);
                int index;
                try {index=std::stoi(suffix);} catch(...) {throw std::invalid_argument(message);}
                auto& region=numberedBounds[index];
                region.present[coordinate]=true;
                args.add(option.substr(1),region.range[coordinate]);
                return true;
            }
            return false;
        };
        if(argc==2)
        {
            std::ifstream file(argv[1]);
            if(!file) throw std::invalid_argument("Could not open inputs file: "+string(argv[1]));
            string line,option;
            while(std::getline(file,line))
            {
                std::istringstream row(line);
                if(row>>option) registerRegion(option);
            }
        }
        else for(int i=1;i<argc;++i)
            if(registerRegion(argv[i]) && i+1==argc)
                throw std::invalid_argument(string(argv[i])+" requires two finite numbers: min,max");
        args.add("BC_file", BC_file);
        args.add("num_probes", num_probes); 
        constexpr int MAX_PROBES = 32; // or more if needed
        // ensure storage exists before parse:
        probe_filenames.assign(MAX_PROBES, "");
        probe_coords.assign(MAX_PROBES, "");
        for (int i = 0; i < MAX_PROBES; ++i) {
            const std::string base = "probe" + std::to_string(i) + ".";
            args.add(base + "filename", probe_filenames[(size_t)i]);
            args.add(base + "coords",   probe_coords[(size_t)i]);
        }
        args.add("Probe_cadence", Probe_cadence);
        args.add("num_jmaps",num_jmaps);
        args.add("Jmap_cadence",Jmap_cadence_raw);
        args.add("Jmap_min_elongation",Jmap_min_elongation_raw);
        args.add("Jmap_max_elongation",Jmap_max_elongation_raw);
        args.add("Jmap_elongation_bins",Jmap_elongation_bins);
        args.add("Jmap_los_step",Jmap_los_step_raw);
        constexpr int MAX_JMAPS=8;
        std::vector<string> jmapPositionAngles(MAX_JMAPS);
        jmaps.resize(MAX_JMAPS);
        for(int i=0;i<MAX_JMAPS;++i)
        {
            const string base="jmap"+std::to_string(i)+".";
            args.add(base+"filename",jmaps[i].filename);
            args.add(base+"observer",jmaps[i].observer);
            args.add(base+"reference",jmaps[i].reference);
            args.add(base+"position_angle",jmapPositionAngles[i]);
        }
        args.add("BC_start_time", BC_start_time_raw);
        args.add("BC_cadence", BC_cadence);
        args.add("BC_frame_rotate", BC_frame_rotate);
        args.add("BC_force_single_frame", BC_force_single_frame);
        args.add("Sun_gravity", Sun_gravity);
        args.add("num_CMEs", num_CMEs);
        // Discover indexed options before InputArgs parses them. Keep their
        // values as strings: InputArgs otherwise rounds doubles through float.
        std::map<int,std::map<string,string>> sourceInputs;
        std::vector<std::pair<int,GCS::Result>> gcsResults;
        auto registerSource = [&](const string& option) {
            if (option.compare(0,4,"-CME") != 0) return;
            const size_t dot = option.find('.',4);
            if (dot == string::npos)
                throw std::invalid_argument("CME inputs now use -num_CMEs and indexed GCS fields; unsupported option: " + option);
            const string number = option.substr(4,dot-4);
            if (number.empty() || !std::all_of(number.begin(),number.end(),[](char c){return c >= '0' && c <= '9';}))
                throw std::invalid_argument("Invalid indexed CME option: " + option);
            const int index = std::stoi(number);
            const string field = option.substr(dot+1);
            static const std::vector<string> fields = {"CME_type","GCS_speed","GCS_fit_height","GCS_fit_time",
                "GCS_latitude","GCS_longitude","GCS_tilt","GCS_half_angle","GCS_aspect_ratio",
                "DBM_gamma","DBM_wind_speed","DBM_target_height","CME_poloidal_flux","CME_toroidal_flux",
                "CME_leading_polarity","CME_helicity_sign","CME_density","GCS_earth_trajectory",
                "CME_injection_method","CME_plasma_beta","CME_boundary_taper_time"};
            if (std::find(fields.begin(),fields.end(),field) == fields.end())
                throw std::invalid_argument("Unknown indexed CME field: " + option);
            args.add(option.substr(1),sourceInputs[index][field]);
        };
        if (argc == 2) {
            std::ifstream file(argv[1]);
            string line, option;
            while (std::getline(file,line)) {
                std::istringstream row(line);
                if (row >> option) registerSource(option);
            }
        } else for (int i=1; i<argc; ++i) registerSource(argv[i]);
        args.parse(argc, argv);
        auto requireGeometry = [](const string& raw, const char* name) {
            if (raw.empty()) {
                throw std::invalid_argument(
                    "HelioCubed requires -inner_radius, -outer_radius, and -radial_stretch "
                    "in the inputs file; -" + string(name) + " is missing or commented out.");
            }
        };
        requireGeometry(inner_radius, "inner_radius");
        requireGeometry(outer_radius, "outer_radius");
        requireGeometry(radial_stretch, "radial_stretch");
        if (spatial_order != 2 && spatial_order != 4) {
            throw std::invalid_argument("spatial_order must be 2 or 4.");
        }
        if (amr_max_level < 0 || amr_regrid_interval < 0
            || amr_buffer < 0 || !std::isfinite(amr_threshold)) {
            throw std::invalid_argument("AMR requires nonnegative max level, regrid interval, and buffer, and a finite threshold.");
        }
        if (amr_subcycling != 0 && amr_subcycling != 1) {
            throw std::invalid_argument("amr_subcycling must be 0 or 1.");
        }
        if (amr_verbosity < 0 || amr_verbosity > 2) {
            throw std::invalid_argument("amr_verbosity must be 0, 1, or 2.");
        }
        if (limiter < 0 || limiter > 3) {
            throw std::invalid_argument("limiter must be 0 (unlimited), 1 (O4 limiter; minmod at O2), 2 (minmod), or 3 (MC at O2).");
        }
        if (limiter == 3 && spatial_order != 2) {
            throw std::invalid_argument("limiter=3 (MC) requires spatial_order=2.");
        }
        if (temporal_order != 1 && temporal_order != 2 && temporal_order != 3 && temporal_order != 4) {
            throw std::invalid_argument("temporal_order must be 1, 2, 3, or 4.");
        }
        if (state_conversion != 0 && state_conversion != 1) {
            throw std::invalid_argument("state_conversion must be 0 (symmetry preserving) or 1 (Cartesian).");
        }
        if (output_coordinates != 0 && output_coordinates != 1) {
            throw std::invalid_argument("output_coordinates must be 0 (spherical) or 1 (Cartesian).");
        }
        if (BC_conserved_averages != 0 && BC_conserved_averages != 1) {
            throw std::invalid_argument("BC_conserved_averages must be 0 or 1.");
        }
        if (seam_pressure_correction != 0 && seam_pressure_correction != 1) {
            throw std::invalid_argument("seam_pressure_correction must be 0 or 1.");
        }
        if ((spatial_order == 2 || state_conversion == 1) && CUBED_SPHERE_SHELL_RADIAL_COORD != 0) {
            throw std::invalid_argument("Cartesian state_conversion requires radial coordinate 0.");
        }
        if (viscosity_order != 3 && viscosity_order != 5) {
            throw std::invalid_argument("viscosity_order must be 3 or 5.");
        }
        auto geometry = cubedSphereShellGeometry();
        auto parsePreciseValue = [](const string& raw, const char* name,
                                     double defaultValue) {
            if (raw.empty()) return defaultValue;
            try {
                size_t end = 0;
                double value = std::stod(raw, &end);
                size_t tail = raw.find_first_not_of(" \t\r\n", end);
                if (tail != string::npos && raw[tail] != '#') {
                    throw std::invalid_argument("trailing characters");
                }
                return value;
            } catch (const std::exception&) {
                throw std::invalid_argument(string(name) + " must be a number.");
            }
        };
        auto refinementThreshold = [&](const string& raw, const char* name, double fallback) {
            const double value = parsePreciseValue(raw, name, fallback);
            if (!std::isfinite(value))
                throw std::invalid_argument(string(name) + " must be finite.");
            return value;
        };
        amr_regrid_time_cadence = parsePreciseValue(amr_regrid_time_cadence_raw,
            "amr_regrid_time_cadence", 0);
        if (!std::isfinite(amr_regrid_time_cadence) || amr_regrid_time_cadence < 0)
            throw std::invalid_argument("amr_regrid_time_cadence must be finite and nonnegative (0 disables it).");
        amr_refine_density_thresh = refinementThreshold(amr_refine_density_thresh_raw,
            "amr_refine_density_thresh", amr_threshold);
        amr_refine_pressure_thresh = refinementThreshold(amr_refine_pressure_thresh_raw,
            "amr_refine_pressure_thresh", amr_threshold);
        amr_refine_B_thresh = refinementThreshold(amr_refine_B_thresh_raw, "amr_refine_B_thresh", -1);
        amr_refine_V_thresh = refinementThreshold(amr_refine_V_thresh_raw, "amr_refine_V_thresh", -1);
        const auto legacyBounds=HelioCubed::AMRRefinementBounds::parse(
            amr_refine_r_range,amr_refine_colat_range,amr_refine_lon_range);
        if(legacyBounds.active() && !numberedBounds.empty())
            throw std::invalid_argument("Use either unnumbered AMR physical tag bounds or numbered .0, .1, ... regions; do not mix them");
        amr_refine_bounds.regions.clear();
        if(legacyBounds.active()) amr_refine_bounds.regions.push_back(legacyBounds);
        for(const auto& entry : numberedBounds)
        {
            const auto& region=entry.second;
            const string suffix="."+std::to_string(entry.first);
            for(size_t c=0;c<rangeNames.size();++c)
                if(region.present[c] && region.range[c].empty())
                    throw std::invalid_argument(rangeNames[c]+suffix+" requires two finite numbers: min,max");
            amr_refine_bounds.regions.push_back(HelioCubed::AMRRefinementBounds::parse(
                region.range[0],region.range[1],region.range[2],suffix));
        }
        if (amr_refine_density_maxlevel < -1 || amr_refine_pressure_maxlevel < -1
            || amr_refine_B_maxlevel < -1 || amr_refine_V_maxlevel < -1)
            throw std::invalid_argument("AMR variable maxlevels must be -1 (hierarchy maximum) or nonnegative.");
        BC_start_epoch = parseCalendarEpoch(BC_start_time_raw, "BC_start_time");
        BC_start_time = BC_start_epoch.decimalYear();
        CMEs.clear();
        if (num_CMEs < 0) throw std::invalid_argument("num_CMEs must be nonnegative");
        for (const auto& entry : sourceInputs)
            if (entry.first >= num_CMEs) throw std::invalid_argument("Indexed CME exceeds num_CMEs");
        for (int i=0; i<num_CMEs; ++i) {
            const string prefix="CME"+std::to_string(i)+".";
            const auto entry=sourceInputs.find(i);
            if (entry==sourceInputs.end()) throw std::invalid_argument("Missing inputs for "+prefix);
            const auto& raw=entry->second;
            auto value=[&](const char* name,double fallback,bool required=false) {
                auto field=raw.find(name);
                if (field==raw.end() || field->second.empty()) {
                    if (required) throw std::invalid_argument("Missing "+prefix+name);
                    return fallback;
                }
                const double result=parsePreciseValue(field->second,(prefix+name).c_str(),fallback);
                if (!std::isfinite(result)) throw std::invalid_argument(prefix+name+" must be finite");
                return result;
            };
            auto textValue=[&](const char* key,const string& fallback) {
                auto it=raw.find(key);return it==raw.end() || it->second.empty() ? fallback : it->second;
            };
            const double type=value("CME_type",1);
            if(type!=0 && type!=1 && type!=2)
                throw std::invalid_argument(prefix+"CME_type must be 0 (disabled), 1 (FRiED), or 2 (GCS)");
            if(type==0) continue;
            if(BC_start_epoch.year==0) throw std::invalid_argument("GCS conversion requires BC_start_time");
            GCS::Input input;
            input.cmeType=static_cast<int>(type);
            input.speed=value("GCS_speed",0,true);input.fitHeight=value("GCS_fit_height",0,true);
            const string fitTime=textValue("GCS_fit_time","");
            if(fitTime.empty()) throw std::invalid_argument("Missing "+prefix+"GCS_fit_time");
            input.fitTime=GCS::parseFitTime(fitTime);
            input.latitude=value("GCS_latitude",0,true);input.longitude=value("GCS_longitude",0,true);
            input.tilt=value("GCS_tilt",0,true);input.halfAngle=value("GCS_half_angle",0,true);
            input.aspectRatio=value("GCS_aspect_ratio",0,true);
            input.gamma=value("DBM_gamma",1e-8);input.wind=value("DBM_wind_speed",450);
            input.targetHeight=value("DBM_target_height",70);
            input.poloidalFlux=value("CME_poloidal_flux",std::numeric_limits<double>::quiet_NaN());
            input.toroidalFlux=value("CME_toroidal_flux",std::numeric_limits<double>::quiet_NaN());
            const double polarity=value("CME_leading_polarity",1),helicity=value("CME_helicity_sign",1);
            if((polarity!=1 && polarity!=-1) || (helicity!=1 && helicity!=-1))
                throw std::invalid_argument("CME leading polarity and helicity must be +1 or -1");
            input.leadingPolarity=static_cast<int>(polarity);input.helicity=static_cast<int>(helicity);
            input.earthTrajectory=textValue("GCS_earth_trajectory","trajEarth.dat");
            const double inner=parsePreciseValue(inner_radius,"inner_radius",0.0)/c_SR;
            auto result=GCS::convert(input,BC_start_epoch,inner);
            result.cme.sourceIndex=i;
            result.cme.density=value("CME_density",2.77e-24);
            result.cme.plasmaBeta=value("CME_plasma_beta",0);
            if(result.cme.plasmaBeta<0) throw std::invalid_argument(prefix+"CME_plasma_beta must be nonnegative");
            result.cme.boundaryTaperTime=value("CME_boundary_taper_time",43200);
            if(result.cme.boundaryTaperTime<0)
                throw std::invalid_argument(prefix+"CME_boundary_taper_time must be nonnegative seconds");
            const string method=textValue("CME_injection_method","volume");
            if(method!="volume" && method!="boundary")
                throw std::invalid_argument(prefix+"CME_injection_method must be volume or boundary");
            result.cme.injectionMethod=method=="boundary";
            result.cme.referenceTime=result.cme.insertionTime;
            if(result.cme.injectionMethod==1) {
                if(result.cme.boundaryTaperTime==0)
                    throw std::invalid_argument(prefix+"CME_boundary_taper_time must be positive for boundary emergence");
                if(spatial_order!=2 || spatial_operator!=1)
                    throw std::invalid_argument("CME boundary emergence requires the standard second-order operator");
                if(!(result.cme.apex_speed>0 && result.cme.r1+result.cme.r0>inner))
                    throw std::invalid_argument("CME boundary emergence needs positive speed and reference height above the inner boundary");
                result.cme.insertionTime=CMEModel::boundaryStartTime(result.cme,inner);
                if(result.cme.insertionTime<0)
                    throw std::invalid_argument("Boundary CME begins before BC_start_time; use an earlier background epoch");
                result.cme.enterTime=advanceCalendarEpoch(BC_start_epoch,result.cme.insertionTime).decimalYear();
            }
            CMEs.push_back(result.cme);gcsResults.push_back({i,result});
        }
        CME_type=CMEs.empty() ? 0 : 1;
        for (auto& c : CMEs) {
            for (double v : {c.r1,c.r0,c.halfangle,c.pol_flux_control,c.FRiED_n,c.lat,c.lon,c.tilt,
                c.apex_speed,c.density,c.helicity_sign,c.a1})
                if (!std::isfinite(v)) throw std::invalid_argument("CME parameters must be finite");
            if (c.r1<=0 || c.r0<=0 || c.halfangle<=0 || c.halfangle>=180
                || (c.fieldModel==1 && c.FRiED_n<=0)
                || c.pol_flux_control==0 || c.density<=0 || c.apex_speed<0
                || (c.helicity_sign!=1 && c.helicity_sign!=-1))
                throw std::invalid_argument("Invalid CME geometry, density, speed, twist, or helicity");
            if (c.fieldModel==1) c.normalAxis.prepare(c.r1,c.r0,c.halfangle,c.FRiED_n);
        }
        std::stable_sort(CMEs.begin(),CMEs.end(),[](const CMEParameters& a,const CMEParameters& b) {
            return a.insertionTime < b.insertionTime;
        });
        for (const auto& c : CMEs) {
            pout(0)<<"Scheduled CME"<<c.sourceIndex<<" at "<<std::setprecision(17)
                <<c.insertionTime<<" s (decimal year "<<c.enterTime<<", CME_type "<<c.fieldModel;
            if(c.fieldModel==2) pout(0)<<", GCS aspect ratio "<<c.gcsTube.aspectRatio
                <<", minimum section metric "<<c.gcsTube.minimumMetric
                <<", smooth join half-width "<<c.gcsTube.joinHalfWidth<<" Rsun";
            else pout(0)<<", max curvature*radius "<<c.normalAxis.maxCurvatureRadius;
            pout(0)<<", "<<(c.injectionMethod ? "boundary emergence" : "volume insertion with boundary leg feeding")
                <<")"<<std::setprecision(6)<<endl;
        }
        if(!gcsResults.empty() && procID()==0) {
            std::ofstream resolved("CME_resolved.inputs"),report("CME_conversion_report.txt");
            if(!resolved || !report) throw std::runtime_error("Cannot write resolved GCS parameter files");
            resolved<<std::setprecision(17)<<"# Resolved GCS CME block; retain the background inputs from the original run file.\n-num_CMEs "<<num_CMEs<<"\n";
            for(int index=0;index<num_CMEs;++index) {
                const string p="-CME"+std::to_string(index)+".";
                const auto item=std::find_if(gcsResults.begin(),gcsResults.end(),
                    [&](const auto& value){return value.first==index;});
                if(item==gcsResults.end()) {resolved<<p<<"CME_type 0\n";continue;}
                resolved<<p<<"CME_type "<<item->second.cme.fieldModel<<"\n";
                for(const auto& field : sourceInputs.at(index))
                    if(field.first!="CME_type" && field.first!="CME_poloidal_flux" && field.first!="CME_toroidal_flux"
                        && field.first!="CME_injection_method" && field.first!="CME_boundary_taper_time")
                        resolved<<p<<field.first<<" "<<field.second<<"\n";
                resolved<<p<<"CME_injection_method "<<(item->second.cme.injectionMethod ? "boundary" : "volume")<<"\n";
                resolved<<p<<"CME_boundary_taper_time "<<item->second.cme.boundaryTaperTime<<"\n";
                resolved<<p<<"CME_poloidal_flux "<<item->second.poloidalFlux<<"\n"
                    <<p<<"CME_toroidal_flux "<<item->second.toroidalFlux<<"\n";
            }
            report<<std::setprecision(17)<<"Native GCS conversion; solar radius 696000 km. Fluxes are positive magnitudes in Mx.\n"
                <<"Stonyhurst GCS longitude; additive solenoidal CME magnetic field and positive-temperature mass insertion.\n"
                <<"The CME driver supplies its kinetic/magnetic energy; background thermal energy is not spent on acceleration.\n"
                <<"Volume and emergence velocities are homologous, v=V_apex*r/H; only volume leg feeding retains an ambient-speed lower bound.\n"
                <<"The source continues the fitted DBM height and speed; B scales as H^-2. This ideal scaling applies to the homologous rope component.\n"
                <<"O2 inner ghosts feed the expanding rooted rope at every RK stage; boundary emergence starts when its front reaches the inner radius.\n"
                <<"Each boundary source fades with a smooth quintic from insertion to CME_boundary_taper_time seconds later; expired sources leave only WSA.\n"
                <<"The taper scales added B, loaded density and excess velocity; thermal and total energy are rebuilt consistently. Volume insertion is unchanged.\n"
                <<"Density loading is automatic for every CME: rho_total=max(rho_bg+CME_density, B_total^2/(4*pi*(u_radial/"
                <<CMEModel::minimumAlfvenMach<<")^2)).\n"
                <<"CME_density is the minimum added density; the common built-in Alfven Mach target is "
                <<CMEModel::minimumAlfvenMach<<". Added mass is reported, not normalized.\n"
                <<"Boundary emergence supports the standard HOST second-order operator on uniform grids and AMR hierarchies.\n"
                <<"Boundary emergence solves an exterior/interior Riemann problem at inner faces, sharing the interior normal B.\n"
                <<"Its radial field evolves through the finite-volume induction flux, without prescribing the analytic radial rope field at that face.\n"
                <<"For input Roe, emergence uses Rusanov and at most half the input CFL while feeding; after feeding and stabilization the input solver and CFL return.\n"
                <<"The emergence CFL estimate also includes a conservative inflow wave-speed bound before the new boundary field reaches owned cells.\n"
                <<"This is boundary superposition, not the external-field displacement method; the cell-centered solver uses Powell cleaning, not constrained transport.\n"
                <<"CME_type 1: normal FRiED sections may be thinned to keep max curvature*radius <= "<<GCS::curvatureMargin<<" at fixed leading height.\n"
                <<"CME_type 2: GCS circular sections and conical legs with C2 maps in short cone/front transition regions.\n"
                <<"Supplied GCS aspect ratio, half angle, apex and geometry outside those transitions are retained.\n"
                <<"Both fields conserve analytic axial flux and have constant dtheta/d(axis arclength); no force-free equilibrium is assumed.\n"
                <<"For GCS, theta is the azimuth in the smoothly joined circular section planes.\n"
                <<"GCS join smoothing acts on the section maps, preserving flux and constant turn without internal tangential B jumps.\n"
                <<"Domain poloidal flux means the mean of inner/outer unsigned plane fluxes above the inner radius.\n"
                <<"Cutting the rope at the inner boundary can leave these two plane contributions unequal.\n"
                <<"Empirical defaults match create_input_files_v3.py: Sindhuja & Gopalswamy 2020, v=438.9*(Phi_p/1e21)^0.34 km/s before DBM.\n"
                <<"Toroidal estimate: Qiu2007, Phi_t=0.32e21*(Phi_p/1e21)^1.20, assuming Phi_p~=Phi_RC.\n"
                <<"References: https://doi.org/10.3847/1538-4357/ab620f ; https://solar.physics.montana.edu/qiuj/publication/apj_flux.pdf\n";
            for(const auto& item : gcsResults) {
                const auto& r=item.second;const auto& c=r.cme;
                report<<"\nCME"<<item.first<<": reference-height epoch "<<formatUTCEpoch(r.epoch)<<", transit "<<r.transitSeconds<<" s\n"
                    <<"injection method "<<(c.injectionMethod ? "boundary" : "volume")<<"; start time "<<c.insertionTime
                    <<" s; reference time "<<c.referenceTime<<" s; added plasma beta "<<c.plasmaBeta<<"\n"
                    <<"boundary taper duration "<<c.boundaryTaperTime<<" s; WSA-only after simulation time "
                    <<c.insertionTime+c.boundaryTaperTime<<" s\n"
                    <<"automatic density loading; minimum added density "<<c.density<<" g/cm^3\n"
                    <<"CME_type "<<c.fieldModel<<"; speed "<<c.apex_speed<<" km/s; half angle "<<c.halfangle
                    <<" deg; Rt "<<c.r1<<" Rsun; Rp "<<c.r0<<" Rsun\n"
                    <<"original aspect ratio "<<r.originalAspectRatio<<"; actual apex ratio Rp/Rt "<<c.r0/c.r1<<"\n";
                if(c.fieldModel==2) report<<"GCS with smooth joins; minimum section metric "<<c.gcsTube.minimumMetric
                    <<"; conical leg join length "<<c.gcsTube.joinLength<<" Rsun"
                    <<"; transition half-width "<<c.gcsTube.joinHalfWidth<<" Rsun\n";
                else report<<"FRiED n "<<c.FRiED_n<<"; original max curvature*radius "<<r.originalMaxCurvatureRadius
                    <<"; adjusted max "<<c.normalAxis.maxCurvatureRadius<<"\n";
                report<<"Phi_p "<<r.poloidalFlux<<(r.estimatedPoloidal ? " estimated from GCS speed" : " supplied")<<"\n"
                    <<"Phi_t "<<r.toroidalFlux<<(r.estimatedToroidal ? " estimated using Qiu2007 from poloidal flux" : " supplied")<<"\n"
                    <<"full-rope Phi_p "<<r.fullPoloidalFlux
                    <<"; domain inner "<<r.domainInnerFlux<<"; domain outer "<<r.domainOuterFlux<<"\n"
                    <<"simulation longitude "<<c.lon<<" deg; tilt "<<c.tilt<<" deg\n"
                    <<"a1 "<<c.a1<<" G; pol multiplier "<<c.pol_flux_control<<" per AU\n";
            }
            pout(0)<<"GCS inputs converted; wrote CME_resolved.inputs and CME_conversion_report.txt"<<endl;
        }
        geometry.r0 = parsePreciseValue(inner_radius, "inner_radius", 0.0);
        geometry.r1 = parsePreciseValue(outer_radius, "outer_radius", 0.0);
        geometry.radialStretch = parsePreciseValue(radial_stretch, "radial_stretch", 0.0);
        if (!std::isfinite(geometry.r0) || !std::isfinite(geometry.r1)
            || geometry.r0 <= 0.0 || geometry.r1 <= geometry.r0) {
            throw std::invalid_argument(
                "Shell radii must be finite and satisfy 0 < inner_radius < outer_radius (cm).");
        }
        if (!std::isfinite(geometry.radialStretch)
            || !std::isfinite(std::expm1(geometry.radialStretch))) {
            throw std::invalid_argument("radial_stretch must be finite without exponential overflow.");
        }
        // Configure Proto before any maps, interpolation operators, or metrics
        // are built, so every consumer uses the same runtime geometry.
        cubedSphereShellGeometry() = geometry;
        parseSlices();
        // use only the first num_probes entries
        if (num_probes < 0 || num_probes > MAX_PROBES)
            throw std::invalid_argument("num_probes must be between 0 and 32.");
        probe_filenames.resize((size_t)std::max(0, num_probes));
        probe_coords.resize((size_t)std::max(0, num_probes));
        if(num_jmaps<0 || num_jmaps>MAX_JMAPS)
            throw std::invalid_argument("num_jmaps must be between 0 and 8");
        jmaps.resize(num_jmaps);
        Jmap_cadence=parsePreciseValue(Jmap_cadence_raw,"Jmap_cadence",Jmap_cadence);
        Jmap_min_elongation=parsePreciseValue(Jmap_min_elongation_raw,"Jmap_min_elongation",Jmap_min_elongation);
        Jmap_max_elongation=parsePreciseValue(Jmap_max_elongation_raw,"Jmap_max_elongation",Jmap_max_elongation);
        Jmap_los_step=parsePreciseValue(Jmap_los_step_raw,"Jmap_los_step",Jmap_los_step);
        for(int i=0;i<num_jmaps;++i)
            jmaps[i].positionAngle=parsePreciseValue(jmapPositionAngles[i],"J-map position angle",90);
        if(num_jmaps && (!(Jmap_cadence>0) || !std::isfinite(Jmap_cadence)
            || !(Jmap_los_step>0) || !std::isfinite(Jmap_los_step)
            || !(Jmap_min_elongation>0 && Jmap_max_elongation<180
                && Jmap_max_elongation>Jmap_min_elongation)
            || Jmap_elongation_bins<2 || Jmap_elongation_bins>10000))
            throw std::invalid_argument("Invalid J-map cadence, LOS step, or elongation grid");
        std::vector<string> jmapNames;
        for(const auto& jmap : jmaps)
        {
            if(jmap.filename.empty() || jmap.observer.empty() || !std::isfinite(jmap.positionAngle)
                || std::find(jmapNames.begin(),jmapNames.end(),jmap.filename)!=jmapNames.end())
                throw std::invalid_argument("Each J-map needs a unique filename, observer trajectory, and finite position angle");
            jmapNames.push_back(jmap.filename);
        }

        pout(0) << setw(75) << setfill('-') << "-" << endl;
        pout(0) << "Last git pull info:" << endl;
        pout(0) << setw(75) << setfill('-') << "-" << endl;
        pout(0) << "Git hash: " << GIT_HASH << endl; 
        pout(0) << "Commit message: " << GIT_MESSAGE << endl;
        pout(0) << setw(75) << setfill('-') << "-" << endl;
        pout(0) << "Running with the following Cubed Sphere geometry from inputs" << endl;
        pout(0) << setw(75) << setfill('-') << "-" << endl;
        pout(0) << "Inner Radius: " << get_inner_radius()/c_SR << " SR" << endl;
        pout(0) << "Outer Radius: " << get_outer_radius()/c_SR << " SR" << endl;
        pout(0) << "Radial Grid Stretching Parameter (C): " << get_radial_stretch() << endl;
        pout(0) << setw(75) << setfill('-') << "-" << endl;
        pout(0) << "Running with the following inputs (these can be changed in the inputs file)" << endl;
        pout(0) << setw(75) << setfill('-') << "-" << endl;
        pout(0) << "convTestType: " << convTestType << endl;
        pout(0) << "convergence_level: " << convergence_level << endl;
        pout(0) << "convergence_dt: " << convergence_dt << endl;
        pout(0) << "init_condition_type: " << init_condition_type << endl;
        pout(0) << "domainSize: " << domainSize << endl;
        pout(0) << "thickness: " << thickness << endl;
        pout(0) << "boxSize_nonrad: " << boxSize_nonrad << endl;
        pout(0) << "boxSize_rad: " << boxSize_rad << endl;
        pout(0) << "max_iter: " << max_iter << endl;
        pout(0) << "max_time: " << max_time << endl;
        pout(0) << "CFL: " << CFL << endl;
        pout(0) << "limiter: " << limiter << endl;
        pout(0) << "state_conversion: " << state_conversion << endl;
        pout(0) << "BC_conserved_averages: " << BC_conserved_averages << endl;
        pout(0) << "seam_pressure_correction: " << seam_pressure_correction << endl;
        pout(0) << "spatial_operator: " << spatial_operator << endl;
        pout(0) << "Riemann_solver_type: " << Riemann_solver_type << endl;
        pout(0) << "entropy_fix_coeff: " << entropy_fix_coeff << endl;
        pout(0) << "Berta_eta_d: " << Berta_eta_d << endl;
        pout(0) << "Berta_detector_type: " << Berta_detector_type << endl;
        pout(0) << "Berta_fallback_type: " << Berta_fallback_type << endl;
        pout(0) << "strength: " << strength << endl;
        pout(0) << "write_cadence: " << write_cadence << endl;
        pout(0) << "slice_cadence: " << slice_cadence << endl;
        pout(0) << "write_time_cadence: " << write_time_cadence << endl;
        pout(0) << "slice_time_cadence: " << slice_time_cadence << endl;
        pout(0) << "checkpoint_cadence: " << checkpoint_cadence << endl;
        pout(0) << "P_floor_cadence: " << P_floor_cadence << endl;
        pout(0) << "data_file_prefix: " << data_file_prefix << endl;
        pout(0) << "checkpoint_file_prefix: " << checkpoint_file_prefix << endl;
        pout(0) << "Slices: " << slices_raw << endl;
        pout(0) << "output_coordinates: " << output_coordinates
                << (output_coordinates == 0 ? " (spherical)" : " (Cartesian)") << endl;
        pout(0) << "scale_data_with_r: " << scale_data_with_r << endl;
        pout(0) << "output_divB: " << output_divB << endl;
        pout(0) << "max_checkpoint_files: " << max_checkpoint_files << endl;
        pout(0) << "restart_file: " << restart_file << endl;
        pout(0) << "gamma: " << gamma << endl;
        pout(0) << "spatial_order: " << spatial_order << endl;
        if (amr_max_level) {
            pout(0) << "AMR: max level " << amr_max_level << ", radial ratios " << amr_ref_ratio_rad
                    << ", angular ratios " << amr_ref_ratio_nonrad << (temporal_order==4 ? ", RK4 with " : temporal_order==3 ? ", RK3 with " : ", SSPRK2 with ")
                    << (amr_subcycling ? "CFL-based time subcycling" : "a common timestep")
                    << "; regrid interval "
                    << amr_regrid_interval << " base steps, time cadence " << amr_regrid_time_cadence
                    << "(s), buffer " << amr_buffer << endl;
            pout(0) << "AMR fine patch sizes: radial " << amr_boxSize_rad
                    << ", angular " << amr_boxSize_nonrad << " (0=automatic)" << endl;
            auto criterion = [&](const char* name, double threshold, int maxLevel) {
                pout(0) << "AMR relative-gradient criterion " << name << ": ";
                if (threshold < 0 || maxLevel == 0) pout(0) << "disabled";
                else pout(0) << "threshold " << threshold << ", finest level " << std::min(maxLevel, amr_max_level);
                pout(0) << endl;
            };
            criterion("density", amr_refine_density_thresh, get_amr_refine_density_maxlevel());
            criterion("pressure", amr_refine_pressure_thresh, get_amr_refine_pressure_maxlevel());
            criterion("|B|", amr_refine_B_thresh, get_amr_refine_B_maxlevel());
            criterion("|v|", amr_refine_V_thresh, get_amr_refine_V_maxlevel());
            auto printBounds=[&](const string& suffix,const std::array<string,3>& ranges)
            {
                pout(0)<<"AMR physical tag bounds"<<suffix<<":";
                if(!ranges[0].empty()) pout(0)<<" r="<<ranges[0]<<" (cm)";
                if(!ranges[1].empty()) pout(0)<<" colatitude="<<ranges[1]<<" (deg)";
                if(!ranges[2].empty()) pout(0)<<" longitude="<<ranges[2]<<" (deg)";
                pout(0)<<endl;
            };
            if(legacyBounds.active()) printBounds("",{amr_refine_r_range,amr_refine_colat_range,amr_refine_lon_range});
            for(const auto& entry : numberedBounds) printBounds("."+std::to_string(entry.first),entry.second.range);
        }
        if (spatial_order == 2) {
            const char* name = limiter == 0 ? "unlimited" : limiter == 3 ? "MC (monotonized central)" : "minmod";
            pout(0) << "Spatial reconstruction: Cartesian PLM with " << name
                    << "; state_conversion, spatial_operator, Berta settings, and seam_pressure_correction are unused." << endl;
        }
        pout(0) << "temporal_order: " << temporal_order << endl;
        pout(0) << "divergence_cleaning: " << divergence_cleaning << endl;
        pout(0) << "linear_viscosity: " << get_linear_viscosity() << endl;
        pout(0) << "nonlinear_viscosity: " << get_nonlinear_viscosity() << endl;
        if (uses_second_order_limiter() && (linear_viscosity || nonlinear_viscosity))
            pout(0) << "Artificial viscosity inputs are ignored for spatial_order=2 with minmod or MC; the slope limiter supplies the dissipation." << endl;
        pout(0) << "viscosity_order: " << viscosity_order << endl;
        pout(0) << "scale_viscosity_with_r: " << scale_viscosity_with_r << endl;
        pout(0) << "boundary_reflux: " << boundary_reflux << endl;
        pout(0) << "radial_refinement: " << radial_refinement << endl;
        pout(0) << "MBInterp_define: " << MBInterp_define << endl;
        pout(0) << "BC_file: " << BC_file << endl;
        pout(0) << "Probe_cadence: " << Probe_cadence << endl;
        pout(0) << "num_probes: " << num_probes << endl; 
        for (int i = 0; i < num_probes; ++i) {        
            pout(0) << "  probe" << i << ".filename: " << probe_filenames[static_cast<size_t>(i)] << endl;
            pout(0) << "  probe" << i << ".coords:   " << probe_coords[static_cast<size_t>(i)]   << endl;
        }
        if(num_jmaps)
        {
            pout(0)<<"J-map cadence: "<<Jmap_cadence<<" s; elongation "<<Jmap_min_elongation
                <<" to "<<Jmap_max_elongation<<" deg in "<<Jmap_elongation_bins
                <<" bins; LOS step <= "<<Jmap_los_step<<" AU"<<endl;
            for(const auto& jmap : jmaps)
            {
                pout(0)<<"J-map "<<jmap.filename<<": observer "<<jmap.observer;
                if(jmap.reference.empty()) pout(0)<<", PA "<<jmap.positionAngle<<" deg";
                else pout(0)<<", slit toward "<<jmap.reference;
                pout(0)<<endl;
            }
        }
        pout(0) << "Probe_cadence: " << Probe_cadence << endl;
        pout(0) << "BC_start_time: " << BC_start_time_raw << endl;
        pout(0) << "BC_cadence: " << BC_cadence << endl;
        pout(0) << "BC_frame_rotate: " << BC_frame_rotate << endl;
        pout(0) << "BC_force_single_frame: " << BC_force_single_frame << endl;
        pout(0) << "Sun_gravity: " << Sun_gravity << endl;
        pout(0) << "num_CMEs: " << num_CMEs << endl;
        pout(0) << setw(75) << setfill('-') << "-" << endl;
                
    }

    // Utility function to remove spaces from a string
    string removeSpaces(string str) {
        str.erase(remove(str.begin(), str.end(), ' '), str.end());
        return str;
    }

    // Additional getters and setters as needed
};
