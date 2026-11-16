# HelioCubed

<p align="center">
  <img src="https://github.com/user-attachments/assets/97dbd3f7-b4e7-486a-86aa-5984fee436ab" alt="HelioCubed logo" width="300">
</p>

HelioCubed solves the ideal magnetohydrodynamic (MHD) equations on a six-panel
cubed-sphere shell using the [Proto](https://github.com/applied-numerical-algorithms-group-lbnl/proto)
framework. It supports solar-wind and coronal mass ejection (CME) simulations, with MPI parallelism and
adaptive mesh refinement (AMR).

The `CubedSphere` branch includes both uniform-grid and AMR solvers:

- Second-order piecewise linear reconstruction with minmod or MC limiting.
- Fourth-order finite-volume reconstruction with the standard mapped scheme
  ([McCorquodale et al., 2015](https://doi.org/10.1016/j.jcp.2015.01.006)) or
  pointwise Berta WENOZ reconstruction
  ([Berta et al., 2024](https://arxiv.org/abs/2310.11831)).
- Rusanov and Roe eight-wave Riemann solvers.
- Multilevel AMR with conservative prolongation, restriction and refluxing;
  separate radial and angular refinement ratios; optional time subcycling.
- File-driven or analytic solar wind, solar gravity, scheduled FRiED CME insertion,
  trajectory probes, slices, and mapped HDF5 output. CME sources include FRiED,
  GCS, exported deformed GCS, cone, and exported deformed cone volumes.

For high-order refinement background, see
[McCorquodale and Colella (2011)](https://doi.org/10.2140/camcos.2011.6.1),
*A high-order finite-volume method for conservation laws on locally refined grids*,
and [McCorquodale, Ullrich, Johansen, and Colella (2015)](https://doi.org/10.2140/camcos.2015.10.121)
on adaptive multiblock methods for the shallow-water equations on the sphere.

## Build

You need a C++17 compiler, GNU Make, Proto, OpenBLAS (BLAS/LAPACK), an MPI
compiler wrapper, and parallel HDF5 built with the same MPI stack. A single
`exec/GNUmakefile` supplies the MPI build rules for every machine; local
configuration files supply the compiler and dependency settings.

Use Proto's `integrate/master-cubedSphereDev` branch, which includes the
multiblock AMR, mapped HDF5, and checkpoint support used here. Proto is a separate
checkout; it is not downloaded by the HelioCubed makefile.

```sh
git clone --branch CubedSphere https://github.com/dhami1234/HelioCubed.git
git clone --branch integrate/master-cubedSphereDev \
  https://github.com/applied-numerical-algorithms-group-lbnl/proto.git
cd HelioCubed/exec
```

Copy the configuration template and edit the compiler and dependency paths
for your installation:

```sh
cp config.mk.example config.mk
# Edit config.mk, then build:
make -j8
```

`config.mk` and other `exec/config*.mk` files are ignored by Git, so your paths
stay local. Commit only the template and shared build rules. `PROTO` defaults
to `../../proto`, matching the sibling checkout layout above. `HDF5_INC`,
`HDF5_LIB`, `LAPACK_INC`, and `LAPACK_LIB` are directory paths, not linker flags.
Always run make from `exec/`, or use `make -C exec` from the repository root.

The default language standard is C++17. Set `CXXSTD` in your local configuration
or on the make command line to select another standard (for example,
`make CXXSTD=c++20 -B -j8`).

The local configuration is optional when dependencies are supplied through
environment variables or command-line assignments. Command-line assignments
take precedence over the configuration file. Missing configuration is reported
before compilation; `make check-config` checks the Proto layout and required
dependency settings without compiling.

The executable is `exec/cubedSphereTest.exe`. The shared makefile enables
`PR_AMR`; select uniform-grid or adaptive operation at runtime with
`amr_max_level`. Rebuild with `-B` after changing compiler flags or makefiles so
existing objects are recompiled. The makefiles are the supported standalone
build path; the CMake fragments are for integration with a parent build.

## First run

Run from `exec/`. Input files use `-option value`, with comments on separate
lines. Paths in inputs are relative to the directory where the executable runs.
The program accepts either one input filename or command-line options; it does
not combine an input filename with extra command-line overrides.

```sh
mpirun -np 1 ./cubedSphereTest.exe inputs_file     # one MPI rank
mpirun -np 8 ./cubedSphereTest.exe inputs_file     # eight MPI ranks
```

The established input files provide complete configurations:

| Input file | Purpose |
| --- | --- |
| [inputs_SW](exec/inputs_SW) | File-driven solar wind, CME, probes, and AMR |
| [inputs_Test_Problems](exec/inputs_Test_Problems) | Pulse, shock, and blast problems |
| [inputs_convergence](exec/inputs_convergence) | Spatial and space/time convergence studies |

`inputs_SW` starts from initial conditions by default and schedules a CME;
its example `restart_file` setting is commented out. Before running, check the
grid, physical epoch, boundary file, CME settings, and output prefixes.
Uncomment `restart_file` only when the matching checkpoint set is available.
With `init_condition_type=0`,
set `BC_file` to a compatible boundary HDF5 file. The repository includes one
example in `scripts/` and a [SWiG conversion script](scripts/SWiG_to_HelioCubed_h5_conversion.py).
Use UTC dates in **`YYYY-MM-DDTHH:MM:SSZ`** format for `BC_start_time`,
`GCS_fit_time`, and `CME_fit_time`, matching the JSON export's `epoch_utc`.
For example, `2024-03-17T12:00:00Z` means 17 March 2024 at 12:00 UTC.
Input dates also accept fractional seconds or an omitted `Z` (still UTC).
Decimal years and existing day-first input dates remain supported. JSON exports
require the year-first format with `Z`. CME insertion times are calculated from
the fit epoch and propagation settings; calendar times in reports also use `Z`.

## CME shapes and insertion

Set `num_CMEs` and supply indexed `CME0.*`, `CME1.*`, ... blocks. Sources are
sorted by start time. Existing FRiED/GCS input blocks remain supported.

| `CME_type` | Shape and magnetic prescription | Required model inputs |
| --- | --- | --- |
| 0 | Disabled | None |
| 1 | FRiED | Existing `GCS_*` and flux inputs |
| 2 | GCS with smooth joins | Existing `GCS_*` and flux inputs |
| 3 | Deformed GCS from dGCS | `CME_shape_file`, `CME_magnetic_file`, `CME_speed` |
| 4 | Finite spherical cone cloud | `Cone_height`, `Cone_half_angle`, `Cone_latitude`, `Cone_longitude`, `CME_fit_time`, `CME_speed` |
| 5 | Expanding deformed cone from dGCS | `CME_shape_file`, `CME_speed` |

Use `CME_injection_method volume` for insertion into owned cells, followed by
optional inner-boundary feeding. Use `surface` (or its alias `boundary`) to
introduce the source through the inner spherical boundary. Types 1–3 and 5
expand with a boundary taper; type 4 passes through as a finite spherical cloud.
Surface insertion and boundary feeding use the standard second-order operator
(`spatial_order=2`, `spatial_operator=1`), on uniform grids or AMR. Fourth-order
volume insertion is supported; its boundary feeding is inactive.

For **deformed GCS**, export **both** shapes and magnetic fields in dGCS and
select the two JSON files from the same member folder:

```text
-num_CMEs                       1
-CME0.CME_type                  3
-CME0.CME_shape_file            /path/to/member_0000/shape.json
-CME0.CME_magnetic_file         /path/to/member_0000/magnetic_field.json
-CME0.CME_speed                 900
-CME0.CME_injection_method      surface
-CME0.DBM_target_height         70
-CME0.DBM_gamma                 1e-8
-CME0.DBM_wind_speed            450
-CME0.CME_density               1.38e-23
-CME0.CME_earth_trajectory      trajEarth.dat
```

These files define the complete continuous volume and field, including the
Bernstein coefficients, front/legs winding ratio, flux magnitudes, handedness
and axial polarity. The C++ reader independently recomputes the whole-box
invertibility certificate, inverts the volume map at each requested point, and
evaluates the analytic Piola field. Shape/field references, coefficients,
rotation, epoch and units must agree. Both uniform and front-weighted magnetic
exports are supported, including zero poloidal flux. The NPZ, VTK and sampled
field-line files are not needed by the solver. Do not supply separate flux or
magnetic-sign overrides for an imported source.

The exported **whole-rope** magnetic fluxes are preserved; they are not converted
to the computational-domain clipped poloidal-flux convention of types 1/2.
Classic GCS cone/front joins from the export are retained, including tangential
field jumps; the importer does not substitute HelioCubed's smoothed GCS model.

For a **parameter-only cone**, replace that entire CME block with, for example:

```text
-num_CMEs                       1
-CME0.CME_type                  4
-CME0.Cone_height               15
-CME0.Cone_half_angle           35
-CME0.Cone_latitude             0
-CME0.Cone_longitude            0
-CME0.CME_fit_time              2024-03-17T12:00:00Z
-CME0.CME_speed                 900
-CME0.CME_injection_method      surface
-CME0.DBM_target_height         70
-CME0.DBM_gamma                 1e-8
-CME0.DBM_wind_speed            450
-CME0.Cone_reference_density    200
-CME0.Cone_reference_temperature 1.5e6
-CME0.Cone_density_factor       4
-CME0.Cone_temperature_factor   1
-CME0.CME_plasma_beta           0
-CME0.CME_earth_trajectory      trajEarth.dat
```

The type-4 fit parameters define an ENLIL-style **spherical plasma cloud**.
`Cone_height` is the fitted nose height in nominal solar radii (695700 km).
`Cone_half_angle` is the maximum angular half-width of the footprint on the
inner boundary; latitude/longitude are Stonyhurst degrees. With inner radius
`R`, the cloud radius is `a=R*sin(half_angle)`. Its center moves along the cone
axis from `R-a` at first contact to `R+a` at final clearance. The resulting
passage lasts `2*a/V_cloud`, with the prescribed uniform radial cloud speed.
This follows the finite homogeneous-cloud boundary approach described by
[Mays et al. (2015), Section 2](https://arxiv.org/pdf/1504.04402#page=3),
using HelioCubed's MHD and boundary solver rather than ENLIL's Fortran code.

For a **deformed cone**, use type 5 with its exported `shape.json`, `CME_speed`,
the chosen insertion method, and the same plasma/DBM/trajectory options above.
Also set `CME_boundary_taper_time` for its smooth boundary fade (default 43200 s).
Its fit epoch and complete geometry come from the export. The fitted/deformed
lemniscate remains rooted and expands homologously with the DBM leading radius.
It retains its exported shape and the existing taper prescription. Do not supply
a magnetic file.
Because type 4 uses a spherical-cloud assumption and type 5 retains the full
export, even an undeformed exported lemniscate has different geometry from type 4.

For types 3–5, the fitted speed is the **leading-radius speed** in km/s. The importer
numerically refines the fitted maximum surface radius, converts the export's
`solar_radius_cm` to physical cm, and propagates that radius with the DBM to
`DBM_target_height` for types 3 and 5 (in the solver's 696000-km solar radii).
For type 4, the fitted radius is the specified nose height; DBM determines the
first boundary arrival time and entry speed, then the finite spherical template
translates at that constant speed. Its volume snapshot has its nose at
`DBM_target_height`. This initializes a homogeneous cloud rather than an
ENLIL-evolved state at that radius. The target must be at least the fitted leading radius.
Stonyhurst orientation is converted at the
fit/export epoch using the Earth trajectory, as for the existing GCS driver.
Use an early enough `BC_start_time` to include the entire surface emergence.

Types 1–3 and 5 use the existing homologous DBM source evolution:
positions scale with the leading radius, radial velocity is `V_lead*r/H`, and
an imported magnetic field scales as expansion factor to the power −2.
The [input templates](exec/inputs_SW) share only the DBM settings below the
global source count; each model block includes its own type, geometry,
insertion, plasma settings, and taper where applicable.

For magnetic models 1–3, `CME_density` is the minimum **added** density at the
reference insertion size. The analytic source density scales as expansion
factor to the power −3. Boundary feeding uses the supplied density. Set
`CME_temperature` in K for the added material; zero/omitted retains the local
ambient temperature. For positive temperature, `p_added=2*rho_added*kB*T/mp`.
`CME_plasma_beta` adds pressure relative to the intrinsic CME magnetic pressure.

For cone models 4/5, the ENLIL-style prescription sets the **total** cloud state
at the physical inner boundary:

```text
n_cloud = Cone_reference_density * Cone_density_factor
T_cloud = Cone_reference_temperature * Cone_temperature_factor
rho_cloud = mp * n_cloud
p_cloud = 2 * n_cloud * kB * T_cloud
```

Defaults are a reference proton density of **200 cm⁻³**, temperature of
**1.5 million K**, density factor **4**, and temperature factor **1**. Thus the
full-strength inlet has total `n=800 cm^-3`, `T=1.5e6 K`, and
`rho=1.3381e-21 g/cm^3`. Pressure assumes fully ionized hydrogen with equal
electron and proton temperatures. These reference values correspond to the
documented `a8b1` configuration in the
[NASA/NOAA ENLIL validation report, Table 3](https://ccmc.gsfc.nasa.gov/static/files/CCMC_SWPC_annex_final_report.pdf#page=6);
the cloud factors follow the prescription in
[Mays et al. (2015)](https://arxiv.org/pdf/1504.04402#page=4).
They are calibrated boundary values, and the input permits changing all four.

The cone state replaces the background plasma inside its mask, including when
the target density or pressure is smaller; it does not add another cloud
density on top. Ambient B is retained without an Alfvén density-loading floor.
The rope inputs `CME_density` and `CME_temperature` are rejected for cones;
`CME_plasma_beta` must be zero because there is no intrinsic CME magnetic field.
Type 4 carries the specified uniform total density, temperature and radial
speed while its sphere covers a boundary patch, and background values otherwise.
Type 5 holds its boundary forcing at full strength for the first half of the
configured duration, then blends total density and pressure back to the
background with a quintic taper; its excess velocity fades with the same weight.
Ambient B is retained throughout.

For **type-4 volume insertion**, cells inside the translated sphere snapshot
receive the same homogeneous plasma prescription. Any part still crossing the
inner boundary is fed only until its trailing edge clears. This partial-cloud
volume insertion requires the standard second-order operator; fourth-order
volume insertion requires a placement after the sphere clears the boundary.
Once inserted, the plasma evolves under MHD rather than remaining rigid or
being forced to retain the template shape.
For **type-5 volume insertion**, the exported volume is scaled to the reference
leading height. The radial initialization profiles are
`n=n_boundary*(r_inner/r)^3` and
`T=T_boundary*(r_inner/r)^(3*(gamma-1))`, with homologous velocity `V_lead*r/H`.
These are expanding-source approximations, rather than ENLIL's uniform sphere.
Both cone models prescribe conservative cell states at both evolution
orders, avoiding negative pressure from high-order averaging across the sharp
shape boundary. A fourth-order run retains its fourth-order evolution scheme.

For types 1–3 and 5, `CME_boundary_taper_time` (default 43200 s) is the **total**
boundary-forcing duration measured from insertion/emergence start. For its first
half the taper weight is exactly 1. During its second half the weight falls
smoothly to zero with zero first and second derivatives at both ends of the
fade. Thus 43200 s gives 6 hours at full strength followed by a 6-hour fade.
The timing uses elapsed injection time; it does not measure a geometric halfway
point of the rope. This weight multiplies added mass, CME field and velocity
perturbations for magnetic ropes, or blends total plasma states for type 5.
Zero disables volume boundary feeding; surface insertion requires a
positive duration. Only type 4 rejects this input: its passage is set by the
finite sphere's geometry and speed, including on restart.

`CME_resolved.inputs` and `CME_conversion_report.txt` record the resolved
schedule, scaling and model details. Restart with the same source files and
source ordering; the saved insertion count prevents duplicate volume insertion.

## Adaptive mesh refinement

`amr_max_level=0` uses the uniform-grid driver. A positive value enables AMR:
`1` allows levels L0 and L1, `2` allows L0 through L2, and so on.

| Spatial method | AMR time integrators | Required settings |
| --- | --- | --- |
| O2 | `temporal_order=2` or `3` | `boundary_reflux=1`; artificial viscosity off |
| O4 standard or Berta | `temporal_order=3` or `4` | `boundary_reflux=1`, `state_conversion=1` |

AMR base dimensions must be at least 12 cells and divisible by their patch
sizes, which must be at least 8. O4 additionally requires at least 20 angular
cells per panel. `domainSize` is the angular cell count **on each panel**;
`thickness` is the radial cell count. The base grid has
`6 * domainSize * domainSize * thickness` cells.

To make the first-run example adaptive, change its `amr_max_level` to `1` and
add the following settings:

```text
-amr_ref_ratio_rad          2
-amr_ref_ratio_nonrad       1
-amr_subcycling            1
-amr_regrid_interval       4
-amr_regrid_time_cadence    3600
-amr_buffer                2
-amr_verbosity             1
-amr_refine_density_thresh -1
-amr_refine_pressure_thresh -1
-amr_refine_B_thresh       -1
-amr_refine_V_thresh       0.2
```

Each ratio can be one integer for every interface or a comma-separated list
with one entry per interface. For example, `amr_max_level=2`,
`amr_ref_ratio_rad=2,4`, and `amr_ref_ratio_nonrad=1,2` refine radially by 2
then 4, and angularly by 1 then 2. Ratios such as 2, 4, and 8 are supported;
1 leaves a direction unchanged. Every interface must refine at least one
direction, and ratios must allow aligned fine patches to tile the domain.

Fine patches can have different cell counts from base patches. Set
`amr_boxSize_rad` and `amr_boxSize_nonrad` to one value for every fine level
or a comma-separated list. Zero retains the existing automatic patch sizing.
Positive sizes must be at least 8 cells, divisible by that level's refinement
ratio, and tile both the refined domain and a refined parent patch; a refined
patch must contain at least two parent cells in each refined direction.

For a base grid of 45 angular by 120 radial cells, base patches of 15 angular
by 30 radial cells, and refinement ratios 2 angular / 4 radial, use:

```text
-amr_boxSize_rad            20
-amr_boxSize_nonrad         0
```

This divides a 60×30×30 fine patch into three patches of 20×30×30. Setting
`amr_boxSize_nonrad=10` instead creates 27 patches of 20×10×10 and spreads very
sparse fine work across more ranks. Smaller patches increase ghost-cell and
message overhead, so the best sizes depend on the number of ranks and fine
patches. Automatic tagging may cover a different region when the patch
dimensions change; fixed regions that tile both layouts allow a comparison
with identical refined cells. Patch dimensions are printed in the AMR grid
summary.

With `amr_subcycling=0`, all levels share the finest required timestep. With
`1`, finer levels take extra steps only when their CFL limits require them.
RK3/RK4 subcycling uses extrapolated SSPRK2 advances, which adds work; its
performance depends on the refined volume and the level timestep limits.

Refinement uses the relative-gradient magnitude of density, pressure,
magnetic-field magnitude, or speed. For an interior cell, each directional
contribution is `(q_plus - q_minus) / (q_plus + q_minus)`; the criterion uses
the Euclidean norm of those contributions. Physical radial boundaries use
one-sided differences. Thus, a threshold of `0.2` is a relative-gradient test,
not a direct 20% difference from each adjacent cell.

Any enabled criterion can tag a cell. Negative thresholds disable a criterion;
zero tags any nonzero gradient. The corresponding
`amr_refine_<variable>_maxlevel` options (`density`, `pressure`, `B`, `V`) limit
the finest child level: `-1` uses `amr_max_level`, and `0` disables that
criterion. Tags grow by `amr_buffer` parent cells and allocate whole fine
patches. Large patches can therefore refine much more volume than the tagged
cells alone suggest.

Optional physical bounds restrict where cells can trigger refinement:

```text
-amr_refine_r_range.0       1.4964e12,7.482e12
-amr_refine_colat_range.0   60,120
-amr_refine_lon_range.0     350,20
-amr_refine_r_range.1       7.482e12,1.61124e13
-amr_refine_colat_range.1   70,110
-amr_refine_lon_range.1     170,200
```

Radius is in cm. Colatitude is in degrees from the north pole: 0 north,
90 equator, 180 south. Longitude is in degrees in the simulation's Cartesian
frame: 0 along +X, 90 along +Y. These are fixed directions in that frame;
boundary-frame rotation does not rotate the bounds. Longitudes are normalized
modulo 360, so negative values work and `350,20` crosses 0. A span of at least
360 degrees, such as `0,360` or `-180,180`, allows every longitude; equal
endpoints select a single meridian. Longitude does not restrict either pole.

Ranges with the same suffix form one region. A cell's center must satisfy
all supplied bounds of at least one region to tag; overlapping regions do
not duplicate refinement. Omit a range to leave that coordinate unrestricted
within its region. Region indices are nonnegative integers, may have gaps,
and are independent of AMR levels. The original unnumbered names still work
for one region; use either numbered or unnumbered bounds, without mixing.
The bounds apply to every AMR level and to both variable-dependent criteria
and `amr_refine_regions`;
they restrict refinement without forcing it. Buffering, whole-patch allocation
and nesting support can extend the actual refined mesh beyond the bounds.
Changing bounds on restart takes effect at the next regrid; the saved mesh
is restored first. Commented examples are included in the established inputs.

Regridding occurs when either cadence is reached: `amr_regrid_interval` counts
base steps, and `amr_regrid_time_cadence` measures elapsed simulation seconds.
Zero disables that cadence. Time-based regrids occur at the first synchronized
coarse step crossing a cadence boundary. Verbosity 1 prints level timesteps,
regrid duration, and coverage as percentages of base-grid volume, including
regrid attempts that leave the hierarchy unchanged; 2 adds detailed advance
and reflux events, while 0 retains ordinary iteration progress.

`P_floor_cadence` also controls AMR density repair. Density uses the existing
`RHO00` floor and pressure has a zero lower bound. Both are checked before
regridding so newly refined parents are admissible; repairs are reported when
they occur.

## Output, VisIt, and restart

AMR volume output is a native multilevel HDF5 hierarchy:

```text
data_amr_0.hdf5
data_amr_0.map.hdf5
```

Keep each data file beside its matching map file. Open the **data** file in
VisIt using its Chombo reader. In the reader options, enable checking for a
mapping file and disable enabling only the root level. For the physical shell,
apply a Displace operator with `_mapping_displacement` and factor 1 to each
plot; the reader creates that expression from the map file. These settings are
supported by the [VisIt Chombo reader](https://github.com/visit-dav/visit/blob/develop/src/databases/Chombo/avtChomboFileFormat.C).

Plot `amr_level` to see which levels exist; `active=1` identifies cells not
covered by finer data. To inspect the refined mesh, enable all available levels
in the Subset/SIL selection and raise any MultiresControl resolution setting to
the finest level before adding a Mesh plot. A file may contain only L0 if no
criterion has tagged cells.

`output_coordinates` selects the vector components written in **both volume
files and slices**, with the same behavior for uniform grids and AMR:

| Setting | Velocity fields | Magnetic fields |
| --- | --- | --- |
| `0` (default) | `Vr`, `Vt`, `Vp` | `Br`, `Bt`, `Bp` |
| `1` | `Vx`, `Vy`, `Vz` | `Bx`, `By`, `Bz` |

Spherical components use the orthonormal radial, colatitude, and azimuthal
directions. Both choices also write `density` and `P`; AMR volumes add `active`
and `amr_level`. This setting controls output independently of the solver's
`state_conversion` setting. The mesh geometry remains the physical shell.

`scale_data_with_r=0` writes raw CGS values in either coordinate basis. With
`1`, write density multiplied by `(r/AU)^2/c_MP`, pressure by
`(r/AU)^(2*gamma)*1e12`, and velocity in km/s. Magnetic fields are converted to
microgauss with radial components multiplied by `(r/AU)^2` and transverse
components by `r/AU`. For Cartesian output this magnetic scaling is applied
in the spherical basis before rotating back to Cartesian components.
`output_divB=1` always adds physical `divB` in G/cm.

`slices` selects any combination of `X Y Z R`; AMR slices omit covered coarse
cells. Checkpoints retain conservative state. Probes sample the finest available
level at each trajectory position. The bundled solar-wind slice plotting
defaults use spherical field names.

Probes write orthonormal RTN velocity components (`VR`, `VT`, `VN`) and magnetic
components (`BR`, `BT`, `BN`) for both uniform-grid and AMR runs, independently
of the volume/slice `output_coordinates` setting. For plotting in spherical
components, use `Vr=VR`, `Vtheta=-VN`, and `Vphi=VT`, with the same mapping for B.
Theta is colatitude, increasing southward.
Each headerless probe row contains decimal year, radius (AU), latitude (degrees),
simulation longitude (degrees), number density (cm^-3), three velocity components
(km/s), pressure (picodyne/cm^2), and three magnetic components (microgauss).
Probes do not apply the volume/slice radial scaling.

Slices from both uniform-grid and AMR runs are self-contained **Tecplot ASCII
`.dat` files**, for example `data.z.100.dat` or `data.r.100.dat`. They include
physical coordinates, mesh connectivity, and cell-centered primitive fields.
`X`, `Y`, and `Z` select the planes x=0, y=0, and z=0; `R` selects the spherical
shell at radius `1.5e13` cm. Slice cadence is controlled by `slice_cadence` and
`slice_time_cadence`.

- **Tecplot:** use **File > Load Data**, choose the Tecplot data loader, and
  open the slice `.dat` file. Enable contours for a field such as `density` or
  `P`, and mesh lines to inspect the refinement. See the
  [Tecplot loading guide](https://tecplot.azureedge.net/products/360/current/help/users_manual/data-loading.html).
- **ParaView:** use **File > Open**, select the slice file or file series,
  choose **Tecplot Reader** if prompted, and click **Apply**. Color by a cell
  field and use **Surface With Edges** to inspect the mesh. ParaView's
  [Tecplot Reader](https://www.paraview.org/paraview-docs/v5.13.0/python/paraview.simple.TecplotReader.html)
  supports multiple zones, block-packed cell data, and file series.

An AMR restart checkpoint is a **set** of files:

```text
checkpoint_amr_100.amr
checkpoint_amr_100_L0.hdf5
checkpoint_amr_100_L1.hdf5
```

Keep the manifest and every matching `_L*.hdf5` file together, and set:

```text
-restart_file checkpoint_amr_100.amr
```

Keep the original base grid, patch sizes, shell geometry, gamma,
`init_condition_type`, configured maximum AMR level, and refinement ratios.
New checkpoints also save the fine patch sizes. Zero-valued fine-size inputs
restore those saved dimensions; explicit positive sizes must match them.
For checkpoints written before fine-size support, keep both fine-size inputs
at zero to retain the original automatic layout.
The restart restores the saved hierarchy, iteration, simulation time, and CME
insertion state. Set stopping limits beyond the saved step/time to continue.
Plot files are for visualization; restart uses the checkpoint manifest and
its conservative-state files. `checkpoint_cadence` controls checkpoint output,
and `max_checkpoint_files` controls retention of sets written during the run.

## Convergence and analysis

The same convergence driver supports uniform grids and AMR:

- `convTestType=1`: refine space while holding the timestep fixed.
- `convTestType=2`: refine space and time together.
- `convTestType=3`: compare initialized states without advancing.
- `convTestType=4`: initialize one grid without advancing.

For modes 1/2, `convergence_level=-1` runs all three study resolutions;
`0`, `1`, or `2` runs one study, and `3` compares saved studies.
`convergence_dt` sets a positive fixed timestep or uses `-1` for automatic
selection. For split mode-2 runs, halve the prescribed timestep and increase
the step limit for each successive study to reach the same physical time.
AMR studies compare conservative composite projections onto each study's
base mesh. Study indices are separate from AMR levels inside each hierarchy.

[run_convergence_test.sh](scripts/run_convergence_test.sh) automates split
studies, and [run_test_problems.sh](scripts/run_test_problems.sh) runs idealized
cases and plots slices. Edit their configuration sections or documented
environment overrides before launching; the test-problem script replaces
existing result folders for the selected cases. With AMR, choose a supported
time integrator in the script settings and the required O4 conversion setting
in the input template. Their `_anvil.sh` variants handle batch runs.

Python plotting dependencies are listed in
[requirements-plotting.txt](scripts/requirements-plotting.txt). Timing analysis
scripts in `scripts/` process Proto `.time.table` reports; the AMR report is
`CubedSphere_AMR.time.table`.
