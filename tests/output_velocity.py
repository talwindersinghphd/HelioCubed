"""Check data-file velocities against conserved checkpoints in both output bases.

Requires numpy and h5py. Pass --reference-solver to also check that output writing
does not change the evolved conserved state compared with a previous executable.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess

import h5py
import numpy as np


def read_valid(path):
    with h5py.File(path) as file:
        level = file["level_0"]
        raw = level["data:datatype=0"][:]
        offsets = level["data:offsets=0"][:]
        components = int(file.attrs["num_components"])
        ghost = tuple(int(v) for v in level["data_attributes"].attrs["ghost"])
        patches = []
        for index, box in enumerate(level["boxes"][:]):
            nx, ny, nz = (int(box[f"hi_{d}"]-box[f"lo_{d}"]+1) for d in "ijk")
            patch = raw[offsets[index]:offsets[index+1]].reshape(
                components, nz+2*ghost[2], ny+2*ghost[1], nx+2*ghost[0])
            patches.append(patch[:, ghost[2]:ghost[2]+nz,
                                 ghost[1]:ghost[1]+ny,
                                 ghost[0]:ghost[0]+nx].reshape(components, -1))
        return np.concatenate(patches, axis=1)


def verify(data, conserved, rotating, spherical):
    assert np.isfinite(data).all()
    # Checkpoints are nondimensional Cartesian conserved states, while output
    # coordinates are in AU and primitive velocities are in km/s.
    rho = conserved[0]
    momentum = conserved[1:4]
    velocity = momentum/rho*300.0
    x, y, z = data[:3]
    omega = 2*np.pi/(float(np.float32(25.38))*86400.0)
    if rotating:
        velocity += omega*1.495755e8*np.array([-y, x, np.zeros_like(x)])
    radius = np.sqrt(x*x+y*y+z*z)
    cylindrical = np.hypot(x, y)
    if spherical:
        basis = np.array([
            [x/radius, y/radius, z/radius],
            [x*z/(radius*cylindrical), y*z/(radius*cylindrical), -cylindrical/radius],
            [-y/cylindrical, x/cylindrical, np.zeros_like(x)]])
        velocity = np.einsum("ijn,jn->in", basis, velocity)
    np.testing.assert_allclose(data[4:7], velocity, rtol=1e-10, atol=1e-10)

    pref = 700.0*1.67262192e-24*(3e7)**2
    internal = (conserved[4]-0.5*np.sum(momentum**2, axis=0)/rho
                -np.sum(conserved[5:8]**2, axis=0)/(8*np.pi))
    if conserved.shape[0] == 11:
        internal -= 0.5*conserved[8]
    np.testing.assert_allclose(data[3], rho*700.0, rtol=1e-12)
    np.testing.assert_allclose(data[7], internal*pref*0.5, rtol=1e-9, atol=1e-20)

    # Initial output must recover the specified inertial radial wind, while the
    # checkpoint still contains the rotating velocity needed for a restart.
    return radius


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--reference-solver", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = (root/"exec/inputs").read_text()
    solvers = {"new": args.solver.resolve()}
    if args.reference_solver:
        solvers["reference"] = args.reference_solver.resolve()
    environment = dict(os.environ, OMPI_MCA_btl="self")
    runs = 0
    for rotating in (0, 1):
        for spherical in (0, 1):
            results = {}
            for name, executable in solvers.items():
                work = args.work_dir.resolve()/f"rot{rotating}_sph{spherical}_{name}"
                work.mkdir(parents=True, exist_ok=True)
                settings = {
                    "domainSizex": 8, "domainSizey": 8, "domainSizez": 16,
                    "BoxSize": 4, "r_out": 0.3, "maxStep": 2, "restartStep": 0,
                    "tstop": 100, "gamma": 1.5, "timeIntegratorType": 3,
                    "outputInterval": 1, "CheckpointInterval": 1, "MaxCheckpointFiles": 3,
                    "sph_inner_BC_hdf5": 0, "initialize_in_spherical_coords": 1,
                    "corotating_frame": rotating, "solar_rotation_period": 25.38,
                    "output_in_spherical_coords": spherical,
                    "density_scale": 700, "velocity_scale": 30000000,
                    "probe_cadence": 1e12, "CME_Enter_Time": 3000,
                    "CME_get_checkpoint_before": 0,
                    "Data_file_Prefix": "Data_", "Checkpoint_file_Prefix": "Checkpoint_"}
                config = base
                for key, value in settings.items():
                    config = re.sub(rf"^-{key}\s+.*$", f"-{key} {value}", config, flags=re.M)
                (work/"inputs").write_text(config)
                with (work/"run.log").open("w") as log:
                    subprocess.run([str(executable), "inputs"], cwd=work, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
                runs += 1
                results[name] = work

            for step in (0, 1, 2):
                work = results["new"]
                data = read_valid(work/f"Data_{step}.hdf5")
                state = read_valid(work/f"Checkpoint_{step}.hdf5")
                radius = verify(data, state, rotating, spherical)
                if step == 0:
                    expected = np.array([np.full_like(radius, 500.0),
                                         np.zeros_like(radius), np.zeros_like(radius)])
                    if not spherical:
                        expected = data[:3]/radius*500.0
                    np.testing.assert_allclose(data[4:7], expected, rtol=1e-10, atol=1e-10)
                if "reference" in results:
                    reference = results["reference"]
                    np.testing.assert_array_equal(state, read_valid(reference/f"Checkpoint_{step}.hdf5"))
                    previous = read_valid(reference/f"Data_{step}.hdf5")
                    unchanged = [0, 1, 2, 3]+list(range(7, data.shape[0]))
                    np.testing.assert_array_equal(data[unchanged], previous[unchanged])
                    if not rotating:
                        np.testing.assert_array_equal(data, previous)
            print(f"rotating={rotating}, spherical={spherical}: output velocity, pressure, "
                  "and restart-state checks passed", flush=True)
    print(f"{runs} solver runs passed")


if __name__ == "__main__":
    main()
