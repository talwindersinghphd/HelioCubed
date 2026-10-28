#!/usr/bin/env python3
"""Synthetic white-light J-maps from HelioCubed diagnostics or saved outputs.

Examples (all times UTC)::

  python scripts/synthetic_jmap.py native StereoA_jmap.csv --output jmaps/StereoA
  python scripts/synthetic_jmap.py slices RUN --observer RUN/trajStereoA.dat \
      --reference RUN/trajEarth.dat --equatorial --output RUN/jmaps/StereoA_equatorial
  python scripts/synthetic_jmap.py volumes RUN --observer RUN/trajStereoA.dat \
      --output RUN/jmaps/StereoA_3D

Outputs include the original MS-FLUKSS LOS-mean ne*(r/AU)^2 density proxy and
finite-disk Thomson brightness. --reference selects the Sun-observer-reference
plane used by MS-FLUKSS (use the Earth trajectory for STEREO J-maps); otherwise
--position-angle selects a slit measured from solar north through solar east.
The native diagnostic uses the full 3D state, including finest AMR coverage.
Saved volumes currently support uniform-grid primitive plot files only. Saved
Z slices require explicit --equatorial: observer and reference are projected to
latitude zero; without a reference, PA must be 90 or 270 degrees. These slices
cannot recover off-plane structure.

Kernel: Xiong et al., doi:10.1007/s11207-012-0047-0, equations 1--6.
Brightness is relative to disk-center solar radiance, with u=0.63 and ne=rho/mp.
Only the modeled shell contributes; no exterior extrapolation, light-travel-time
correction, instrument response, or artificial temporal interpolation is added.
"""
import argparse
import datetime as dt
import json
import io
from pathlib import Path
import re

import numpy as np

AU = 14959787070000.0
RS = 69600000000.0
MP = 1.67262192e-24


def read_inputs(path):
    values = {}
    for line in path.read_text().splitlines():
        row = line.split("#", 1)[0].split()
        if len(row) >= 2 and row[0].startswith("-"):
            values[row[0][1:]] = row[1]
    return values


def decimal_datetime(epoch):
    year = int(epoch)
    start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    end = start.replace(year=year + 1)
    return start + (end - start) * (epoch - year)


def start_datetime(raw):
    try:
        return decimal_datetime(float(raw))
    except ValueError:
        return dt.datetime.strptime(raw, "%d-%m-%YT%H:%M:%S").replace(tzinfo=dt.timezone.utc)


def trajectory_position(path, dates, equatorial=False):
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[1] != 5 or len(rows) < 2 or not np.isfinite(rows).all():
        raise ValueError("Trajectory needs finite year, DOY, radius_AU, latitude, longitude rows")
    times = np.array([(dt.datetime(int(y), 1, 1, tzinfo=dt.timezone.utc)
                       + dt.timedelta(days=int(d) - 1)).timestamp() for y, d in rows[:, :2]])
    if np.any(np.diff(times) <= 0) or np.any(rows[:, 2] <= RS / AU) or np.any(abs(rows[:, 3]) > 90):
        raise ValueError("Invalid trajectory times, radius, or latitude")
    target = np.array([date.timestamp() for date in dates])
    if target.min() < times[0] or target.max() > times[-1]:
        raise ValueError("Observer trajectory does not cover the requested frames")
    radius = np.interp(target, times, rows[:, 2]) * AU
    latitude = np.deg2rad(np.interp(target, times, rows[:, 3]))
    longitude = np.interp(target, times, np.unwrap(np.deg2rad(rows[:, 4]))) + np.deg2rad(181.02)
    if equatorial:
        latitude[:] = 0
    return radius[:, None] * np.column_stack((np.cos(latitude) * np.cos(longitude),
                                            np.cos(latitude) * np.sin(longitude), np.sin(latitude)))


def thomson_kernel(radius, cosine, limb=0.63):
    radius = np.asarray(radius)
    if np.any(radius <= RS) or not np.isfinite(radius).all():
        raise ValueError("Thomson scattering needs radius above the photosphere")
    s = RS / radius
    q = s * s
    c = np.sqrt(1 - q)
    a = c * q
    cc = q / (1 + c) * (1 + (1 + c + c*c) / 3)
    small = s < .1
    b = np.empty_like(q)
    d = np.empty_like(q)
    qs = q[small]
    b[small] = qs * (2/3 + qs * (-4/15 + qs * (-2/35 + qs * (-8/315 - qs*10/693))))
    d[small] = qs * (2/3 + qs*qs * (2/105 + qs * (4/315 + qs*2/231)))
    f = (1 - q[~small]) * np.arctanh(s[~small]) / s[~small]
    b[~small] = (-1 + 3*q[~small] + f*(1 + 3*q[~small])) / 8
    d[~small] = (5 + q[~small] - f*(5 - q[~small])) / 8
    tangent = (1 - limb) * cc + limb * d
    polarized = np.maximum(0, 1 - cosine*cosine) * ((1 - limb)*a + limb*b)
    return np.column_stack((2*tangent - polarized, polarized)) * (np.pi * 7.95e-26 / 2)


def ray_samples(observer, elongations, pa, inner, outer, step, reference=None):
    radius = np.linalg.norm(observer)
    lon = np.arctan2(observer[1], observer[0])
    lat = np.arcsin(observer[2] / radius)
    north = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    east = np.array([np.sin(lon), -np.cos(lon), 0])
    p = np.deg2rad(pa)
    tangent = np.cos(p)*north + np.sin(p)*east
    if reference is not None:
        tangent = reference - (reference @ observer)/(radius*radius)*observer
        norm = np.linalg.norm(tangent)
        if norm < 1e-10*np.linalg.norm(reference):
            raise ValueError("Observer and reference are collinear with the Sun")
        tangent /= norm
    points, weights, bins = [], [], []
    for i, angle in enumerate(np.deg2rad(elongations)):
        direction = -np.cos(angle)*observer/radius + np.sin(angle)*tangent
        along = observer @ direction
        impact2 = max(0., radius*radius - along*along)
        if impact2 >= outer*outer:
            continue
        root = np.sqrt(outer*outer - impact2)
        low, high = max(0., -along - root), -along + root
        if impact2 < RS*RS and -along > 0:
            high = min(high, -along - np.sqrt(RS*RS - impact2))
        if high <= low:
            continue
        if impact2 < inner*inner:
            hole = np.sqrt(inner*inner - impact2)
            segments = [(low, min(high, -along - hole)), (max(low, -along + hole), high)]
        else:
            segments = [(low, high)]
        for start, end in segments:
            if end <= start:
                continue
            count = max(1, int(np.ceil((end - start)/step)))
            ds = (end - start) / count
            distance = start + (np.arange(count) + .5)*ds
            point = observer[None, :] + distance[:, None]*direction[None, :]
            r = np.linalg.norm(point, axis=1)
            cosine = -(point @ direction) / r
            points.append(point)
            weights.append(np.column_stack((thomson_kernel(r, cosine)*ds, (r/AU)**2*ds,
                                             np.full(count, ds))))
            bins.append(np.full(count, i, dtype=int))
    if not points:
        raise ValueError("No rays intersect the simulated shell")
    return np.concatenate(points), np.concatenate(weights), np.concatenate(bins)


class Shell:
    def __init__(self, inputs):
        self.inner = float(inputs["inner_radius"])
        self.outer = float(inputs["outer_radius"])
        self.stretch = float(inputs["radial_stretch"])
        self.nr = int(inputs["thickness"])
        self.na = int(inputs["domainSize"])
        self.scaled = int(inputs.get("scale_data_with_r", 0)) == 1
        if not (RS < self.inner < self.outer) or min(self.nr, self.na) <= 0:
            raise ValueError("Invalid shell geometry")
        xi = (np.arange(self.nr) + .5) / self.nr
        self.radii = self.inner + (self.outer - self.inner) * (
            np.expm1(self.stretch*xi)/np.expm1(self.stretch) if self.stretch else xi)

    def radial_coordinate(self, radius):
        fraction = (radius - self.inner)/(self.outer - self.inner)
        return np.log1p(fraction*np.expm1(self.stretch))/self.stretch if self.stretch else fraction

    def radial_cells(self, radius):
        return np.clip(np.floor(self.radial_coordinate(radius)*self.nr).astype(int), 0, self.nr - 1)

    def physical_density(self, density, radial_axis=-1):
        if not self.scaled:
            return density / MP
        shape = [1]*density.ndim
        shape[radial_axis] = self.nr
        return density * (AU/self.radii.reshape(shape))**2


def read_slice(path, shell):
    text = path.read_text()
    time = float(re.search(r"t=([\deE+.-]+)", text).group(1))
    nvar = len(re.findall(r'"[^"\n]+"', re.search(r"^VARIABLES=(.*)$", text, re.M).group(1))) - 3
    density = np.full((4*shell.na, shell.nr), np.nan)
    count = np.zeros_like(density, dtype=int)
    for zone in re.split(r"^ZONE ", text, flags=re.M)[1:]:
        header, raw = zone.split("\n", 1)
        ni = int(re.search(r"\bI=(\d+)", header).group(1))
        nj = int(re.search(r"\bJ=(\d+)", header).group(1))
        nn, nc = ni*nj, (ni - 1)*(nj - 1)
        data = np.fromstring(raw.split("DATASETAUXDATA", 1)[0], sep=" ")
        if data.size != 3*nn + nvar*nc:
            raise ValueError(f"Malformed slice zone: {path}")
        xyz = data[:3*nn].reshape(3, nj, ni)
        if np.max(abs(xyz[2])) > shell.outer*1e-8:
            raise ValueError("Slice is not the solar-equatorial Z=0 plane")
        centers = .25*(xyz[:, 1:, 1:] + xyz[:, :-1, 1:] + xyz[:, 1:, :-1] + xyz[:, :-1, :-1])
        radial = shell.radial_cells(np.linalg.norm(centers, axis=0))
        longitude = np.mod(np.arctan2(centers[1], centers[0]), 2*np.pi)
        angular = np.floor(longitude/(2*np.pi)*(4*shell.na)).astype(int) % (4*shell.na)
        density[angular, radial] = data[3*nn:3*nn + nc].reshape(nj - 1, ni - 1)
        count[angular, radial] += 1
    if not np.all(count == 1) or not np.isfinite(density).all() or np.any(density < 0):
        raise ValueError(f"Incomplete, overlapping, or invalid equatorial density: {path}")
    return time, shell.physical_density(density)


def read_volume(path, shell):
    import h5py
    density = np.full((6, shell.na, shell.na, shell.nr), np.nan)
    with h5py.File(path, "r") as file:
        if int(file.attrs.get("num_levels", 1)) != 1:
            raise ValueError("Archived AMR volumes are not supported; use the native finest-level J-map diagnostic")
        level = file["level_0"]
        time = float(level.attrs["time"])
        components = int(file.attrs["num_components"])
        if components not in (8, 9):
            raise ValueError("Expected a uniform-grid primitive plot file, not a checkpoint or map")
        boxes = level["boxes"][:]
        blocks = level["blocks"][:]
        offsets = level["data:offsets=0"][:]
        flat = level["data:datatype=0"]
        for i, box in enumerate(boxes):
            lo, hi = np.array(tuple(box)[:3]), np.array(tuple(box)[3:])
            block = next((b for b, bounds in enumerate(blocks)
                          if all(lo >= np.array(tuple(bounds)[:3])) and all(hi <= np.array(tuple(bounds)[3:]))), None)
            if block is None:
                raise ValueError("Plot box is outside its declared panel")
            shape = hi - lo + 1
            cells = int(np.prod(shape))
            if offsets[i + 1] - offsets[i] != components*cells:
                raise ValueError("Ghosted archived volumes are not supported")
            values = flat[int(offsets[i]):int(offsets[i]) + cells].reshape(tuple(shape[::-1]))
            local = lo - np.array(tuple(blocks[block])[:3])
            slices = tuple(slice(int(a), int(a + n)) for a, n in zip(local[::-1], shape[::-1]))
            density[(block,) + slices] = values
    if not np.isfinite(density).all() or np.any(density < 0):
        raise ValueError(f"Incomplete or invalid volume density: {path}")
    return time, shell.physical_density(density)


def sample_density(density, points, shell):
    radial = shell.radial_cells(np.linalg.norm(points, axis=1))
    if density.ndim == 2:
        phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2*np.pi)
        angular = np.floor(phi/(2*np.pi)*(4*shell.na)).astype(int) % (4*shell.na)
        return density[angular, radial]
    # Same panel ownership and inverse map as the native diagnostic.
    major = np.argmax(np.abs(points[:, [2, 0, 1]]), axis=1)
    block = np.choose(major, (np.where(points[:, 2] >= 0, 1, 0),
                             np.where(points[:, 0] >= 0, 3, 5), np.where(points[:, 1] >= 0, 4, 2)))
    x, y, z = points.T
    alpha = np.empty(len(points))
    beta = np.empty(len(points))
    pairs = ((x, -z, -y), (x, z, y), (x, -y, z), (y, x, z), (-x, y, z), (-y, -x, z))
    for b, (first, denominator, second) in enumerate(pairs):
        select = block == b
        alpha[select] = np.arctan2(first[select], denominator[select])
        beta[select] = np.arctan2(second[select], denominator[select])
    j = np.clip(np.floor((alpha/(np.pi/2) + .5)*shell.na).astype(int), 0, shell.na - 1)
    k = np.clip(np.floor((beta/(np.pi/2) + .5)*shell.na).astype(int), 0, shell.na - 1)
    return density[block, k, j, radial]


def integrate_frame(density, observer, shell, elongations, pa, step, reference=None):
    points, weights, bins = ray_samples(observer, elongations, pa, shell.inner, shell.outer, step, reference)
    ne = sample_density(density, points, shell)
    result = [np.bincount(bins, weights=ne*weights[:, c], minlength=len(elongations)) for c in range(3)]
    length = np.bincount(bins, weights=weights[:, 3], minlength=len(elongations))
    result[2] = np.divide(result[2], length, out=np.zeros_like(length), where=length > 0)
    return np.column_stack(result)


def load_native(path):
    text = "".join(line for line in path.read_text().splitlines(keepends=True) if not line.startswith("#"))
    rows = np.atleast_1d(np.genfromtxt(io.StringIO(text), delimiter=",", names=True))
    times = np.unique(rows["time_s"])
    elongations = np.unique(rows["elongation_deg"])
    brightness, polarized, proxy, epochs = [], [], [], []
    for time in times:
        frame = rows[rows["time_s"] == time]
        frame = frame[np.argsort(frame["elongation_deg"])]
        if len(frame) != len(elongations) or not np.array_equal(frame["elongation_deg"], elongations):
            raise ValueError("Incomplete native J-map frame")
        brightness.append(frame["B_Bsun"])
        polarized.append(frame["pB_Bsun"])
        proxy.append(frame["mean_ne_r2_cm3"])
        epochs.append(frame["decimal_year"][0])
    return times, [decimal_datetime(e) for e in epochs], elongations, np.array(brightness), np.array(polarized), np.array(proxy)


def plot_map(times, dates, elongations, brightness, prefix, title, geometry, proxy=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    if len(times) < 2:
        return
    baseline = np.maximum(brightness[0], np.finfo(float).tiny)
    normalized = brightness / baseline
    intervals = np.diff(times)/3600
    difference = np.diff(brightness, axis=0) / baseline / intervals[:, None]
    # Do not stretch a missing time interval into a false continuous feature.
    typical = np.median(intervals)
    # Event snapshots can sit between the regular daily/hourly outputs.
    # Estimate the regular interval from the upper part of the distribution.
    difference[intervals > 1.5*np.percentile(intervals, 90)] = np.nan
    limit = max(.01, np.nanpercentile(abs(difference), 98))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    first = axes[0].pcolormesh(dates, elongations, np.log10(np.maximum(normalized, 1e-12)).T,
                              shading="nearest", cmap="magma", vmin=-.5, vmax=1.5, rasterized=True)
    symbol = "D" if proxy else "B"
    fig.colorbar(first, ax=axes[0], label=f"log10[{symbol}(t) / {symbol}(first frame)]")
    second = axes[1].pcolormesh(dates[1:], elongations, difference.T, shading="nearest",
                               cmap="gray", vmin=-limit, vmax=limit, rasterized=True)
    fig.colorbar(second, ax=axes[1], label=f"Running difference / {symbol}(first frame), per hour")
    axes[0].set_title(title + (" — MS-FLUKSS density proxy" if proxy else " — normalized Thomson brightness"))
    axes[1].set_title("Running-difference J-map")
    for axis in axes:
        axis.set_ylabel("Elongation (degrees)")
        axis.set_ylim(elongations[0], elongations[-1])
        axis.set_xlim(dates[0], dates[-1])
        axis.grid(alpha=.12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M", tz=dt.timezone.utc))
    axes[-1].set_xlabel("Time (UTC)")
    detail = " | D = LOS mean of ne(r/AU)^2" if proxy else ""
    fig.suptitle(geometry + detail + f"\n{len(times)} times; median spacing {typical:.2f} h | modeled shell only", fontsize=10)
    fig.savefig(str(prefix) + ".png", dpi=180)
    fig.savefig(str(prefix) + ".pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["native", "slices", "volumes"])
    parser.add_argument("path", type=Path, help="Native CSV file or simulation run folder")
    parser.add_argument("--inputs", type=Path, help="Used simulation inputs (default: RUN/inputs)")
    parser.add_argument("--observer", type=Path, help="Observer trajectory in the same HGI format as probes")
    parser.add_argument("--reference", type=Path, help="Reference trajectory defining the slit plane (Earth for the MS-FLUKSS STEREO geometry)")
    parser.add_argument("--output", type=Path, required=True, help="Output prefix, without extension")
    parser.add_argument("--position-angle", type=float, default=90)
    parser.add_argument("--equatorial", action="store_true", help="Explicitly project observer to the solar-equatorial plane")
    parser.add_argument("--min-elongation", type=float, default=10)
    parser.add_argument("--max-elongation", type=float, default=50)
    parser.add_argument("--bins", type=int, default=201)
    parser.add_argument("--los-step", type=float, default=.001, help="Maximum LOS quadrature spacing in AU")
    parser.add_argument("--title", default="STEREO-A synthetic J-map")
    args = parser.parse_args()
    if not (0 < args.min_elongation < args.max_elongation < 180 and 2 <= args.bins <= 10000
            and np.isfinite(args.los_step) and args.los_step > 0 and np.isfinite(args.position_angle)):
        parser.error("Invalid elongation grid, position angle, or LOS step")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"source": args.source, "path": str(args.path.resolve()), "kernel": "finite solar disk; limb darkening 0.63",
                "normalization": "disk-center solar radiance", "electron_density": "rho/mp (pure hydrogen)",
                "domain": "modeled shell only; no extrapolation", "temporal_interpolation": False}
    if args.source == "native":
        times, dates, elongations, brightness, polarized, proxy = load_native(args.path)
        geometry = "Full 3D line-of-sight integration"
        metadata["source_header"] = [line for line in args.path.read_text().splitlines() if line.startswith("#")]
    else:
        if args.observer is None:
            parser.error("Saved-data processing requires --observer")
        if args.source == "slices" and (not args.equatorial or (args.reference is None and abs(np.cos(np.deg2rad(args.position_angle))) > 1e-10)):
            parser.error("Z slices require --equatorial and position angle 90 or 270 degrees")
        inputs_path = args.inputs or args.path / "inputs"
        if args.inputs is None and not inputs_path.exists():
            inputs_path = args.path / "inputs_CME_run"
        inputs = read_inputs(inputs_path)
        if int(inputs.get("amr_max_level", 0)):
            parser.error("Use the native diagnostic for AMR; archived composite volumes are not yet read here")
        shell = Shell(inputs)
        if 2*shell.outer/(args.los_step*AU) > 1000000:
            parser.error("LOS step would require more than one million samples per ray")
        data_prefix = Path(inputs.get("data_file_prefix", "data"))
        stem = data_prefix.name
        pattern = stem + (".z.*.dat" if args.source == "slices" else ".*.hdf5")
        name_pattern = re.escape(stem) + (r"\.z\.\d+\.dat" if args.source == "slices" else r"\.\d+\.hdf5")
        files = sorted((p for p in (args.path / data_prefix.parent).glob(pattern) if re.fullmatch(name_pattern, p.name)),
            key=lambda p: int(p.name.split(".")[-2]))
        if not files:
            parser.error("No saved density frames found")
        reader = read_slice if args.source == "slices" else read_volume
        start = start_datetime(inputs["BC_start_time"])
        # Read timestamps first; density volumes are read one at a time.
        if args.source == "slices":
            times = np.array([float(re.search(r"t=([\deE+.-]+)", p.open().readline()).group(1)) for p in files])
        else:
            import h5py
            times = []
            for path in files:
                with h5py.File(path, "r") as file:
                    times.append(float(file["level_0"].attrs["time"]))
            times = np.array(times)
        if np.any(np.diff(times) <= 0):
            raise ValueError("Saved frame times must increase")
        dates = [start + dt.timedelta(seconds=float(time)) for time in times]
        observers = trajectory_position(args.observer, dates, args.equatorial)
        references = trajectory_position(args.reference, dates, args.equatorial) if args.reference else [None]*len(dates)
        elongations = np.linspace(args.min_elongation, args.max_elongation, args.bins)
        result = []
        for i, (path, observer, reference) in enumerate(zip(files, observers, references)):
            _, density = reader(path, shell)
            result.append(integrate_frame(density, observer, shell, elongations, args.position_angle, args.los_step*AU, reference))
            if i % 10 == 0 or i == len(files) - 1:
                print(f"{args.source}: {i + 1}/{len(files)} — {dates[i].isoformat()}", flush=True)
        result = np.array(result)
        brightness, polarized, proxy = result[:, :, 0], result[:, :, 1], result[:, :, 2]
        geometry = "Solar-equatorial approximation (observer latitude set to zero)" if args.equatorial else "Full 3D; saved volume cadence"
        metadata.update(inputs=str(inputs_path.resolve()), observer=str(args.observer.resolve()),
                        observer_positions_cm=observers.tolist(), files=[p.name for p in files],
                        position_angle_deg=args.position_angle, los_step_AU=args.los_step,
                        reference=str(args.reference.resolve()) if args.reference else None,
                        shell_inner_AU=shell.inner/AU, shell_outer_AU=shell.outer/AU)
    if (not np.isfinite(brightness).all() or not np.isfinite(polarized).all()
            or not np.isfinite(proxy).all() or np.any(proxy < 0) or np.any(brightness < 0)
            or np.any(polarized < 0) or np.any(polarized > brightness*(1+1e-12))):
        raise ValueError("Invalid synthetic brightness")
    metadata.update(geometry=geometry, frames=len(times), times_UTC=[date.isoformat() for date in dates],
                    density_proxy="LOS mean of ne*(r/AU)^2 over the modeled shell, in cm^-3 (MS-FLUKSS definition)",
                    reconstruction="owned cell averages; midpoint LOS quadrature")
    np.savez_compressed(str(args.output) + ".npz", time_s=times, elongation_deg=elongations,
                        B_Bsun=brightness, pB_Bsun=polarized, mean_ne_r2_cm3=proxy)
    Path(str(args.output) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = np.column_stack((np.repeat(times, len(elongations)), np.tile(elongations, len(times)), brightness.ravel(), polarized.ravel(), proxy.ravel()))
    np.savetxt(str(args.output) + ".csv", rows, delimiter=",", header="time_s,elongation_deg,B_Bsun,pB_Bsun,mean_ne_r2_cm3", comments="")
    plot_map(times, dates, elongations, brightness, args.output, args.title, geometry)
    plot_map(times, dates, elongations, proxy, Path(str(args.output) + "_density_proxy"), args.title, geometry, proxy=True)


if __name__ == "__main__":
    main()
