#!/usr/bin/env python3
"""Extract launch-associated CME tracks from synthetic Thomson-brightness J-maps.

Example, from a simulation run folder::

    python /path/to/HelioCubed/scripts/extract_jmap_tracks.py \
        jmaps/StereoA.npz --run . --output jmaps/StereoA_tracks

Input is the NPZ produced by synthetic_jmap.py, or the native J-map CSV. Only
total Thomson brightness B_Bsun is used. CME identities, launch times and soft
kinematic priors come from the run manifest, GCS inputs and observer trajectory.

The observable is a positive, motion-filtered running-difference ridge: an apparent white-light
front, rather than a material element or the magnetic-rope axis. At successive
elongations, temporal peaks become nodes in a graph. Second-order dynamic
programming selects an outward, smoothly moving path through that graph.
The motion filter suppresses stationary/inward motion and slow background
streaks. Its automatic speed floor is 55% of the smallest projected GCS speed
over this map; --minimum-angular-speed 0 disables it. Genuine fronts slower
than this floor can be missed. Launch-time gates reject unrelated early streaks;
capped rewards prevent a
very bright CIR from winning solely through its amplitude. Paths can bridge
small gaps. Weak, unstable and shared ridges are flagged instead of assigning
confident identities to every feature. The method can return no detection.

Output includes fitted sub-frame arrival times, native time-resolution limits,
parameter-sensitivity ranges, quality flags, CSV/JSON tables and overlay figures.
Sensitivity ranges and quality labels are heuristic, not calibrated statistical
confidence intervals. A single J-map cannot uniquely establish CME identity at
every crossing. Review the overlays before using tracks for physical inference.

Related HI tracking/feature-definition studies: Pant et al. (2016),
doi:10.3847/1538-4357/833/1/80; Barnard et al. (2015),
doi:10.1002/2015SW001280. This graph tracker is a project-specific implementation.
"""
import argparse
from dataclasses import dataclass, replace, asdict
import datetime as dt
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from synthetic_jmap import (AU, read_inputs, start_datetime, trajectory_position,
                            load_native)


@dataclass
class Event:
    name: str
    launch_time_s: float
    speed_kms: float
    separation_deg: float
    observer_radius_AU: float
    inner_radius_AU: float

    def __post_init__(self):
        values = [self.launch_time_s, self.speed_kms, self.separation_deg,
                  self.observer_radius_AU, self.inner_radius_AU]
        if (not np.isfinite(values).all() or self.speed_kms <= 0
                or not 2 < self.separation_deg < 178
                or not 0 < self.inner_radius_AU < self.observer_radius_AU):
            raise ValueError('Invalid CME launch/motion prior')

    def expected_hours_per_degree(self, elongation):
        # Fixed-phi projection is a broad motion prior, not a fitted CME shape.
        phi = np.deg2rad(self.separation_deg)
        return (self.observer_radius_AU * AU / 1e5 * np.sin(phi)
                / np.sin(np.deg2rad(elongation) + phi)**2
                * np.pi / 180 / self.speed_kms / 3600)

    def seed_time_hours(self, elongation):
        phi = np.deg2rad(self.separation_deg)
        radius = self.observer_radius_AU * np.sin(np.deg2rad(elongation)) / np.sin(np.deg2rad(elongation) + phi)
        travel = max(0., radius - self.inner_radius_AU) * AU / 1e5 / self.speed_kms / 3600
        return self.launch_time_s / 3600 + travel, max(3., .8*travel)


@dataclass
class Settings:
    angle_step_deg: float = 1.
    angular_smoothing_deg: float = .35
    temporal_smoothing_samples: float = .45
    minimum_peak: float = 1.3
    minimum_prominence: float = .6
    reward_cap: float = 5.
    motion_penalty: float = .5
    curvature_penalty: float = .5
    maximum_angular_gap_deg: float = 3.
    maximum_time_gap_hours: float = 4.
    launch_time_tolerance_hours: float = .5
    minimum_angular_speed_deg_per_hour: float = -1.  # -1: infer a broad floor from GCS.
    minimum_points: int = 8
    minimum_span_deg: float = 8.


@dataclass
class Peak:
    layer: int
    angle: float
    time: float  # Hours since the simulation epoch; fitted peak time.
    strength: float
    prominence: float
    width_hours: float
    near_gap: bool


@dataclass
class State:
    peak: int
    rate: float  # dt / d(elongation), hours per degree.
    score: float
    parent: int
    points: int


def filter_slow_motion(hours, angles, image, minimum_speed, maximum_gap):
    """Smooth Fourier velocity filter, applied within observed time segments.

    A translating feature has Fourier frequencies ft=-v*fe. Suppress stationary,
    inward and slower outward structures before differencing. Irregular times
    are resampled only inside contiguous observed segments, then mapped back
    to the original times. This does not improve the native time resolution.
    """
    if minimum_speed <= 0:
        return image.copy()
    result = np.zeros_like(image)
    boundaries = np.r_[0, np.flatnonzero(np.diff(hours) > maximum_gap)+1, len(hours)]
    spacing = np.median(np.diff(angles))
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop-start < 8:
            result[start:stop] = image[start:stop]
            continue
        time = hours[start:stop]
        cadence = np.median(np.diff(time))
        regular = np.linspace(time[0], time[-1], max(2, int(round((time[-1]-time[0])/cadence))+1))
        cadence = regular[1]-regular[0]
        field = np.column_stack([np.interp(regular, time, image[start:stop, j]) for j in range(len(angles))])
        field -= np.median(field, axis=0)
        pad_t, pad_e = max(8, len(regular)//2), max(8, len(angles)//2)
        padded = np.pad(field, ((pad_t, pad_t), (pad_e, pad_e)), mode='reflect')
        ft = np.fft.fftfreq(padded.shape[0], cadence)[:, None]
        fe = np.fft.fftfreq(padded.shape[1], spacing)[None, :]
        velocity = np.divide(-ft, fe, out=np.zeros(padded.shape), where=abs(fe)>1e-12)
        mask = .5*(1+np.tanh((velocity-minimum_speed)/max(.04, .1*minimum_speed)))
        mask[:, abs(fe[0])<1e-12] = 0
        filtered = np.fft.ifft2(np.fft.fft2(padded)*mask).real
        filtered = filtered[pad_t:pad_t+len(regular), pad_e:pad_e+len(angles)]
        result[start:stop] = np.column_stack([np.interp(time, regular, filtered[:, j]) for j in range(len(angles))])
    return result


def prepare_evidence(time_s, angles, brightness, settings):
    """Remove radial brightness falloff, then locate positive temporal changes."""
    hours = np.asarray(time_s, dtype=float)/3600
    brightness = np.asarray(brightness, dtype=float)
    if len(hours) < 3 or len(angles) < 2 or not np.isfinite(hours).all() or not np.isfinite(angles).all():
        raise ValueError('At least three finite times and two finite elongations are required')
    intervals = np.diff(hours)
    if brightness.shape != (len(hours), len(angles)) or np.any(intervals <= 0):
        raise ValueError("Brightness must be time x elongation, with increasing times")
    if not np.isfinite(brightness).all() or np.any(brightness <= 0):
        raise ValueError("Provide positive finite Thomson brightness; omit missing frames rather than filling them")
    if np.any(np.diff(angles) <= 0):
        raise ValueError("Elongations must increase")
    background = np.maximum(np.percentile(brightness, 20, axis=0), np.finfo(float).tiny)
    normalized = brightness/background
    spacing = np.median(np.diff(angles))
    smoothed = gaussian_filter1d(normalized, settings.angular_smoothing_deg/spacing, axis=1)
    smoothed = filter_slow_motion(hours, angles, smoothed,
        settings.minimum_angular_speed_deg_per_hour, settings.maximum_time_gap_hours)
    rate = np.diff(smoothed, axis=0)/intervals[:, None]
    valid = intervals <= settings.maximum_time_gap_hours
    # Smooth each contiguous observed segment separately, across no large gap.
    start = 0
    for stop in np.r_[np.flatnonzero(~valid), len(rate)]:
        if stop > start:
            rate[start:stop] = gaussian_filter1d(rate[start:stop], settings.temporal_smoothing_samples, axis=0)
        start = stop+1
    available = rate[valid]
    if not len(available):
        return hours[1:], np.zeros_like(rate), rate, valid
    median = np.median(available, axis=0)
    noise = 1.4826*np.median(abs(available - median), axis=0)
    noise = np.maximum(noise, .04*np.percentile(abs(available), 95, axis=0))
    noise = np.maximum(noise, max(np.median(noise)*.2, 1e-12))
    evidence = np.log1p(np.maximum(rate, 0)/noise)/np.log(2)
    evidence[~valid] = 0
    return hours[1:], evidence, rate, valid


def find_candidates(hours, angles, evidence, valid, settings):
    """Find temporal maxima at approximately one-degree angular intervals."""
    stride = max(1, int(round(settings.angle_step_deg/np.median(np.diff(angles)))))
    columns = np.unique(np.r_[np.arange(0, len(angles), stride), len(angles)-1])
    peaks, layers = [], []
    cadence = np.median(np.diff(hours))
    for layer, column in enumerate(columns):
        profile = evidence[:, column]
        locations, properties = find_peaks(profile, height=settings.minimum_peak,
            prominence=settings.minimum_prominence, distance=2)
        widths = peak_widths(profile, locations, rel_height=.5)
        indices = []
        for p, location in enumerate(locations):
            if not valid[location] or location == 0 or location == len(hours)-1:
                continue
            # Fit the peak on actual, potentially irregular time coordinates.
            x = hours[location-1:location+2] - hours[location]
            y = profile[location-1:location+2]
            coefficient = np.polyfit(x, y, 2)
            offset = -coefficient[1]/(2*coefficient[0]) if coefficient[0] < 0 else 0.
            offset = np.clip(offset, .5*x[0], .5*x[-1])
            left = np.interp(widths[2][p], np.arange(len(hours)), hours)
            right = np.interp(widths[3][p], np.arange(len(hours)), hours)
            indices.append(len(peaks))
            peaks.append(Peak(layer, float(angles[column]), float(hours[location]+offset),
                float(min(settings.reward_cap, profile[location])),
                float(properties['prominences'][p]), float(max(cadence, right-left)),
                bool(not valid[location-1] or not valid[location+1])))
        layers.append(indices)
    return peaks, layers, angles[columns]


def trace_front(peaks, layers, track_angles, event, settings):
    """Second-order graph optimization; state retains the incoming motion rate."""
    launch = event.launch_time_s/3600
    seed, seed_sigma = event.seed_time_hours(track_angles[0])
    states, incoming = [], [[] for _ in peaks]
    for layer, indices in enumerate(layers):
        for index in indices:
            current = peaks[index]
            if (current.angle <= track_angles[0]+2*settings.angle_step_deg
                    and launch-settings.launch_time_tolerance_hours <= current.time <= seed+3*seed_sigma):
                score = (current.strength - settings.minimum_peak
                    - .5*((current.time-seed)/seed_sigma)**2
                    - 2*(current.angle-track_angles[0]))
                incoming[index].append(len(states))
                states.append(State(index, event.expected_hours_per_degree(current.angle), score, -1, 1))
            for previous_layer in range(layer-1, -1, -1):
                delta_angle = current.angle-track_angles[previous_layer]
                if delta_angle > settings.maximum_angular_gap_deg+1e-8:
                    break
                for previous in layers[previous_layer]:
                    if not incoming[previous]:
                        continue
                    before = peaks[previous]
                    delta_time = current.time-before.time
                    expected = event.expected_hours_per_degree((current.angle+before.angle)/2)
                    rate = delta_time/delta_angle
                    if not (0 < delta_time <= settings.maximum_time_gap_hours
                            and .12*expected < rate < 5*expected):
                        continue
                    choices = incoming[previous]
                    values = np.array([states[i].score for i in choices])
                    rates = np.array([states[i].rate for i in choices])
                    curvature = settings.curvature_penalty*((rate-rates)/(.6*expected+.15))**2
                    motion = settings.motion_penalty*np.log(rate/expected)**2
                    # Integrate reward in angle, so a long-lived slow CIR gets
                    # no extra reward simply by occupying more time samples.
                    reward = .5*(current.strength+before.strength)-settings.minimum_peak
                    scores = values+delta_angle*(reward-motion-curvature)-2*(layer-previous_layer-1)
                    selected = int(np.argmax(scores))
                    parent = choices[selected]
                    incoming[index].append(len(states))
                    states.append(State(index, rate, float(scores[selected]), parent, states[parent].points+1))
    for best in sorted(range(len(states)), key=lambda i: states[i].score, reverse=True):
        if states[best].points < settings.minimum_points or states[best].score <= 0:
            continue
        path, state = [], best
        while state >= 0:
            path.append(peaks[states[state].peak])
            state = states[state].parent
        path.reverse()
        if path[-1].angle-path[0].angle >= settings.minimum_span_deg:
            return path, states[best].score
    return [], None


def extract_tracks(time_s, angles, brightness, events, settings=None, ensemble=True):
    settings = settings or Settings()
    cadence = float(np.median(np.diff(np.asarray(time_s)/3600)))
    settings = replace(settings, launch_time_tolerance_hours=.5*cadence)
    if settings.minimum_angular_speed_deg_per_hour < 0:
        minimum_motion = min((np.min(1/event.expected_hours_per_degree(angles)) for event in events), default=0.)
        settings = replace(settings, minimum_angular_speed_deg_per_hour=float(.55*minimum_motion))
    hours, evidence, rate, valid = prepare_evidence(time_s, angles, brightness, settings)
    peaks, layers, track_angles = find_candidates(hours, angles, evidence, valid, settings)
    results = []
    # These deterministic perturbations test parameter sensitivity. They do
    # not constitute independent samples or formal statistical error bars.
    variants = [(a, t, p) for a, t, p in [(.25, .30, .7), (.5, .6, 1.4),
        (.35, .45, .65), (.35, .45, 1.5), (.25, .6, 1.), (.5, .3, 1.)]] if ensemble else []
    alternative_graphs = []
    for angular, temporal, penalty in variants:
        setting = replace(settings, angular_smoothing_deg=angular,
            temporal_smoothing_samples=temporal, motion_penalty=settings.motion_penalty*penalty,
            curvature_penalty=settings.curvature_penalty*penalty)
        ht, score, _, observed = prepare_evidence(time_s, angles, brightness, setting)
        graph = find_candidates(ht, angles, score, observed, setting)
        alternative_graphs.append((graph, setting, 1.))
    if ensemble and settings.minimum_angular_speed_deg_per_hour > 0:
        for scale in [.8, 1.2]:
            setting = replace(settings, minimum_angular_speed_deg_per_hour=settings.minimum_angular_speed_deg_per_hour*scale)
            ht, score, _, observed = prepare_evidence(time_s, angles, brightness, setting)
            alternative_graphs.append((find_candidates(ht, angles, score, observed, setting), setting, scale))
    for event in events:
        path, objective = trace_front(peaks, layers, track_angles, event, settings)
        if not path:
            seed, seed_sigma = event.seed_time_hours(angles[0])
            earliest_length = seed+seed_sigma+settings.minimum_span_deg*event.expected_hours_per_degree(angles[0]+settings.minimum_span_deg/2)
            status = ('not_yet_in_window' if hours[-1] < event.launch_time_s/3600
                else 'waiting_for_longer_track' if hours[-1] < earliest_length else 'no_supported_track')
            results.append({'event': asdict(event), 'status': status, 'points': []})
            continue
        alternatives = [path]
        for (variant_peaks, variant_layers, variant_angles), setting, scale in alternative_graphs:
            alternative, _ = trace_front(variant_peaks, variant_layers, variant_angles,
                replace(event, speed_kms=event.speed_kms*scale), setting)
            if alternative:
                alternatives.append(alternative)
        points = []
        for peak in path:
            arrivals = []
            for alternative in alternatives:
                match = next((p for p in alternative if abs(p.angle-peak.angle) < 1e-7), None)
                if match:
                    arrivals.append(match.time)
            low, high = np.percentile(arrivals, [10, 90])
            support = len(arrivals)/(1+len(alternative_graphs))
            flags = []
            if peak.near_gap:
                flags.append('gap_edge')
            if peak.time < event.launch_time_s/3600:
                flags.append('launch_time_unresolved')
            if support < .7 or high-low > max(1., cadence):
                flags.append('parameter_sensitive')
            if peak.strength < settings.minimum_peak+1:
                flags.append('weak_ridge')
            if any(abs(p.time-peak.time) > .75*cadence and abs(p.time-peak.time) < max(2., 2*cadence)
                and p.strength >= peak.strength-.75 for i in layers[peak.layer] for p in [peaks[i]]):
                flags.append('competing_ridge')
            points.append({'elongation_deg': peak.angle, 'time_s': peak.time*3600,
                'sensitivity_low_time_s': float(min(low, peak.time)*3600),
                'sensitivity_high_time_s': float(max(high, peak.time)*3600),
                'time_resolution_floor_s': cadence*1800, 'ridge_FWHM_s': peak.width_hours*3600,
                'ensemble_support': support, 'evidence_strength': peak.strength,
                'prominence': peak.prominence, 'flags': flags,
                'quality': 'review' if flags else 'stable'})
        status = 'reaches_upper_FOV' if path[-1].angle >= angles[-1]-.5*settings.angle_step_deg else 'partial_track'
        results.append({'event': asdict(event), 'status': status, 'objective': objective,
            'sensitivity_paths_found': len(alternatives), 'points': points})
    # Close ridges are not two independently identified CME fronts. Keep their
    # candidate associations in the output and explicitly flag shared evidence.
    for a in range(len(results)):
        for b in range(a+1, len(results)):
            other = {point['elongation_deg']: point for point in results[b]['points']}
            for point in results[a]['points']:
                match = other.get(point['elongation_deg'])
                if match and abs(match['time_s']-point['time_s']) < max(3600., cadence*3600):
                    for item in [point, match]:
                        if 'shared_or_merged' not in item['flags']:
                            item['flags'].append('shared_or_merged')
                        item['quality'] = 'review'
    raw_rate = np.diff(brightness, axis=0)/brightness[0]/(np.diff(time_s)/3600)[:, None]
    for track in results:
        for point in track['points']:
            column = int(np.argmin(abs(angles-point['elongation_deg'])))
            value = np.interp(point['time_s']/3600, hours, raw_rate[:, column])
            if value <= 0:
                point['flags'].append('background_dominated')
                point['quality'] = 'review'
    return results, {'time_hours': hours, 'evidence': evidence, 'rate': rate,
                     'valid_intervals': valid, 'candidates': peaks, 'settings_used': asdict(settings)}


def events_from_run(run, metadata):
    manifest = json.loads((run/'run_manifest.json').read_text())
    inputs = read_inputs(run/'inputs_CME_run')
    epoch = start_datetime(inputs['BC_start_time'])
    trajectories = manifest.get('jmap', {})
    observer_file = Path(trajectories.get('observer', inputs.get('probe1.coords', 'trajStereoA.dat')))
    earth_file = Path(trajectories.get('reference', inputs.get('probe0.coords', 'trajEarth.dat')))
    if not observer_file.is_absolute():
        observer_file = run/observer_file
    if not earth_file.is_absolute():
        earth_file = run/earth_file
    equatorial = 'equatorial' in metadata.get('geometry', '').lower()
    launch_times = manifest.get('expected_emergence_start_times_seconds', manifest.get('expected_insertion_times_seconds'))
    if launch_times is None and manifest.get('events'):
        raise ValueError("Run manifest needs expected CME launch times")
    events = []
    for i, source in enumerate(manifest.get('events', [])):
        launch = float(launch_times[i])
        date = epoch+dt.timedelta(seconds=launch)
        observer = trajectory_position(observer_file, [date], equatorial)[0]
        earth = trajectory_position(earth_file, [date], equatorial)[0]
        values = source['input_values']
        longitude = np.arctan2(earth[1], earth[0])+np.deg2rad(float(values['GCS_longitude']))
        latitude = np.deg2rad(float(values['GCS_latitude']))
        axis = np.array([np.cos(latitude)*np.cos(longitude), np.cos(latitude)*np.sin(longitude), np.sin(latitude)])
        cosine = np.clip(axis @ observer/np.linalg.norm(observer), -1, 1)
        separation = float(np.rad2deg(np.arccos(cosine)))
        if not 2 < separation < 178:
            raise ValueError("The fixed-phi motion prior is degenerate for an almost observer-aligned axis")
        events.append(Event('CME'+str(source.get('source', i)), launch, float(values['GCS_speed']),
            separation, float(np.linalg.norm(observer)/AU), float(inputs['inner_radius'])/AU))
    return events, epoch


def save_products(prefix, times, angles, brightness, tracks, epoch, settings, diagnostics, metadata):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    def timestamp(seconds):
        return (epoch+dt.timedelta(seconds=float(seconds))).isoformat()
    for track in tracks:
        for point in track['points']:
            point['time_UTC'] = timestamp(point['time_s'])
    report = {'quantity': 'total Thomson brightness B_Bsun', 'observable': 'positive motion-filtered running-difference ridge',
        'method': 'motion filtering and launch-associated second-order graph optimization',
        'settings': diagnostics['settings_used'],
        'geometry': metadata.get('geometry', 'see source J-map'),
        'time_estimates': 'Quadratic sub-frame peak fits; resolution floors and parameter sensitivity are exported separately.',
        'quality_labels': 'Heuristic stability diagnostics, not probabilities; shared/crossing fronts require review.',
        'gaps_larger_than_limit': [{'start_time_s': float(times[i]), 'end_time_s': float(times[i+1])}
            for i in np.flatnonzero(np.diff(times)/3600 > settings.maximum_time_gap_hours)],
        'tracks': tracks}
    Path(str(prefix)+'.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with Path(str(prefix)+'.csv').open('w', newline='') as file:
        fields = ['CME', 'time_UTC', 'time_s', 'elongation_deg', 'sensitivity_low_time_s',
            'sensitivity_high_time_s', 'time_resolution_floor_s', 'ridge_FWHM_s',
            'ensemble_support', 'evidence_strength', 'prominence', 'quality', 'flags']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for track in tracks:
            for point in track['points']:
                row = dict(point, CME=track['event']['name'])
                row['flags'] = ';'.join(row['flags'])
                writer.writerow(row)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    dates = [epoch+dt.timedelta(seconds=float(time)) for time in times]
    intervals = np.diff(times)/3600
    difference = np.diff(brightness, axis=0)/brightness[0]/intervals[:, None]
    difference[intervals > settings.maximum_time_gap_hours] = np.nan
    limit = max(.01, np.nanpercentile(abs(difference), 98))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    image = axes[0].pcolormesh(dates[1:], angles, difference.T, shading='nearest', cmap='gray',
        vmin=-limit, vmax=limit, rasterized=True)
    fig.colorbar(image, ax=axes[0], label='Running difference / first brightness, per hour')
    evidence_image = axes[1].pcolormesh(dates[1:], angles,
        np.minimum(settings.reward_cap, diagnostics['evidence']).T, shading='nearest',
        cmap='magma', vmin=0, vmax=settings.reward_cap, rasterized=True)
    fig.colorbar(evidence_image, ax=axes[1], label='Capped ridge evidence')
    colors = ['#00a6d6', '#ffab00', '#ed3e8a', '#44aa66']
    for i, track in enumerate(tracks):
        points = track['points']
        if not points:
            continue
        color = colors[i % len(colors)]
        time = np.array([mdates.date2num(epoch+dt.timedelta(seconds=p['time_s'])) for p in points])
        angle = np.array([p['elongation_deg'] for p in points])
        low = [mdates.date2num(epoch+dt.timedelta(seconds=p['sensitivity_low_time_s'])) for p in points]
        high = [mdates.date2num(epoch+dt.timedelta(seconds=p['sensitivity_high_time_s'])) for p in points]
        review = np.array([p['quality'] != 'stable' for p in points])
        for axis in axes:
            axis.fill_betweenx(angle, low, high, color=color, alpha=.2)
            # A faint dotted connector also exposes missing angular nodes.
            axis.plot(time, angle, ':', color=color, lw=1)
            for j in range(len(points)-1):
                if angle[j+1]-angle[j] <= 1.5*settings.angle_step_deg:
                    axis.plot(time[j:j+2], angle[j:j+2], '--' if review[j:j+2].any() else '-', color=color, lw=1.7)
            axis.scatter(time[~review], angle[~review], s=11, color=color,
                label=track['event']['name']+' — '+track['status'].replace('_', ' '))
            axis.scatter(time[review], angle[review], s=19, facecolors='none', edgecolors=color)
    for axis in axes:
        axis.set_ylabel('Elongation (degrees)')
        axis.set_ylim(angles[0], angles[-1])
        axis.set_xlim(dates[0], dates[-1])
        axis.grid(alpha=.1)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc='upper left', fontsize=9)
    elif tracks:
        axes[0].text(.02, .98, '\n'.join(item['event']['name']+': '+item['status'].replace('_', ' ') for item in tracks),
            transform=axes[0].transAxes, va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
    axes[0].set_title('Thomson running-difference map and extracted CME candidates')
    axes[1].set_title('Detection evidence | hollow markers/dashes: review | bands: parameter sensitivity')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M', tz=dt.timezone.utc))
    axes[1].set_xlabel('Time (UTC)')
    fig.suptitle(metadata.get('geometry', 'Synthetic J-map')+'\nLaunch-associated apparent fronts; crossings and merged ridges can remain ambiguous', fontsize=11)
    fig.savefig(str(prefix)+'.png', dpi=180)
    fig.savefig(str(prefix)+'.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('jmap', type=Path, help='Thomson J-map NPZ or native CSV')
    parser.add_argument('--run', type=Path, required=True, help='Run folder containing used inputs, manifest and trajectories')
    parser.add_argument('--output', type=Path, required=True, help='Output prefix')
    parser.add_argument('--minimum-prominence', type=float, default=.6)
    parser.add_argument('--maximum-gap-hours', type=float, default=4.)
    parser.add_argument('--minimum-angular-speed', type=float, default=-1.,
        help='Fourier motion floor in degrees/hour; -1 derives it from GCS priors, 0 disables filtering')
    parser.add_argument('--no-ensemble', action='store_true', help='Skip parameter-sensitivity checks')
    args = parser.parse_args()
    if args.minimum_prominence <= 0 or args.maximum_gap_hours <= 0:
        parser.error('Prominence and maximum gap must be positive')
    if (not np.isfinite([args.minimum_prominence, args.maximum_gap_hours, args.minimum_angular_speed]).all()
            or (args.minimum_angular_speed < 0 and args.minimum_angular_speed != -1)):
        parser.error('Use finite settings and a speed floor of -1, 0, or a positive value')
    metadata_file = args.jmap.with_suffix('.json')
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    if args.jmap.suffix == '.npz':
        with np.load(args.jmap) as data:
            times, angles, brightness = data['time_s'], data['elongation_deg'], data['B_Bsun']
    else:
        times, _, angles, brightness, _, _ = load_native(args.jmap)
        metadata.setdefault('geometry', 'Full 3D native line-of-sight integration')
    if len(times) < 4:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Path(str(args.output)+'.json').write_text(json.dumps({'status': 'waiting_for_more_profiles', 'profiles': len(times)}, indent=2)+'\n')
        return
    events, epoch = events_from_run(args.run.resolve(), metadata)
    settings = Settings(minimum_prominence=args.minimum_prominence, maximum_time_gap_hours=args.maximum_gap_hours,
        minimum_angular_speed_deg_per_hour=args.minimum_angular_speed)
    tracks, diagnostics = extract_tracks(times, angles, brightness, events, settings, not args.no_ensemble)
    save_products(args.output, times, angles, brightness, tracks, epoch, settings, diagnostics, metadata)
    provenance = {'jmap': str(args.jmap.resolve()), 'jmap_sha256': hashlib.sha256(args.jmap.read_bytes()).hexdigest(),
        'run': str(args.run.resolve()), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(str(args.output)+'_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps([{'CME': item['event']['name'], 'status': item['status'],
        'points': len(item['points']), 'review_points': sum(p['quality'] != 'stable' for p in item['points'])}
        for item in tracks], indent=2))


if __name__ == '__main__':
    main()
