#pragma once

#include "MHD_Constants.H"
#include "MHD_Time.H"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

// CME models and source physics.
//
// 1. FRiED geometry and constant-turn field.
// 2. GCS geometry and its smoothly joined constant-turn field.
// 3. Derived source parameters and conversion from observed GCS inputs.
// 4. Shared expansion, density loading and conservative energy increment.
// 5. Multiple-CME event scheduling.
//
// Geometry kernels use Rsun and G. The public source evaluator uses Cartesian
// positions in cm, velocities in cm/s, density in g/cm^3 and pressure in dyn/cm^2.
// Numerical grid operations belong to the solver and call this same model API.

// ------------------------- Geometry and magnetic fields -------------------------

// HOST evaluator for the regular circular-normal CTFR. Lengths are in solar
// radii, fields in gauss, and b in inverse solar radii. No field clipping.
namespace CTFRNormal {
constexpr double pi = 3.1415926535897932384626433832795;
// Axis position, unit tangent/normal, d(arclength)/d(phi), curvature and section radius.
struct Geometry {
    double x, y, tx, ty, nx, ny, speed, curvature, radius, radius_s;
};
struct Sample {
    std::array<double, 3> field, displacement;
    double phi = 0;
};

class Axis {
  public:
    static constexpr int intervals = 256;
    double major = 0, minor = 0, width = 0, flattening = 0;
    double maxCurvatureRadius = 0;
    struct Node {
        double phi, x, y, tx, ty;
    };
    std::array<Node, intervals + 1> nodes{};

    Geometry geometry(double phi) const {
        const double k = pi / (2 * width), arg = k * phi, co = std::cos(arg);
        const double axisR = major * std::pow(co, flattening);
        const double u = -flattening * k * std::tan(arg), norm = std::sqrt(1 + u * u);
        const double cp = std::cos(phi), sp = std::sin(phi);
        const double tx = (u * cp - sp) / norm, ty = (u * sp + cp) / norm;
        const double curvature =
            (1 + u * u + flattening * k * k / (co * co)) / (axisR * norm * norm * norm);
        return {axisR * cp,
                axisR * sp,
                tx,
                ty,
                ty,
                -tx,
                axisR * norm,
                curvature,
                minor * axisR / major,
                minor / major * u / norm};
    }

    void prepare(double rt, double rp, double halfDegrees, double n) {
        major = rt;
        minor = rp;
        width = halfDegrees * pi / 180;
        flattening = n;
        if (!(major > minor && minor > 0 && width > 0 && width < pi / 2 && n >= 0))
            throw std::invalid_argument("Normal CTFR needs Rt>Rp>0 and 0<halfangle<90 degrees");
        maxCurvatureRadius = 0;
        for (int i = 0; i <= 32768; ++i) {
            const auto g = geometry(width * .9999999999 * (2. * i / 32768 - 1));
            maxCurvatureRadius = std::max(maxCurvatureRadius, g.curvature * g.radius);
        }
        if (!(maxCurvatureRadius < 1 - 1e-8))
            throw std::invalid_argument(
                "Normal CTFR geometry folds: max(curvature*radius) must be <1");
        // Cluster toward the footpoints; small sections there must still be
        // resolved by the normal-coordinate root search.
        for (int i = 0; i <= intervals; ++i) {
            const double p = width * .9999999999 * std::sin(pi * (double(i) / intervals - .5));
            const auto g = geometry(p);
            nodes[i] = {p, g.x, g.y, g.tx, g.ty};
        }
    }

    // b is the constant turn rate [Rsun^-1]; b0 is the apex axial field [G].
    bool sample(double x, double y, double z, double b, double b0, int helicity,
                Sample &result) const {
        const double eta = minor / major, distance = std::sqrt(x * x + y * y + z * z);
        if (distance > major + minor || std::abs(z) > minor ||
            std::abs(std::atan2(y, x)) > width + std::asin(eta))
            return false;
        int found = 0;
        double previousRoot = -10;
        double last = (x - nodes[0].x) * nodes[0].tx + (y - nodes[0].y) * nodes[0].ty;
        for (int i = 0; i < intervals; ++i) {
            const auto &a = nodes[i];
            const auto &next = nodes[i + 1];
            const double rightValue = (x - next.x) * next.tx + (y - next.y) * next.ty;
            if (last * rightValue <= 0) {
                double lo = a.phi, hi = next.phi, flo = last, p = .5 * (lo + hi);
                for (int iteration = 0; iteration < 48; ++iteration) {
                    const auto g = geometry(p);
                    const double dx = x - g.x, dy = y - g.y;
                    const double f = dx * g.tx + dy * g.ty;
                    if (std::abs(f) < 1e-11 || hi - lo < 1e-13)
                        break;
                    if (flo * f <= 0)
                        hi = p;
                    else {
                        lo = p;
                        flo = f;
                    }
                    const double metric = 1 + g.curvature * (dx * g.nx + dy * g.ny);
                    const double candidate = p + f / (g.speed * metric);
                    p = (std::isfinite(candidate) && candidate > lo + .05 * (hi - lo) &&
                         candidate < hi - .05 * (hi - lo))
                            ? candidate
                            : .5 * (lo + hi);
                }
                const auto g = geometry(p);
                const double dx = x - g.x, dy = y - g.y, outward = dx * g.nx + dy * g.ny;
                const double r = std::hypot(outward, z);
                if (r <= g.radius && std::abs(p - previousRoot) > 1e-9) {
                    previousRoot = p;
                    if (++found > 1)
                        throw std::runtime_error(
                            "Normal CTFR point has multiple interior axis assignments");
                    const double logarithm = std::log1p(b * b * g.radius * g.radius);
                    const double localB0 = b0 * std::log1p(b * b * minor * minor) / logarithm;
                    const double bs = localB0 / (1 + b * b * r * r);
                    const double h = 1 + g.curvature * outward;
                    const double theta = std::atan2(z, outward), ct = std::cos(theta),
                                 st = std::sin(theta);
                    double radialMotion = 0;
                    if (r > 0) {
                        const double lambda = std::log1p(b * b * r * r) / logarithm;
                        radialMotion = lambda * g.radius * g.radius_s * (1 + b * b * r * r) /
                                       (r * (1 + b * b * g.radius * g.radius));
                    }
                    const double br = bs * radialMotion / h, bt = -helicity * b * r * bs / h;
                    result.field = {bs * g.tx + (br * ct - bt * st) * g.nx,
                                    bs * g.ty + (br * ct - bt * st) * g.ny, br * st + bt * ct};
                    result.displacement = {dx, dy, z};
                    result.phi = p;
                }
            }
            last = rightValue;
        }
        return found == 1;
    }
};
} // namespace CTFRNormal

// GCS circular sections and conical legs, with C2 section maps over a short
// region at each cone/front join (lengths: Rsun). The apex and the geometry
// outside those regions are unchanged. Smoothing the map, rather than B,
// removes the internal current sheets while retaining the Piola identities.
// The front section planes are oblique to their centerline. With center C(s),
// in-plane section direction e(s), and f=e_phi, write
//   X = C(s) + r*cos(theta)*e(s) + r*sin(theta)*ez.
// On the exact front h=(Q+r*cos(theta))/|dC/dphi|; in general,
// h=T.f+r*cos(theta)*d(angle)/ds. Section planes need not be normal to T.
// A Piola field B = Bs/h * (X_s + r_s|lambda*X_r - helicity*b*X_theta),
// lambda=log(1+b^2*r^2)/log(1+b^2*a(s)^2), conserves axial flux and gives
// dtheta/ds=-helicity*b. It reduces to Vandas & Romashets (2017) for a torus.
// This is a solenoidal GCS extension, not a force-free GCS equilibrium.
namespace GCSTube {
constexpr double pi = CTFRNormal::pi;

struct Section {
    double x = 0, y = 0, ex = 0, ey = 0, fx = 0, fy = 0;
    double radius = 0, radius_s = 0, lengthDensity = 0;
    double normalProjection = 1, drift = 0, rotationRate = 0;
};

class Tube {
  public:
    double height = 0, halfAngle = 0, aspectRatio = 0;
    double legB = 0, rho = 0, denominator = 0, discriminant = 0;
    double frontEnd = 0, joinLength = 0, coneSlope = 0, apexRadius = 0;
    double minimumMetric = 0, joinHalfWidth = 0;
    // Quintics in u=(t+joinHalfWidth)/(2*joinHalfWidth), for center x/y,
    // section angle, and radius, relative to the original upper join.
    // t follows the original circular-arc arclength into the conical leg.
    std::array<std::array<double, 6>, 4> joinCoefficients{};

    Section exactFront(double phi) const {
        const double co = std::cos(phi), si = std::sin(phi);
        const double q = (rho + legB * aspectRatio * aspectRatio * co) / denominator;
        const double qp = -legB * aspectRatio * aspectRatio * si / denominator;
        // Algebraically sqrt(Q^2+D), written without cancellation for thin GCS.
        const double a = aspectRatio / denominator *
                         std::hypot(rho + legB * co, legB * std::sqrt(denominator) * si),
                     ell = std::hypot(q, qp);
        return {legB + q * co,      q * si, co,      si,       -si,    co, a,
                q * qp / (a * ell), ell,    q / ell, qp / ell, 1 / ell};
    }

    Section exactLeg(double length, int side) const {
        const double co = std::cos(halfAngle), si = std::sin(halfAngle);
        // Increasing axis arclength goes from the lower leg, over the front,
        // and back to the Sun on the upper leg. e is continuous at each join.
        return {length * co,        side * length * si, -si, side * co, -side * co, -si,
                length * coneSlope, -side * coneSlope,  1,   1,         0,          0};
    }

    Section joinSection(double t, int side) const {
        const double span = 2 * joinHalfWidth, u = (t + joinHalfWidth) / span;
        std::array<double, 4> value{}, derivative{};
        for (int j = 0; j < 4; ++j) {
            const auto &c = joinCoefficients[j];
            value[j] = c[5];
            derivative[j] = 5 * c[5];
            for (int i = 4; i >= 0; --i)
                value[j] = value[j] * u + c[i];
            for (int i = 4; i >= 1; --i)
                derivative[j] = derivative[j] * u + i * c[i];
            derivative[j] /= span;
        }
        const double angle = frontEnd + value[2], co = std::cos(angle), si = std::sin(angle);
        const double ell = std::hypot(derivative[0], derivative[1]);
        const double normal = (-derivative[0] * si + derivative[1] * co) / ell;
        const double drift = side * (derivative[0] * co + derivative[1] * si) / ell;
        return {joinLength * std::cos(halfAngle) + value[0],
                side * (joinLength * std::sin(halfAngle) + value[1]),
                co,
                side * si,
                -side * si,
                co,
                joinLength * coneSlope + value[3],
                side * derivative[3] / ell,
                ell,
                normal,
                drift,
                derivative[2] / ell};
    }

    Section front(double phi) const {
        const double t = rho * (std::abs(phi) - frontEnd);
        if (joinHalfWidth > 0 && std::abs(t) <= joinHalfWidth) {
            auto g = joinSection(t, phi < 0 ? -1 : 1);
            g.lengthDensity *= rho; // ds/dphi, including the smoothed centerline
            return g;
        }
        return exactFront(phi);
    }

    Section leg(double length, int side) const {
        const double t = joinLength - length;
        if (joinHalfWidth > 0 && std::abs(t) <= joinHalfWidth)
            return joinSection(t, side);
        return exactLeg(length, side);
    }

    void prepareJoin(double fraction = .15) {
        // A fixed fraction of the local scales: no additional user inputs.
        joinHalfWidth = fraction * std::min({rho, joinLength, joinLength * coneSlope});
        const double span = 2 * joinHalfWidth, delta = -joinHalfWidth / rho;
        const double phi = frontEnd + delta, co = std::cos(phi), si = std::sin(phi);
        const auto g = exactFront(phi);
        const double qp = -legB * aspectRatio * aspectRatio * si / denominator;
        const double qpp = -legB * aspectRatio * aspectRatio * co / denominator;
        // Relative endpoint positions avoid subtracting two solar distances
        // when the aspect ratio (and the transition width) is extremely small.
        const double halfSin = std::sin(delta / 2), mid = frontEnd + delta / 2;
        const double dex = -2 * std::sin(mid) * halfSin, dey = 2 * std::cos(mid) * halfSin;
        const double dq = legB * aspectRatio * aspectRatio * dex / denominator, q = rho + dq;
        const double aj = joinLength * coneSlope, ca = std::cos(halfAngle),
                     sa = std::sin(halfAngle);
        const std::array<double, 4> start = {dq * co + rho * dex, dq * si + rho * dey, delta,
                                             dq * (q + rho) / (g.radius + aj)};
        const std::array<double, 4> startD = {(qp * co - q * si) / rho, (qp * si + q * co) / rho,
                                              1 / rho, q * qp / (g.radius * rho)};
        const std::array<double, 4> startDD = {
            ((qpp - q) * co - 2 * qp * si) / (rho * rho),
            ((qpp - q) * si + 2 * qp * co) / (rho * rho), 0,
            (q * qpp / g.radius + discriminant * qp * qp / (g.radius * g.radius * g.radius)) /
                (rho * rho)};
        const std::array<double, 4> end = {-joinHalfWidth * ca, -joinHalfWidth * sa, 0,
                                           -joinHalfWidth * coneSlope};
        const std::array<double, 4> endD = {-ca, -sa, 0, -coneSlope};
        for (int j = 0; j < 4; ++j) {
            auto &c = joinCoefficients[j];
            c[0] = start[j];
            c[1] = span * startD[j];
            c[2] = span * span * startDD[j] / 2;
            const double a = end[j] - c[0] - c[1] - c[2];
            const double b = span * endD[j] - c[1] - 2 * c[2], d = -2 * c[2];
            c[3] = 10 * a - 4 * b + d / 2;
            c[4] = -15 * a + 7 * b - d;
            c[5] = 6 * a - 3 * b + d / 2;
        }
    }

    void prepare(double leadingHeight, double halfDegrees, double ratio) {
        if (!(std::isfinite(leadingHeight) && std::isfinite(halfDegrees) && std::isfinite(ratio) &&
              leadingHeight > 0 && halfDegrees > 0 && halfDegrees < 90 && ratio > 0 && ratio < 1))
            throw std::invalid_argument(
                "GCS needs height>0, 0<half angle<90 degrees, and 0<aspect ratio<1");
        height = leadingHeight;
        halfAngle = halfDegrees * pi / 180;
        aspectRatio = ratio;
        // At kappa>=sin(alpha), GCS circular sections overlap at the inner
        // bend; a single-valued constant-turn tube cannot use this foliation.
        // Preserve the requested dimensions: reject this case, never thin it.
        if (!(aspectRatio < std::sin(halfAngle)))
            throw std::invalid_argument(
                "GCS constant-turn sections overlap: GCS_aspect_ratio must be < "
                "sin(GCS_half_angle). No geometry adjustment was made.");
        denominator = 1 - ratio * ratio;
        legB = height * (1 - ratio) / (1 + std::sin(halfAngle));
        rho = legB * std::sin(halfAngle);
        discriminant = (legB * legB * ratio * ratio - rho * rho) / denominator;
        frontEnd = pi / 2 + halfAngle;
        joinLength = legB * std::cos(halfAngle);
        coneSlope = ratio / std::sqrt(denominator);
        apexRadius = ratio * height / (1 + ratio);
        minimumMetric = 1;
        for (int i = 0; i <= 4096; ++i) {
            const auto g = exactFront(frontEnd * i / 4096.);
            const double q = g.normalProjection / g.rotationRate;
            minimumMetric =
                std::min(minimumMetric, -discriminant / (q + g.radius) * g.rotationRate);
        }
        if (!(minimumMetric > 1e-12))
            throw std::invalid_argument("GCS constant-turn section metric is numerically singular; "
                                        "no geometry adjustment was made");
        bool validJoin = false;
        for (int attempt = 0; attempt < 24; ++attempt) {
            prepareJoin(std::ldexp(.15, -attempt));
            double joinMetric = 1;
            validJoin = true;
            for (int i = 0; i <= 1024; ++i) {
                const auto g = joinSection(joinHalfWidth * (2. * i / 1024. - 1), 1);
                const double h = g.normalProjection - g.radius * std::abs(g.rotationRate);
                // A local positive Jacobian alone does not exclude an upper
                // section intersecting its lower counterpart near kappa=sin(alpha).
                const double innerY = g.y - g.radius * std::abs(g.ey);
                if (!(std::isfinite(h) && h > 1e-12 && g.radius > 0 && g.lengthDensity > 0 &&
                      innerY > 0)) {
                    validJoin = false;
                    break;
                }
                joinMetric = std::min(joinMetric, h);
            }
            if (validJoin) {
                minimumMetric = std::min(minimumMetric, joinMetric);
                break;
            }
        }
        if (!validJoin)
            throw std::invalid_argument("Cannot construct a non-overlapping smooth GCS join");
    }

    bool fieldAtSection(const Section &g, double outward, double z, double b, double b0,
                        int helicity, CTFRNormal::Sample &result) const {
        const double r = std::hypot(outward, z);
        if (!(g.radius > 0) || r > g.radius)
            return false;
        const double h = g.normalProjection + outward * g.rotationRate;
        if (!(h > 0))
            throw std::runtime_error("GCS constant-turn coordinate fold");
        const double logarithm = std::log1p(b * b * g.radius * g.radius);
        const double bs =
            b0 * std::log1p(b * b * apexRadius * apexRadius) / logarithm / (1 + b * b * r * r);
        const double co = r > 0 ? outward / r : 1, si = r > 0 ? z / r : 0;
        double radialMotion = 0;
        if (r > 0) {
            const double lambda = std::log1p(b * b * r * r) / logarithm;
            radialMotion = lambda * g.radius * g.radius_s * (1 + b * b * r * r) /
                           (r * (1 + b * b * g.radius * g.radius));
        }
        const double be = bs * (g.drift + radialMotion * co + helicity * b * r * si) / h;
        const double bz = bs * (radialMotion * si - helicity * b * r * co) / h;
        result.field = {bs * g.fx + be * g.ex, bs * g.fy + be * g.ey, bz};
        result.displacement = {outward * g.ex, outward * g.ey, z};
        return true;
    }

    bool sample(double x, double y, double z, double b, double b0, int helicity,
                CTFRNormal::Sample &result) const {
        if (x * x + y * y + z * z > height * height || std::abs(z) > apexRadius)
            return false;
        const double phi = std::atan2(y, x - legB);
        if (std::abs(phi) <= frontEnd - joinHalfWidth / rho) {
            const auto g = front(phi);
            const double outward = (x - g.x) * g.ex + (y - g.y) * g.ey;
            if (fieldAtSection(g, outward, z, b, b0, helicity, result)) {
                result.phi = phi;
                return true;
            }
        }
        for (int side : {-1, 1}) {
            const double length = x * std::cos(halfAngle) + side * y * std::sin(halfAngle);
            if (!(length > 0 && length <= joinLength - joinHalfWidth))
                continue;
            const auto g = leg(length, side);
            const double outward = x * g.ex + y * g.ey;
            if (fieldAtSection(g, outward, z, b, b0, helicity, result)) {
                result.phi = side * frontEnd;
                return true;
            }
        }
        // Invert the smooth section planes. The derivative of their plane
        // residual is -ds/dt*h, so the positive metric gives a unique root.
        const auto start = joinSection(-joinHalfWidth, 1), end = joinSection(joinHalfWidth, 1);
        for (int side : {-1, 1}) {
            const double yy = side * y;
            const double first = (x - start.x) * start.fx + (yy - start.y) * start.fy;
            const double last = (x - end.x) * end.fx + (yy - end.y) * end.fy;
            if (first < 0 || last > 0)
                continue;
            double lo = -joinHalfWidth, hi = joinHalfWidth, t = 0;
            for (int i = 0; i < 64; ++i) {
                const auto g = joinSection(t, 1);
                const double residual = (x - g.x) * g.fx + (yy - g.y) * g.fy;
                if (std::abs(residual) < 2e-15 * height || hi - lo < 2e-15 * height)
                    break;
                if (residual > 0)
                    lo = t;
                else
                    hi = t;
                const double outward = (x - g.x) * g.ex + (yy - g.y) * g.ey;
                const double slope =
                    -g.lengthDensity * (g.normalProjection + outward * g.rotationRate);
                const double next = t - residual / slope;
                t = (slope < 0 && next > lo && next < hi) ? next : (lo + hi) / 2;
            }
            const auto g = joinSection(t, side);
            const double outward = (x - g.x) * g.ex + (y - g.y) * g.ey;
            if (fieldAtSection(g, outward, z, b, b0, helicity, result)) {
                result.phi = side * (frontEnd + t / rho);
                return true;
            }
        }
        return false;
    }
};
} // namespace GCSTube

// ------------------------- GCS input conversion -------------------------

// All inputs are GCS observations; these derived quantities stay internal.
struct CMEParameters {
    int sourceIndex = 0;
    int fieldModel = 1;      // CME_type: 1 = normal-section FRiED, 2 = GCS.
    int injectionMethod = 0; // 0 = volume + leg feeding, 1 = boundary emergence.
    CTFRNormal::Axis normalAxis;
    GCSTube::Tube gcsTube;

    double insertionTime = 0; // Start of the source, seconds after BC_start_time.
    double referenceTime = 0; // Epoch at which the apex is at r1 + r0.
    double enterTime = 0;     // The same start epoch expressed as decimal year.
    double geometryScale = 1; // Current apex height / reference apex height.
    double dragGamma = 0;     // DBM drag coefficient [km^-1].
    double windSpeed = 450;   // DBM ambient speed [km/s].

    double r1 = 0;                     // Reference axis apex distance [Rsun].
    double r0 = 0;                     // Reference apex section radius [Rsun].
    double halfangle = 0;              // Geometry half angle [degrees].
    double FRiED_n = 0;                // FRiED flattening, derived from GCS half angle / 100.
    double lat = 0, lon = 0, tilt = 0; // Orientation [degrees].
    double apex_speed = 0;             // Reference apex speed after DBM [km/s].
    double pol_flux_control = 0;       // Constant field-line turn rate [AU^-1].
    double a1 = 0;                     // Reference apex axial field strength [G].
    double helicity_sign = 1;
    double density = 0;    // Minimum added source density [g/cm^3].
    double plasmaBeta = 0; // Added thermal / CME magnetic pressure; 0 retains ambient T.
    double boundaryTaperTime = 43200; // Seconds after insertion until only WSA remains; 0 disables feeding.
};

// Native GCS inputs -> selected CME geometry. Uses the solver's 696000 km solar
// radius consistently. Fluxes are Mx; speeds km/s; DBM gamma km^-1; angles deg.
namespace GCS {
constexpr double rs = 696000., au = 1.496e13, pi = CTFRNormal::pi;
// Keep circular normal sections away from the coordinate fold at kappa*a=1.
constexpr double curvatureMargin = .8;
struct Input {
    int cmeType = 1; // 1=regular normal-section FRiED; 2=GCS with smooth joins.
    double speed = 0, fitHeight = 0, targetHeight = 70, gamma = 1e-8, wind = 450;
    double latitude = 0, longitude = 0, tilt = 0, halfAngle = 0, aspectRatio = 0;
    double poloidalFlux = std::numeric_limits<double>::quiet_NaN();
    double toroidalFlux = std::numeric_limits<double>::quiet_NaN();
    int leadingPolarity = 1, helicity = 1;
    CalendarEpoch fitTime;
    std::string earthTrajectory = "trajEarth.dat";
};
struct Result {
    CMEParameters cme;
    CalendarEpoch epoch;
    double transitSeconds = 0, poloidalFlux = 0, toroidalFlux = 0;
    double fullPoloidalFlux = 0, domainInnerFlux = 0, domainOuterFlux = 0;
    double originalMaxCurvatureRadius = 0, fittedHalfAngle = 0, originalAspectRatio = 0;
    bool estimatedPoloidal = false, estimatedToroidal = false;
};

inline CalendarEpoch parseFitTime(std::string raw) {
    // Also accept the ISO and YYYY/MM/DDTHH:MM forms used in GCS fitting files.
    if (raw.size() == 16 && raw[10] == 'T')
        raw += ":00";
    if (raw.size() == 19 && raw[10] == 'T' && (raw[4] == '-' || raw[4] == '/'))
        raw = raw.substr(8, 2) + "-" + raw.substr(5, 2) + "-" + raw.substr(0, 4) + raw.substr(10);
    auto result = parseCalendarEpoch(raw, "GCS_fit_time");
    if (!result.year)
        throw std::invalid_argument("GCS_fit_time must specify an actual UTC epoch");
    return result;
}

inline double simpson(const std::function<double(double)> &f, double a, double b, double fa,
                      double fm, double fb, double whole, double eps, int depth) {
    const double m = .5 * (a + b), l = .5 * (a + m), r = .5 * (m + b), fl = f(l), fr = f(r);
    const double left = (m - a) * (fa + 4 * fl + fm) / 6, right = (b - m) * (fm + 4 * fr + fb) / 6;
    const double delta = left + right - whole;
    if (depth == 0 || std::abs(delta) <= 15 * eps)
        return left + right + delta / 15;
    return simpson(f, a, m, fa, fl, fm, left, eps / 2, depth - 1) +
           simpson(f, m, b, fm, fr, fb, right, eps / 2, depth - 1);
}
inline double integrate(const std::function<double(double)> &f, double eps) {
    const double a = f(0), m = f(.5), b = f(1);
    return simpson(f, 0, 1, a, m, b, (a + 4 * m + b) / 6, eps, 24);
}

inline std::pair<double, double> dbm(double height, double target, double v0, double wind,
                                     double gamma) {
    for (double value : {height, target, v0, wind, gamma})
        if (!std::isfinite(value))
            throw std::invalid_argument("DBM parameters must be finite");
    if (!(height > 0 && target >= height && v0 > 0 && wind >= 0 && gamma >= 0))
        throw std::invalid_argument(
            "DBM requires target>=fit height>0, speed>0, wind>=0 and gamma>=0");
    const double travel = (target - height) * rs, dv = v0 - wind;
    if (travel == 0)
        return {0, v0};
    if (gamma == 0 || dv == 0)
        return {travel / v0, v0};
    auto distance = [&](double t) {
        return wind * t + std::copysign(1., dv) / gamma * std::log1p(gamma * std::abs(dv) * t);
    };
    double lo = 0, hi = std::max(1., travel / v0);
    while (distance(hi) < travel) {
        hi *= 2;
        if (hi > 1e11)
            throw std::invalid_argument("DBM transit exceeds supported time span");
    }
    for (int i = 0; i < 90; ++i) {
        const double mid = .5 * (lo + hi);
        if (distance(mid) < travel)
            lo = mid;
        else
            hi = mid;
    }
    const double t = .5 * (lo + hi);
    return {t, wind + dv / (1 + gamma * std::abs(dv) * t)};
}

inline double earthLongitude(const std::string &file, CalendarEpoch epoch) {
    std::ifstream input(file);
    if (!input)
        throw std::invalid_argument("Unable to open GCS Earth trajectory: " + file);
    std::string line;
    double previousTime = 0, previousLon = 0;
    bool have = false;
    const double target = secondsBetweenEpochs({1, 0}, epoch);
    while (std::getline(input, line)) {
        std::istringstream row(line);
        int year;
        double day, radius, latitude, lon;
        if (!(row >> year >> day >> radius >> latitude >> lon))
            continue;
        if (year < 1 || !(day >= 1 && day <= 367) || !std::isfinite(lon))
            throw std::invalid_argument("Invalid GCS Earth trajectory row");
        const double time = secondsBetweenEpochs({1, 0}, {year, (day - 1) * 86400.L});
        if (have && time <= previousTime)
            throw std::invalid_argument("GCS trajectory times must increase");
        if (have)
            lon = previousLon + std::remainder(lon - previousLon, 360.);
        if (time == target)
            return lon;
        if (time > target) {
            if (!have)
                break;
            return previousLon +
                   (lon - previousLon) * (target - previousTime) / (time - previousTime);
        }
        previousTime = time;
        previousLon = lon;
        have = true;
    }
    throw std::invalid_argument(
        "GCS fit time lies outside the Earth trajectory; no longitude extrapolation");
}

inline double fittedHalfAngle(double hw, double ratio, double n) {
    // Outer GCS outline and one-degree half-width search follow the supplied
    // FRIED_HW_func. Use analytic axis normals and the actual leg endpoint.
    const double alpha = hw * pi / 180, leg = (1 - ratio) / (1 + std::sin(alpha));
    const double rho = leg * std::sin(alpha), rt = 1 / (1 + ratio), rp = ratio * rt;
    std::vector<std::pair<double, double>> outline{{0, 0}};
    for (int i = 0;; ++i) {
        const double beta = (-hw + .01 * i) * pi / 180;
        if (beta >= pi / 2)
            break;
        const double bc = (rho + leg * ratio * ratio * std::sin(beta)) / (1 - ratio * ratio);
        const double cp = std::sqrt(
            std::max(0., bc * bc + (leg * leg * ratio * ratio - rho * rho) / (1 - ratio * ratio)));
        const double bp = bc + cp;
        outline.push_back({bp * std::sin(beta) + leg, bp * std::cos(beta)});
    }
    std::stable_sort(outline.begin(), outline.end());
    auto gcsY = [&](double x) {
        auto hi = std::lower_bound(outline.begin(), outline.end(),
                                   std::make_pair(x, -std::numeric_limits<double>::infinity()));
        if (hi == outline.begin())
            return hi->second;
        if (hi == outline.end())
            return outline.back().second;
        auto lo = hi - 1;
        const double delta = hi->first - lo->first;
        return delta > 0 ? lo->second + (hi->second - lo->second) * (x - lo->first) / delta
                         : hi->second;
    };
    double best = hw, bestScore = std::numeric_limits<double>::infinity();
    for (double candidate = hw; candidate < 90; candidate += 1) {
        CTFRNormal::Axis axis;
        axis.major = rt;
        axis.minor = rp;
        axis.width = candidate * pi / 180;
        axis.flattening = n;
        double score = 0;
        for (int i = 0; i <= int(candidate); ++i) {
            if (i >= candidate - 1e-9)
                continue; // both zero-radius leg endpoints are at the origin
            const auto g = axis.geometry(i * pi / 180);
            const double x = g.x + g.radius * g.nx, y = g.y + g.radius * g.ny;
            score += std::abs(gcsY(x) - y);
        }
        if (score < bestScore) {
            best = candidate;
            bestScore = score;
        }
    }
    return best;
}

struct AlongAxis {
    double radius, minor, radialNormal, lengthDensity;
};
inline AlongAxis alongAxis(double u, const CMEParameters &c) {
    // radius/Rt=1-u^2 removes the integrable singularity at the rooted legs.
    const double q = 1 - u * u, k = pi / (2 * c.halfangle * pi / 180);
    double angular = 0;
    if (u == 0)
        angular = c.r1 * std::sqrt(2 / c.FRiED_n) / k;
    else if (q > 0) {
        const double t = std::pow(q, 1 / c.FRiED_n);
        angular = c.r1 * 2 * u / (c.FRiED_n * k) * t / std::sqrt(std::max(1e-300, 1 - t * t));
    }
    const double length = std::hypot(2 * c.r1 * u, angular), radius = c.r1 * q;
    return {radius, c.r0 * q, length > 0 ? radius * angular / length : 0, length};
}
inline double radialFluxFraction(double radius, double a, double radialNormal, double inner,
                                 double b, int sign) {
    if (a == 0)
        return 0;
    const double discriminant = inner * inner - radius * radius + radialNormal * radialNormal;
    if (discriminant <= 0)
        return 1;
    const double center = -sign * radialNormal, root = std::sqrt(discriminant);
    const double lo = std::max(0., center - root), hi = std::min(a, center + root);
    if (hi <= lo)
        return 1;
    const double excluded =
        (std::log1p(b * b * hi * hi) - std::log1p(b * b * lo * lo)) / std::log1p(b * b * a * a);
    return std::max(0., std::min(1., 1 - excluded));
}
inline double tubeHalfIntegral(const GCSTube::Tube &tube,
                               const std::function<double(const GCSTube::Section &)> &f,
                               double eps) {
    // Split at the transition endpoints, including its actual centerline
    // arclength. This also resolves narrow joins for small aspect ratios.
    const double frontExtent = tube.frontEnd - tube.joinHalfWidth / tube.rho;
    const double legExtent = tube.joinLength - tube.joinHalfWidth, span = 2 * tube.joinHalfWidth;
    const double front = frontExtent * integrate(
                                           [&](double u) {
                                               const auto g = tube.front(u * frontExtent);
                                               return g.lengthDensity * f(g);
                                           },
                                           eps / (3 * frontExtent));
    const double join = span * integrate(
                                   [&](double u) {
                                       const auto g =
                                           tube.joinSection(span * u - tube.joinHalfWidth, 1);
                                       return g.lengthDensity * f(g);
                                   },
                                   eps / (3 * span));
    const double leg = legExtent * integrate(
                                       [&](double u) {
                                           const auto g = tube.leg(u * legExtent, 1);
                                           return g.lengthDensity * f(g);
                                       },
                                       eps / (3 * legExtent));
    return front + join + leg;
}
inline double fluxRatio(const CMEParameters &c, double inner, double b, int sign) {
    if (c.fieldModel == 2) {
        auto fraction = [&](const GCSTube::Section &g) {
            return radialFluxFraction(std::hypot(g.x, g.y), g.radius, g.x * g.ex + g.y * g.ey,
                                      inner, b, sign);
        };
        return b / pi * tubeHalfIntegral(c.gcsTube, fraction, 1e-9 * c.r1);
    }
    auto integrand = [&](double u) {
        auto g = alongAxis(u, c);
        return g.lengthDensity *
               radialFluxFraction(g.radius, g.minor, g.radialNormal, inner, b, sign);
    };
    return b / pi * integrate(integrand, 1e-9 * c.r1); // both symmetric legs
}

inline double axisLength(const CMEParameters &c) {
    if (c.fieldModel == 2) {
        return 2 * tubeHalfIntegral(
                       c.gcsTube, [](const GCSTube::Section &) { return 1.; }, 1e-10 * c.r1);
    }
    return 2 * integrate([&](double u) { return alongAxis(u, c).lengthDensity; }, 1e-10 * c.r1);
}

inline Result convert(const Input &in, CalendarEpoch start, double innerRsun) {
    for (double value : {in.speed, in.fitHeight, in.targetHeight, in.gamma, in.wind, in.latitude,
                         in.longitude, in.tilt, in.halfAngle, in.aspectRatio, innerRsun})
        if (!std::isfinite(value))
            throw std::invalid_argument("GCS and DBM parameters must be finite");
    if (!in.fitTime.year || !start.year || !(innerRsun > 0))
        throw std::invalid_argument(
            "GCS conversion requires valid fit/BC epochs and a positive inner radius");
    if (!(in.halfAngle > 0 && in.halfAngle < 90 && in.aspectRatio > 0 && in.aspectRatio < 1 &&
          std::abs(in.latitude) <= 90 && (in.leadingPolarity == 1 || in.leadingPolarity == -1) &&
          (in.helicity == 1 || in.helicity == -1)))
        throw std::invalid_argument("Invalid GCS half angle, aspect ratio, latitude or polarity");
    if (in.cmeType != 1 && in.cmeType != 2)
        throw std::invalid_argument("CME_type must be 1 (FRiED) or 2 (GCS)");
    Result result;
    auto &c = result.cme;
    c.fieldModel = in.cmeType;
    c.helicity_sign = in.helicity;
    c.lat = in.latitude;
    result.originalAspectRatio = in.aspectRatio;
    c.r1 = in.targetHeight / (1 + in.aspectRatio);
    c.r0 = in.aspectRatio * c.r1;
    if (in.cmeType == 2) {
        c.halfangle = in.halfAngle;
        c.gcsTube.prepare(in.targetHeight, in.halfAngle, in.aspectRatio);
    } else {
        c.FRiED_n = in.halfAngle / 100;
        result.fittedHalfAngle = c.halfangle =
            fittedHalfAngle(in.halfAngle, in.aspectRatio, c.FRiED_n);
        CTFRNormal::Axis original;
        original.major = c.r1;
        original.minor = c.r0;
        original.width = c.halfangle * pi / 180;
        original.flattening = c.FRiED_n;
        for (int i = 0; i <= 32768; ++i) {
            auto g = original.geometry(original.width * .9999999999 * (2. * i / 32768 - 1));
            result.originalMaxCurvatureRadius =
                std::max(result.originalMaxCurvatureRadius, g.curvature * g.radius);
        }
        const double ratio =
            in.aspectRatio * std::min(1., curvatureMargin / result.originalMaxCurvatureRadius);
        c.r1 = in.targetHeight / (1 + ratio);
        c.r0 = in.targetHeight - c.r1;
        c.normalAxis.prepare(c.r1, c.r0, c.halfangle, c.FRiED_n);
    }
    auto transit = dbm(in.fitHeight, in.targetHeight, in.speed, in.wind, in.gamma);
    result.transitSeconds = transit.first;
    c.apex_speed = transit.second;
    result.epoch = advanceCalendarEpoch(in.fitTime, transit.first);
    c.enterTime = result.epoch.decimalYear();
    c.insertionTime = secondsBetweenEpochs(start, result.epoch);
    c.referenceTime = c.insertionTime;
    c.dragGamma = in.gamma;
    c.windSpeed = in.wind;
    if (c.insertionTime < 0)
        throw std::invalid_argument("Derived GCS insertion precedes BC_start_time");
    c.lon = in.longitude + earthLongitude(in.earthTrajectory, in.fitTime) + 181.02;
    c.lon = std::fmod(std::fmod(c.lon, 360.) + 360., 360.);
    c.tilt = in.tilt + (in.leadingPolarity < 0 ? (in.tilt >= 0 ? -180 : 180) : 0);
    result.estimatedPoloidal = std::isnan(in.poloidalFlux) || in.poloidalFlux < 0;
    result.estimatedToroidal = std::isnan(in.toroidalFlux) || in.toroidalFlux < 0;
    // Match Programs/create_input_files_v3.py: Sindhuja & Gopalswamy 2020,
    // v=438.9*(Phi_pol/1e21)^.34 km/s, using the pre-DBM GCS speed.
    // https://doi.org/10.3847/1538-4357/ab620f
    result.poloidalFlux =
        result.estimatedPoloidal ? 1e21 * std::pow(in.speed / 438.9, 1 / .34) : in.poloidalFlux;
    // Match that script's Qiu et al. 2007 reconnection-flux regression,
    // Phi_t=.32*Phi_RC^1.20, treating Phi_pol ~= Phi_RC (units 1e21 Mx).
    // https://solar.physics.montana.edu/qiuj/publication/apj_flux.pdf
    result.toroidalFlux = result.estimatedToroidal
                              ? 1e21 * .32 * std::pow(result.poloidalFlux / 1e21, 1.20)
                              : in.toroidalFlux;
    if (!(result.poloidalFlux > 0 && result.toroidalFlux > 0 &&
          std::isfinite(result.poloidalFlux) && std::isfinite(result.toroidalFlux)))
        throw std::invalid_argument("CME fluxes must be positive magnitudes; omit a flux or use a "
                                    "negative value to estimate it");
    const double length = axisLength(c);
    double b = 2 * pi * result.poloidalFlux / (result.toroidalFlux * length);
    {
        auto ratioAt = [&](double trial) {
            return .5 * (fluxRatio(c, innerRsun, trial, -1) + fluxRatio(c, innerRsun, trial, 1));
        };
        const double wanted = result.poloidalFlux / result.toroidalFlux;
        double lo = 0, hi = b;
        while (ratioAt(hi) < wanted) {
            hi *= 2;
            if (hi > 1e8)
                throw std::invalid_argument(
                    "Cannot normalize GCS poloidal flux in the computational domain");
        }
        for (int i = 0; i < 55; ++i) {
            double mid = .5 * (lo + hi);
            if (ratioAt(mid) < wanted)
                lo = mid;
            else
                hi = mid;
        }
        b = .5 * (lo + hi);
    }
    c.pol_flux_control = b * au / (rs * 1e5);
    c.a1 = result.toroidalFlux / (rs * rs * 1e10) * b * b / (pi * std::log1p(b * b * c.r0 * c.r0));
    result.fullPoloidalFlux = result.toroidalFlux * b * length / (2 * pi);
    result.domainInnerFlux = result.toroidalFlux * fluxRatio(c, innerRsun, b, -1);
    result.domainOuterFlux = result.toroidalFlux * fluxRatio(c, innerRsun, b, 1);
    return result;
}
} // namespace GCS

// ------------------------- Shared source physics -------------------------

namespace CMEModel {
using Vector = std::array<double, 3>;
// Cartesian field [G], velocity [cm/s] and added source density [g/cm^3].
// sample() sets the minimum density; loadDensity() increases it when needed.
struct Sample {
    Vector field{}, velocity{};
    double density = 0;
};

// The same initialization rule is used for every CME. Allow vA up to four
// times the local driver speed to limit added mass. This numerical loading
// threshold is not a measured mass or an empirical law.
// It is deliberately not a per-CME input knob. Evolved interior density is
// subsequently governed by the continuity equation, without a speed cap.
constexpr double minimumAlfvenMach = 0.25;

// Required added density [g/cm^3], using B_total [G] and radial speed [cm/s]:
// rho_total = max(rho_ambient + rho_min, B_total^2 / (4*pi*(4*u_radial)^2)).
inline double loadedDensity(double minimumAddedDensity, double ambientDensity,
                            double totalFieldSquared, double radialSpeed) {
    const double targetAlfvenSpeed = radialSpeed / minimumAlfvenMach;
    const double requiredTotalDensity =
        totalFieldSquared / (4 * CTFRNormal::pi * targetAlfvenSpeed * targetAlfvenSpeed);
    return std::max(minimumAddedDensity, requiredTotalDensity - ambientDensity);
}

// Apply the common loading rule after the driver velocity has been selected.
// The volume driver is purely radial. The boundary driver passes its radial
// speed explicitly, retaining the ambient transverse velocity.
inline void loadDensity(Sample &source, const std::array<double, 8> &ambient, double radialSpeed) {
    double totalFieldSquared = 0;
    for (int d = 0; d < 3; ++d) {
        const double field = ambient[5 + d] + source.field[d];
        totalFieldSquared += field * field;
    }
    source.density = loadedDensity(source.density, ambient[0], totalFieldSquared, radialSpeed);
}

inline void loadDensity(Sample &source, const std::array<double, 8> &ambient) {
    double speedSquared = 0;
    for (double velocity : source.velocity)
        speedSquared += velocity * velocity;
    loadDensity(source, ambient, std::sqrt(speedSquared));
}

// Each CME fades independently from full boundary forcing at insertion to
// zero at boundaryTaperTime. The quintic has zero first and second derivatives
// at both ends. Physical time alone determines the weight, including on restart.
inline double boundaryTaperWeight(const CMEParameters &source, double time) {
    const double age = time - source.insertionTime;
    if (age < 0 || age >= source.boundaryTaperTime)
        return 0;
    const double remaining = 1 - age / source.boundaryTaperTime;
    return remaining * remaining * remaining * (10 + remaining * (-15 + 6 * remaining));
}

// Fade primitive perturbations, then let increment() rebuild their energy.
// Load the full source density before fading it, so the added mass also tends
// continuously to zero instead of imposing a residual loading floor on WSA.
inline void taperBoundarySource(Sample &source, const std::array<double, 8> &ambient,
                                 double weight) {
    source.density *= weight;
    for (int d = 0; d < 3; ++d) {
        source.field[d] *= weight;
        source.velocity[d] = ambient[1 + d] + weight * (source.velocity[d] - ambient[1 + d]);
    }
}

// The geometry and B are prescribed at referenceTime. Isotropic expansion
// X=a(t)*X_ref, B=B_ref/a^2 is an exact solution of ideal induction for
// v=(da/dt)/a*X. In particular, rooted legs do not move at the apex speed.
inline bool sample(const CMEParameters &c, const Vector &position, Sample &result) {
    const double scale = c.geometryScale;
    if (!(scale > 0))
        return false;
    const double lat = -c.lat * CTFRNormal::pi / 180, lon = c.lon * CTFRNormal::pi / 180;
    const double tilt = c.tilt * CTFRNormal::pi / 180;
    const double cl = std::cos(lat), sl = std::sin(lat), co = std::cos(lon), so = std::sin(lon);
    const double ct = std::cos(tilt), st = std::sin(tilt);
    const double rz[3][3] = {{co, so, 0}, {-so, co, 0}, {0, 0, 1}};
    const double ry[3][3] = {{cl, 0, -sl}, {0, 1, 0}, {sl, 0, cl}};
    const double rx[3][3] = {{1, 0, 0}, {0, ct, st}, {0, -st, ct}};
    double rotation[3][3]{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q)
                    rotation[i][j] += rx[i][p] * ry[p][q] * rz[q][j];
    Vector local{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            local[i] += rotation[i][j] * position[j] / c_SR;
    CTFRNormal::Sample field;
    const double twist = c.pol_flux_control * c_SR / 1.496e13;
    const bool inside =
        c.fieldModel == 2
            ? c.gcsTube.sample(local[0] / scale, local[1] / scale, local[2] / scale, twist, c.a1,
                               int(c.helicity_sign), field)
            : c.normalAxis.sample(local[0] / scale, local[1] / scale, local[2] / scale, twist, c.a1,
                                  int(c.helicity_sign), field);
    if (!inside)
        return false;
    result = {};
    for (int i = 0; i < 3; ++i) {
        // Retain the existing axial-polarity convention; inverse=transpose.
        for (int j = 0; j < 3; ++j)
            result.field[i] -= rotation[j][i] * field.field[j] / (scale * scale);
        result.velocity[i] = c.apex_speed * 1e5 * position[i] / (c_SR * (c.r1 + c.r0) * scale);
    }
    result.density = c.density / (scale * scale * scale);
    return true;
}

// Continue the fitted DBM trajectory both forward and back from the reference
// height. A backward extrapolation is valid only while the speed stays outward.
inline bool trajectory(const CMEParameters &source, double elapsed, double &height, double &speed) {
    const double dv = source.apex_speed - source.windSpeed;
    double distance = source.apex_speed * elapsed;
    speed = source.apex_speed;
    if (source.dragGamma > 0 && dv != 0) {
        const double argument = source.dragGamma * std::abs(dv) * elapsed;
        if (!(argument > -1))
            return false;
        distance = source.windSpeed * elapsed +
                   std::copysign(1., dv) * std::log1p(argument) / source.dragGamma;
        speed = source.windSpeed + dv / (1 + argument);
    }
    height = source.r1 + source.r0 + distance * 1e5 / c_SR;
    return std::isfinite(height) && std::isfinite(speed) && speed > 0;
}

inline double boundaryStartTime(const CMEParameters &source, double innerRsun) {
    double lower = -(source.r1 + source.r0 - innerRsun) * c_SR / (source.apex_speed * 1e5),
           upper = 0;
    const double dv = source.apex_speed - source.windSpeed;
    if (source.dragGamma == 0 || dv == 0)
        return source.referenceTime + lower;
    if (source.dragGamma > 0 && dv != 0) {
        if (dv > 0)
            lower = -(1 - 1e-12) / (source.dragGamma * dv);
        else
            lower = (std::abs(dv) / source.windSpeed - 1) / (source.dragGamma * std::abs(dv));
    }
    double height, speed;
    // The lower limit for slow CMEs is their zero-speed turning point.
    // Evaluate just above that limit to retain an outward trajectory.
    if (!trajectory(source, lower * (1 - 1e-12), height, speed) || height > innerRsun + 1e-8)
        throw std::invalid_argument(
            "DBM cannot be extrapolated outward from the inner boundary to this GCS fit");
    lower *= 1 - 1e-12;
    for (int i = 0; i < 100; ++i) {
        const double mid = .5 * (lower + upper);
        if (!trajectory(source, mid, height, speed) || height < innerRsun)
            lower = mid;
        else
            upper = mid;
    }
    return source.referenceTime + .5 * (lower + upper);
}

inline bool expandedSource(const CMEParameters &source, double time, CMEParameters &current) {
    if (time < source.insertionTime)
        return false;
    current = source;
    double height, speed;
    if (!trajectory(source, time - source.referenceTime, height, speed))
        return false;
    current.geometryScale = height / (source.r1 + source.r0);
    current.apex_speed = speed;
    return current.geometryScale > 0;
}

// A new boundary field can be stronger than every owned cell. Include its
// crossing rate in the CFL estimate before it has entered the grid.
// Loading gives vA_full <= u/minimumAlfvenMach. With taper weight w, convex
// mixing bounds vA by vA_ambient + sqrt(w)*vA_full; velocity adds at most w*u.
// The beta term also bounds the extra sound speed from source heating.
inline double boundaryCFLRate(double interiorRate, const std::vector<CMEParameters> &sources,
                              double time, double innerRadius, double crossingWidth,
                              double gamma) {
    double rate = interiorRate;
    for (const auto &source : sources) {
        const double weight = boundaryTaperWeight(source, time);
        if (source.injectionMethod != 1 || weight <= 0) continue;
        CMEParameters current;
        if (!expandedSource(source, time, current)) continue;
        const double height = c_SR * (current.r1 + current.r0) * current.geometryScale;
        const double speed = current.apex_speed * 1e5 * innerRadius / height;
        const double rootWeight = std::sqrt(weight);
        const double thermal = std::sqrt(.5 * gamma * source.plasmaBeta);
        rate = rate * (1 + thermal * rootWeight) + speed / crossingWidth *
            (weight + rootWeight * (1 + thermal) / minimumAlfvenMach);
    }
    return rate;
}

// Gas pressure after adding source mass at the ambient temperature.
inline double thermalPressure(const std::array<double, 8> &ambient, const Sample &source,
                              double beta) {
    double sourceMagneticPressure = 0;
    for (double field : source.field)
        sourceMagneticPressure += field * field / (8 * CTFRNormal::pi);
    return std::max(0., ambient[4]) * (ambient[0] + source.density) / ambient[0] +
           beta * sourceMagneticPressure;
}

// Cartesian primitives -> the conservative change inside the source mask.
// Accelerating the plasma supplies kinetic energy; it never consumes its
// thermal energy. Added mass starts at the ambient temperature. Adding
// the complete tangent, solenoidal rope field avoids deleting ambient B_n at
// its surface; this is superposition, not ambient-field displacement.
inline std::array<double, 8> increment(const std::array<double, 8> &old, const Sample &cme,
                                       double gamma, double beta) {
    std::array<double, 8> delta{};
    const double density = old[0] + cme.density;
    double oldKinetic = 0, newKinetic = 0, oldMagnetic = 0, newMagnetic = 0, cmeMagnetic = 0;
    delta[0] = cme.density;
    for (int d = 0; d < 3; ++d) {
        delta[1 + d] = density * cme.velocity[d] - old[0] * old[1 + d];
        delta[5 + d] = cme.field[d];
        oldKinetic += .5 * old[0] * old[1 + d] * old[1 + d];
        newKinetic += .5 * density * cme.velocity[d] * cme.velocity[d];
        oldMagnetic += old[5 + d] * old[5 + d] / (8 * CTFRNormal::pi);
        newMagnetic +=
            (old[5 + d] + cme.field[d]) * (old[5 + d] + cme.field[d]) / (8 * CTFRNormal::pi);
        cmeMagnetic += cme.field[d] * cme.field[d] / (8 * CTFRNormal::pi);
    }
    const double addedPressure = std::max(0., old[4]) * cme.density / old[0] + beta * cmeMagnetic;
    delta[4] = addedPressure / (gamma - 1) + newKinetic - oldKinetic + newMagnetic - oldMagnetic;
    return delta;
}
} // namespace CMEModel

// ------------------------- Event scheduling -------------------------

class CMEInsertionSchedule {
  public:
    CMEInsertionSchedule(bool a_enabled, double a_insertionTime)
        : m_times(a_enabled ? std::vector<double>{a_insertionTime} : std::vector<double>{}),
          m_boundary(m_times.size(), false), m_feedingEnd(m_times) {}

    explicit CMEInsertionSchedule(const std::vector<CMEParameters> &sources) {
        for (const auto &source : sources) {
            m_times.push_back(source.insertionTime);
            m_boundary.push_back(source.injectionMethod == 1);
            m_feedingEnd.push_back(source.insertionTime + source.boundaryTaperTime);
        }
        if (!std::is_sorted(m_times.begin(), m_times.end()))
            throw std::invalid_argument("CME sources must be ordered by insertion time");
    }

    double time() const {
        return inserted() ? std::numeric_limits<double>::infinity() : m_times[m_next];
    }
    bool inserted() const {
        return m_next == m_times.size();
    }
    size_t index() const {
        return m_next;
    }
    int countInserted() const {
        return static_cast<int>(m_next);
    }
    // A volume insertion is instantaneous. Boundary emergence continues
    // injecting for its full feeding interval. Stabilize through the end of
    // that interval and ten more minutes, including overlapping sources.
    double rusanovEndTime() const {
        double end = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < m_next; ++i) end = std::max(end, stabilizationEnd(i));
        return end;
    }
    // Volume: 20% for ten minutes, then return to the input CFL over 30 minutes.
    // Emergence: ramp to a safe fraction while feeding, then recover the input
    // value over 30 minutes after stabilization. All limits scale with target.
    double cfl(double time, double target, double feedingFraction = .5) const {
        double fraction = 1;
        for (size_t i = 0; i < m_next; ++i) {
            const double early = std::clamp((time - m_times[i] - 600) / 1800., 0., 1.);
            double sourceFraction = .2 + .8 * early;
            if (m_boundary[i]) {
                const double recovery = std::clamp((time - stabilizationEnd(i)) / 1800., 0., 1.);
                sourceFraction = .2 + (feedingFraction - .2) * early + (1 - feedingFraction) * recovery;
            }
            fraction = std::min(fraction, sourceFraction);
        }
        return target * fraction;
    }
    void markInserted() {
        if (!inserted())
            ++m_next;
    }
    void setInserted(bool value) {
        m_next = value ? m_times.size() : 0;
    }
    void restore(int count, double restartTime) {
        if (count < -1)
            throw std::invalid_argument("Invalid checkpoint CME count");
        if (count < 0) {
            m_next =
                std::upper_bound(m_times.begin(), m_times.end(), restartTime) - m_times.begin();
        } else {
            if (static_cast<size_t>(count) > m_times.size())
                throw std::invalid_argument("Checkpoint CME count exceeds the configured sources");
            m_next = static_cast<size_t>(count);
        }
        for (size_t i = 0; i < m_times.size(); ++i)
            if (m_boundary[i] && m_times[i] < restartTime - 1e-6 &&
                (count < 0 || i >= m_next))
                throw std::invalid_argument(
                    "Checkpoint is after the start of an unrecorded boundary CME; "
                    "use an earlier solar-wind checkpoint or a matching emergence checkpoint");
    }
    bool due(double a_time) const {
        return !inserted() && a_time >= time();
    }

  private:
    std::vector<double> m_times;
    std::vector<bool> m_boundary;
    std::vector<double> m_feedingEnd;
    size_t m_next = 0;
    double stabilizationEnd(size_t i) const {
        return m_boundary[i] ? std::max(m_feedingEnd[i] + 600, m_times[i] + 2400)
                             : m_times[i] + 600;
    }
};
