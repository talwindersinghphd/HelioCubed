#pragma once

#include "Proto.H"
#include "Inputs_Parsing.H"
#include "BoxOp_EulerCubedSphere.H"
#include "MHD_Constants.H"
#include "MHD_Output.H"
#include "MHD_Probe.H"
#include <iomanip>
#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <time.h>
#include <cmath>
#include <limits>
#include <string>
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <cstdio>

double getPhysTime(double a_time)
{
  return advanceCalendarEpoch(ParseInputs::get_BC_start_epoch(), a_time).decimalYear();
}

std::string getPhysTime_duration(double a_time)
{
    // Convert time in seconds to days, hours, minutes, seconds
    int total_seconds = static_cast<int>(a_time);

    int days = total_seconds / 86400;
    total_seconds %= 86400;

    int hours = total_seconds / 3600;
    total_seconds %= 3600;

    int minutes = total_seconds / 60;
    int seconds = total_seconds % 60;

    // Create string representation
    char timeStr[100];
    std::snprintf(timeStr, sizeof(timeStr),
                  "%d:%02d:%02d:%02d",
                  days, hours, minutes, seconds);

    return std::string(timeStr);
}

std::string getPhysTime_date(double a_time)
{
  return formatUTCEpoch(advanceCalendarEpoch(ParseInputs::get_BC_start_epoch(), a_time));
}

inline bool progressUsesCalendarTime()
{
    const int problem=ParseInputs::get_init_condition_type();
    return problem==0 || problem==1;
}

inline std::string formatProgressDt(double dt)
{
    char value[128];
    std::snprintf(value,sizeof(value),progressUsesCalendarTime() ? "%.6f(s)" : "%.6e(s)",dt);
    return value;
}

inline std::string formatProgressTime(double time)
{
    char value[128];
    std::snprintf(value,sizeof(value),"%.3e(s)",time);
    if(progressUsesCalendarTime())
        return std::string(value)+" ("+getPhysTime_duration(time)+") phystime: "+getPhysTime_date(time);
    return value;
}

inline std::string formatProgressElapsed(double seconds)
{
    char value[128];
    std::snprintf(value,sizeof(value),progressUsesCalendarTime()
        ? " Time_taken = %.3f(s)" : "  Time taken = %.3f(s)",seconds);
    return value;
}

inline std::string formatIterationProgress(int iteration,double dt,double time,double seconds)
{
    return "iter = "+std::to_string(iteration)+" dt = "+formatProgressDt(dt)
        +" time = "+formatProgressTime(time)+formatProgressElapsed(seconds);
}

template<
  typename MAP,
  typename T, unsigned int C, MemType MEM, Centering CTR
>
void writeTecplotSlice_plane(
    const MBLevelMap<MAP, MEM>&            a_map,         // PR_NODE: x,y,z at nodes
    const MBLevelBoxData<T, C, MEM, CTR>&  a_data,        // cell-centered variables
    const std::vector<std::string>&        a_varNames,    // size C
    const std::string&                     a_filename,    // one Tecplot .dat
    double                                 a_physTime,
    double                                 a_time,
    int                                     a_iter, 
    int                                     a_axis,        // 0=x, 1=y, 2=z
    double                                  a_value = 0.0  // plane value (default 0)
)
{
    if (DIM != 3) {
        cout << "writePlaneSlice_Tecplot_SingleFile: DIM!=3 not supported\n";
        return;
    }
    if ((int)a_varNames.size() != (int)C) {
        cout << "writePlaneSlice_Tecplot_SingleFile: a_varNames.size()!=C\n";
        return;
    }
    if (a_axis < 0 || a_axis > 2) {
        cout << "writePlaneSlice_Tecplot_SingleFile: a_axis must be 0(x)/1(y)/2(z)\n";
        return;
    }

    const_cast<MBLevelMap<MAP, MEM>&>(a_map).initialize();

    std::ostringstream zbuf;
    zbuf.setf(std::ios::scientific);
    zbuf << std::setprecision(6);

    constexpr int Xc = 0, Yc = 1, Zc = 2;
    const int comp = (a_axis == 0 ? Xc : (a_axis == 1 ? Yc : Zc)); // component being sliced
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    const double tol = 1e-12;

    auto lerp = [](double a,double b,double t){ return a + t*(b-a); };

    auto titleAxis = [&](int ax)->const char*{
        return (ax==0 ? "x" : (ax==1 ? "y" : "z"));
    };

    int patchOrdinal = 0;
    for (auto it : a_data.layout()) {
        const auto& dataPatch = a_data[it];      // cell-centered
        const auto& mapPatch  = a_map.map()[it]; // node-centered

        // Only write valid cells.  a_data may carry interpolated angular
        // ghosts so a plane can be located robustly, but emitting those
        // ghosts creates overlapping Tecplot zones at cubed-sphere seams.
        const Box  cellBox = a_data.layout()[it] & dataPatch.box();
        const auto lo = cellBox.low();
        const auto hi = cellBox.high();
        const int ni = hi[0]-lo[0]+1;  // cells in i
        const int nj = hi[1]-lo[1]+1;  // cells in j
        const int nk = hi[2]-lo[2]+1;  // cells in k
        if (ni<=0 || nj<=0 || nk<=0) { ++patchOrdinal; continue; }

        // Skip patches with no plane crossing in their node envelope for chosen axis/value
        double vmin = +1e300, vmax = -1e300;
        {
            const Box mapBox = mapPatch.box();
            for (int k=lo[2]; k<=hi[2]+1; ++k)
            for (int j=lo[1]; j<=hi[1]+1; ++j)
            for (int i=lo[0]; i<=hi[0]+1; ++i) {
                Point pn(i,j,k);
                if (!cellBox.containsPoint(pn)) continue;
                const double v = mapPatch(pn, comp);
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
        }
        if (vmin > a_value + tol || vmax < a_value - tol) { ++patchOrdinal; continue; }

        const int blockId = a_data.layout().block(it);
        const bool isZface = (blockId == 0 || blockId == 1); // your mapping: 0/1 are ±Z

        // Choose marching direction and node grid based on your rules
        enum MarchDir { MARCH_I, MARCH_J, MARCH_K };
        MarchDir march;
        // Default grid sizes and lin indexers (set inside branches)
        int Nin=0, Njn=0, Nkn=0;

        // axis rules:
        // - Z plane: always march K (original behavior)
        // - Y plane: march J on non-±Z, march K on ±Z
        // - X plane: march I on non-±Z, march K on ±Z (rule-of-thumb)
        if (a_axis == Zc) {
            march = MARCH_K;
        } else if (a_axis == Yc) {
            march = isZface ? MARCH_K : MARCH_J;
        } else { // a_axis == Xc
            march = MARCH_J;
        }

        const Box  boxMap  = mapPatch.box();
        const auto mhi = boxMap.high();

        auto nodeClamped = [&](int ii, int jj, int kk) {
            int ci = std::min(ii, mhi[0]);
            int cj = std::min(jj, mhi[1]);
            int ck = std::min(kk, mhi[2]);
            return Point(ci, cj, ck);
        };

        // Common helpers for cell-center averages of map coords
        auto avg8_at = [&](int i,int j,int k,int component)->double{
            Point b00 = nodeClamped(lo[0]+i  , lo[1]+j  , lo[2]+k);
            Point b10 = nodeClamped(lo[0]+i+1, lo[1]+j  , lo[2]+k);
            Point b01 = nodeClamped(lo[0]+i  , lo[1]+j+1, lo[2]+k);
            Point b11 = nodeClamped(lo[0]+i+1, lo[1]+j+1, lo[2]+k);
            Point t00 = nodeClamped(lo[0]+i  , lo[1]+j  , lo[2]+k+1);
            Point t10 = nodeClamped(lo[0]+i+1, lo[1]+j  , lo[2]+k+1);
            Point t01 = nodeClamped(lo[0]+i  , lo[1]+j+1, lo[2]+k+1);
            Point t11 = nodeClamped(lo[0]+i+1, lo[1]+j+1, lo[2]+k+1);
            return 0.125*( mapPatch(b00,component)+mapPatch(b10,component)+mapPatch(b01,component)+mapPatch(b11,component)
                         + mapPatch(t00,component)+mapPatch(t10,component)+mapPatch(t01,component)+mapPatch(t11,component) );
        };

        // ============ Three branches: march K / march J / march I ============
        if (march == MARCH_K) {
            // Grid (i,j), cells ni*nj
            Nin = ni + 1; Njn = nj + 1;
            auto linN = [&](int i,int j){ return j*Nin + i; }; // nodes (i,j)
            auto linC = [&](int i,int j){ return j*ni  + i;  }; // cells (i,j)

            std::vector<double> X(Nin*Njn, NaN), Y(Nin*Njn, NaN), Z(Nin*Njn, NaN);
            // Intersect vertical node columns (vary k) with comp = a_value
            for (int jn=0; jn<Njn; ++jn) {
                for (int in=0; in<Nin; ++in) {
                    const int gi = lo[0] + in;
                    const int gj = lo[1] + jn;
                    const int mi = (gi > mhi[0]) ? mhi[0] : gi;
                    const int mj = (gj > mhi[1]) ? mhi[1] : gj;

                    bool hit=false;
                    for (int k=0; k<nk; ++k) {
                        const int kk  = lo[2] + k;
                        const int kk1 = lo[2] + std::min(k+1, nk);
                        Point p0 = nodeClamped(mi, mj, kk );
                        Point p1 = nodeClamped(mi, mj, kk1);
                        if (!boxMap.containsPoint(p0) || !boxMap.containsPoint(p1)) continue;

                        const double v0 = mapPatch(p0, comp);
                        const double v1 = mapPatch(p1, comp);
                        const bool cross = ( (v0 - a_value)<=+tol && (v1 - a_value)>=-tol ) ||
                                           ( (v0 - a_value)>=-tol && (v1 - a_value)<=+tol );
                        if (!cross) continue;
                        const double denom = (v1 - v0);
                        const double t = (std::fabs(denom) < 1e-30) ? 0.0 : ((a_value - v0)/denom);

                        const double x0 = mapPatch(p0, Xc), x1 = mapPatch(p1, Xc);
                        const double y0 = mapPatch(p0, Yc), y1 = mapPatch(p1, Yc);
                        const double z0 = mapPatch(p0, Zc), z1 = mapPatch(p1, Zc);

                        const int L = linN(in, jn);
                        X[L] = lerp(x0, x1, t);
                        Y[L] = lerp(y0, y1, t);
                        Z[L] = lerp(z0, z1, t);
                        // Pin sliced component numerically to a_value to avoid 1e-15 drift
                        if (comp==Xc) X[L]=a_value; else if (comp==Yc) Y[L]=a_value; else Z[L]=a_value;

                        hit = true; break;
                    }
                    (void)hit;
                }
            }

            // Cell-center interpolation along k
            std::vector<std::vector<double>> VarsCell(C, std::vector<double>(ni*nj, NaN));
            // The plotted footprint is valid-cell only, but use available
            // ghosts in the marching direction. This is required when the
            // requested plane coincides with a patch boundary: neither
            // patch's valid cell centers alone straddle the plane.
            const int kSampleLo = dataPatch.box().low()[2] - lo[2];
            const int kSampleHi = dataPatch.box().high()[2] - lo[2];
            for (int j=0; j<nj; ++j) {
                for (int i=0; i<ni; ++i) {
                    bool have=false;
                    for (int k=kSampleLo; k<kSampleHi; ++k) {
                        double c0 = avg8_at(i,j,k  , comp);
                        double c1 = avg8_at(i,j,k+1, comp);
                        const bool cross = ( (c0 - a_value)<=+tol && (c1 - a_value)>=-tol ) ||
                                           ( (c0 - a_value)>=-tol && (c1 - a_value)<=+tol );
                        if (!cross) continue;
                        const double denom = (c1 - c0);
                        const double t = (std::fabs(denom) < 1e-30) ? 0.0 : ((a_value - c0)/denom);

                        Point cp0(lo[0]+i, lo[1]+j, lo[2]+k  );
                        Point cp1(lo[0]+i, lo[1]+j, lo[2]+k+1);
                        for (unsigned c=0; c<C; ++c) {
                            const double v0 = static_cast<double>(dataPatch(cp0, c));
                            const double v1 = static_cast<double>(dataPatch(cp1, c));
                            VarsCell[c][linC(i,j)] = lerp(v0, v1, t);
                        }
                        have = true; break;
                    }
                    (void)have;
                }
            }

            // Emit zone (i,j)
            zbuf << "ZONE T=\"B" << a_data.layout().block(it) << "_P" << patchOrdinal
                 << "\" DATAPACKING=BLOCK, VARLOCATION=([4-" << (3+(int)C)
                 << "]=CELLCENTERED), I=" << (ni+1) << ", J=" << (nj+1) << "\n";

            auto dumpBlockN = [&](const std::vector<double>& A){
                for (int jj=0; jj<Njn; ++jj) {
                    for (int ii=0; ii<Nin; ++ii) zbuf << A[jj*Nin + ii] << " ";
                    zbuf << "\n";
                }
            };
            auto dumpBlockC = [&](const std::vector<double>& A){
                for (int jj=0; jj<nj; ++jj) {
                    for (int ii=0; ii<ni; ++ii) zbuf << A[jj*ni + ii] << " ";
                    zbuf << "\n";
                }
            };

            dumpBlockN(X); dumpBlockN(Y); dumpBlockN(Z);
            for (unsigned c=0; c<C; ++c) dumpBlockC(VarsCell[c]);
        }
        else if (march == MARCH_J) {
            // Grid (i,k), cells ni*nk
            Nin = ni + 1; Nkn = nk + 1;
            auto linN = [&](int i,int k){ return k*Nin + i; }; // nodes (i,k)
            auto linC = [&](int i,int k){ return k*ni  + i;  }; // cells (i,k)

            std::vector<double> X(Nin*Nkn, NaN), Y(Nin*Nkn, NaN), Z(Nin*Nkn, NaN);
            // Intersect along j at fixed (i,k): comp = a_value
            for (int kn=0; kn<Nkn; ++kn) {
                for (int in=0; in<Nin; ++in) {
                    const int gi = lo[0] + in;
                    const int gk = lo[2] + kn;
                    const int mi = (gi > mhi[0]) ? mhi[0] : gi;
                    const int mk = (gk > mhi[2]) ? mhi[2] : gk;

                    bool hit=false;
                    for (int j=0; j<nj; ++j) {
                        const int jj  = lo[1] + j;
                        const int jj1 = lo[1] + std::min(j+1, nj);
                        Point p0 = nodeClamped(mi, jj , mk);
                        Point p1 = nodeClamped(mi, jj1, mk);
                        if (!boxMap.containsPoint(p0) || !boxMap.containsPoint(p1)) continue;

                        const double v0 = mapPatch(p0, comp);
                        const double v1 = mapPatch(p1, comp);
                        const bool cross = ( (v0 - a_value)<=+tol && (v1 - a_value)>=-tol ) ||
                                           ( (v0 - a_value)>=-tol && (v1 - a_value)<=+tol );
                        if (!cross) continue;
                        const double denom = (v1 - v0);
                        const double t = (std::fabs(denom) < 1e-30) ? 0.0 : ((a_value - v0)/denom);

                        const double x0 = mapPatch(p0, Xc), x1 = mapPatch(p1, Xc);
                        const double y0 = mapPatch(p0, Yc), y1 = mapPatch(p1, Yc);
                        const double z0 = mapPatch(p0, Zc), z1 = mapPatch(p1, Zc);

                        const int L = linN(in, kn);
                        X[L] = lerp(x0, x1, t);
                        Y[L] = lerp(y0, y1, t);
                        Z[L] = lerp(z0, z1, t);
                        if (comp==Xc) X[L]=a_value; else if (comp==Yc) Y[L]=a_value; else Z[L]=a_value;

                        hit = true; break;
                    }
                    (void)hit;
                }
            }

            // Cell-center interpolation along j
            std::vector<std::vector<double>> VarsCell(C, std::vector<double>(ni*nk, NaN));
            const int jSampleLo = dataPatch.box().low()[1] - lo[1];
            const int jSampleHi = dataPatch.box().high()[1] - lo[1];
            for (int k=0; k<nk; ++k) {
                for (int i=0; i<ni; ++i) {
                    bool have=false;
                    for (int j=jSampleLo; j<jSampleHi; ++j) {
                        double c0 = avg8_at(i,j  ,k, comp);
                        double c1 = avg8_at(i,j+1,k, comp);
                        const bool cross = ( (c0 - a_value)<=+tol && (c1 - a_value)>=-tol ) ||
                                           ( (c0 - a_value)>=-tol && (c1 - a_value)<=+tol );
                        if (!cross) continue;
                        const double denom = (c1 - c0);
                        const double t = (std::fabs(denom) < 1e-30) ? 0.0 : ((a_value - c0)/denom);

                        Point cp0(lo[0]+i, lo[1]+j  , lo[2]+k);
                        Point cp1(lo[0]+i, lo[1]+j+1, lo[2]+k);
                        for (unsigned c=0; c<C; ++c) {
                            const double v0 = static_cast<double>(dataPatch(cp0, c));
                            const double v1 = static_cast<double>(dataPatch(cp1, c));
                            VarsCell[c][linC(i,k)] = lerp(v0, v1, t);
                        }
                        have = true; break;
                    }
                    (void)have;
                }
            }

            // Emit zone (i,k)
            zbuf << "ZONE T=\"B" << a_data.layout().block(it) << "_P" << patchOrdinal
                 << "\" DATAPACKING=BLOCK, VARLOCATION=([4-" << (3+(int)C)
                 << "]=CELLCENTERED), I=" << (ni+1) << ", J=" << (nk+1) << "\n";

            auto dumpBlockN = [&](const std::vector<double>& A){
                for (int kk=0; kk<Nkn; ++kk) {
                    for (int ii=0; ii<Nin; ++ii) zbuf << A[kk*Nin + ii] << " ";
                    zbuf << "\n";
                }
            };
            auto dumpBlockC = [&](const std::vector<double>& A){
                for (int kk=0; kk<nk; ++kk) {
                    for (int ii=0; ii<ni; ++ii) zbuf << A[kk*ni + ii] << " ";
                    zbuf << "\n";
                }
            };

            dumpBlockN(X); dumpBlockN(Y); dumpBlockN(Z);
            for (unsigned c=0; c<C; ++c) dumpBlockC(VarsCell[c]);
        }
        else { // march == MARCH_I
            // Grid (j,k), cells nj*nk
            Njn = nj + 1; Nkn = nk + 1;
            auto linN = [&](int j,int k){ return k*Njn + j; }; // nodes (j,k)
            auto linC = [&](int j,int k){ return k*nj  + j;  }; // cells (j,k)

            std::vector<double> X(Njn*Nkn, NaN), Y(Njn*Nkn, NaN), Z(Njn*Nkn, NaN);
            // Intersect along i at fixed (j,k): comp = a_value
            for (int kn=0; kn<Nkn; ++kn) {
                for (int jn=0; jn<Njn; ++jn) {
                    const int gj = lo[1] + jn;
                    const int gk = lo[2] + kn;
                    const int mj = (gj > mhi[1]) ? mhi[1] : gj;
                    const int mk = (gk > mhi[2]) ? mhi[2] : gk;

                    bool hit=false;
                    for (int i=0; i<ni; ++i) {
                        const int ii  = lo[0] + i;
                        const int ii1 = lo[0] + std::min(i+1, ni);
                        Point p0 = nodeClamped(ii , mj, mk);
                        Point p1 = nodeClamped(ii1, mj, mk);
                        if (!boxMap.containsPoint(p0) || !boxMap.containsPoint(p1)) continue;

                        const double v0 = mapPatch(p0, comp);
                        const double v1 = mapPatch(p1, comp);
                        const bool cross = ( (v0 - a_value)<=+tol && (v1 - a_value)>=-tol ) ||
                                           ( (v0 - a_value)>=-tol && (v1 - a_value)<=+tol );
                        if (!cross) continue;
                        const double denom = (v1 - v0);
                        const double t = (std::fabs(denom) < 1e-30) ? 0.0 : ((a_value - v0)/denom);

                        const double x0 = mapPatch(p0, Xc), x1 = mapPatch(p1, Xc);
                        const double y0 = mapPatch(p0, Yc), y1 = mapPatch(p1, Yc);
                        const double z0 = mapPatch(p0, Zc), z1 = mapPatch(p1, Zc);

                        const int L = linN(jn, kn);
                        X[L] = lerp(x0, x1, t);
                        Y[L] = lerp(y0, y1, t);
                        Z[L] = lerp(z0, z1, t);
                        if (comp==Xc) X[L]=a_value; else if (comp==Yc) Y[L]=a_value; else Z[L]=a_value;

                        hit = true; break;
                    }
                    (void)hit;
                }
            }

            // Cell-center interpolation along i
            std::vector<std::vector<double>> VarsCell(C, std::vector<double>(nj*nk, NaN));
            const int iSampleLo = dataPatch.box().low()[0] - lo[0];
            const int iSampleHi = dataPatch.box().high()[0] - lo[0];
            for (int k=0; k<nk; ++k) {
                for (int j=0; j<nj; ++j) {
                    bool have=false;
                    for (int i=iSampleLo; i<iSampleHi; ++i) {
                        double c0 = avg8_at(i  ,j,k, comp);
                        double c1 = avg8_at(i+1,j,k, comp);
                        const bool cross = ( (c0 - a_value)<=+tol && (c1 - a_value)>=-tol ) ||
                                           ( (c0 - a_value)>=-tol && (c1 - a_value)<=+tol );
                        if (!cross) continue;
                        const double denom = (c1 - c0);
                        const double t = (std::fabs(denom) < 1e-30) ? 0.0 : ((a_value - c0)/denom);

                        Point cp0(lo[0]+i  , lo[1]+j, lo[2]+k);
                        Point cp1(lo[0]+i+1, lo[1]+j, lo[2]+k);
                        for (unsigned c=0; c<C; ++c) {
                            const double v0 = static_cast<double>(dataPatch(cp0, c));
                            const double v1 = static_cast<double>(dataPatch(cp1, c));
                            VarsCell[c][linC(j,k)] = lerp(v0, v1, t);
                        }
                        have = true; break;
                    }
                    (void)have;
                }
            }

            // Emit zone (j,k)
            zbuf << "ZONE T=\"B" << a_data.layout().block(it) << "_P" << patchOrdinal
                 << "\" DATAPACKING=BLOCK, VARLOCATION=([4-" << (3+(int)C)
                 << "]=CELLCENTERED), I=" << (nj+1) << ", J=" << (nk+1) << "\n";

            auto dumpBlockN = [&](const std::vector<double>& A){
                for (int kk=0; kk<Nkn; ++kk) {
                    for (int jj=0; jj<Njn; ++jj) zbuf << A[kk*Njn + jj] << " ";
                    zbuf << "\n";
                }
            };
            auto dumpBlockC = [&](const std::vector<double>& A){
                for (int kk=0; kk<nk; ++kk) {
                    for (int jj=0; jj<nj; ++jj) zbuf << A[kk*nj + jj] << " ";
                    zbuf << "\n";
                }
            };

            dumpBlockN(X); dumpBlockN(Y); dumpBlockN(Z);
            for (unsigned c=0; c<C; ++c) dumpBlockC(VarsCell[c]);
        }

        ++patchOrdinal;
    }

    // ---- Gather and write single file on rank 0 ----
    std::string localStr = zbuf.str();
    std::vector<char> localBuf(localStr.begin(), localStr.end());
    int localN = (int)localBuf.size();

    int rank = 0, nprocs = 1;
#ifdef PR_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    std::vector<int> counts, displs;
    if (rank == 0) counts.resize(nprocs);
    MPI_Gather(&localN, 1, MPI_INT,
               (rank == 0 ? counts.data() : nullptr), 1, MPI_INT,
               0, comm);

    std::vector<char> big;
    if (rank == 0) {
        displs.resize(nprocs, 0);
        int tot = 0;
        for (int p = 0; p < nprocs; ++p) {
            displs[p] = tot;
            tot += counts[p];
        }
        big.resize(tot);
        MPI_Gatherv(localBuf.data(), localN, MPI_CHAR,
                    big.data(), counts.data(), displs.data(), MPI_CHAR,
                    0, comm);

        // ---------- Keep only the unique ZONE blocks ----------
        std::string merged(big.begin(), big.end());
        std::string filtered;

        // we use a hash of the entire ZONE text as the signature
        std::unordered_set<std::size_t> seen;
        std::hash<std::string_view> hasher;

        std::size_t pos = 0;
        const std::size_t N = merged.size();

        while (pos < N) {
            // Find next "ZONE"
            std::size_t zoneStart = merged.find("ZONE", pos);
            if (zoneStart == std::string::npos) {
                // no more zones; append any trailing whitespace and break
                filtered.append(merged.substr(pos));
                break;
            }

            // Append anything between pos and zoneStart (usually nothing or whitespace)
            if (zoneStart > pos) {
                filtered.append(merged.substr(pos, zoneStart - pos));
            }

            // Find the start of the next zone (or end of string)
            std::size_t nextZone = merged.find("ZONE", zoneStart + 4);
            std::size_t zoneEnd  = (nextZone == std::string::npos) ? N : nextZone;

            std::string_view zv(merged.data() + zoneStart,
                                zoneEnd - zoneStart);
            auto h = hasher(zv);

            // only keep this ZONE block if we have not seen its hash yet
            if (seen.insert(h).second) {
                filtered.append(zv);
            }

            pos = zoneEnd;
        }
        // -------------------------------------------

        std::ofstream ofs(a_filename, std::ios::out);
        ofs.setf(std::ios::scientific);
        ofs << std::setprecision(6);
        ofs << "TITLE = \"t=" << a_time << " step=" << a_iter << "\"\n";
        ofs << "VARIABLES=\"X\" \"Y\" \"Z\"";
        for (const auto& nm : a_varNames) ofs << " \"" << nm << "\"";
        ofs << "\n";
        ofs << "DATASETAUXDATA PhysTime = \"" << std::setprecision(12)
            << a_physTime << "\"\n";
        ofs << std::setprecision(6);
        ofs.write(filtered.data(), (std::streamsize)filtered.size());
        ofs.close();
    } else {
        MPI_Gatherv(localBuf.data(), localN, MPI_CHAR,
                    nullptr, nullptr, nullptr, MPI_CHAR,
                    0, comm);
    }
#else
    std::ofstream ofs(a_filename, std::ios::out);
    ofs.setf(std::ios::scientific);
    ofs << std::setprecision(6);
    ofs << "TITLE = \"t=" << a_time << " step="<< a_iter<<"\"\n";
    ofs << "VARIABLES=\"X\" \"Y\" \"Z\"";
    for (const auto& nm : a_varNames) ofs << " \"" << nm << "\"";
    ofs << "\n";
    ofs << "DATASETAUXDATA PhysTime = \"" << std::setprecision(12)
        << a_physTime << "\"\n";
    ofs << std::setprecision(6);
    ofs << localStr;
    ofs.close();
#endif
}



// Extract a spherical slice r = a_radius into a single Tecplot file.
// Assumes "i" index marches monotonically outward in radius (increasing r).
// Robustness: we *search along i* for every (j,k) line, so even if r is not strictly
// monotone everywhere, any local crossing between i and i+1 will still be detected.
//
// - Geometry (X,Y,Z) is node-based on a (j,k) structured grid: I = nj+1, J = nk+1
// - Variables are interpolated at cell centers between (i,j,k) and (i+1,j,k): size nj*nk
//
// Usage:
// writeTecplotSlice_spherical(map, data, varNames, "r1.50.dat", time, /*radius=*/1.50);
template<
  typename MAP,
  typename T, unsigned int C, MemType MEM, Centering CTR
>
void writeTecplotSlice_spherical(
    const MBLevelMap<MAP, MEM>&            a_map,        // node-centered geometry: X,Y,Z
    const MBLevelBoxData<T, C, MEM, CTR>&  a_data,       // cell-centered variables
    const std::vector<std::string>&        a_varNames,   // size C
    const std::string&                     a_filename,   // output Tecplot .dat (single file)
    double                                  a_physTime,
    double                                 a_time,
    int                                     a_iter, 
    double                                  a_radius      // target radius r
)
{
    static_assert(DIM==3, "Tecplot slicer assumes DIM==3.");
    if ((int)a_varNames.size() != (int)C) {
        pr_out() << "writeSphericalSlice_Tecplot_SingleFile: a_varNames.size()!=C\n";
        return;
    }

    const_cast<MBLevelMap<MAP, MEM>&>(a_map).initialize();

    std::ostringstream zbuf;
    zbuf.setf(std::ios::scientific);
    zbuf << std::setprecision(6);

    constexpr int Xc = 0, Yc = 1, Zc = 2;
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    const double tol = 1e-12;

    auto lerp = [](double a,double b,double t){ return a + t*(b-a); };
    auto hypot3 = [](double x,double y,double z){ return std::sqrt(x*x + y*y + z*z); };

    int patchOrdinal = 0;
    for (auto it : a_data.layout()) {
        const auto& dataPatch = a_data[it];      // cell-centered
        const auto& mapPatch  = a_map.map()[it]; // node-centered

        // Cell interior box and sizes
        // Do not emit interpolated ghost cells as overlapping physical data.
        const Box  cellBox = a_data.layout()[it] & dataPatch.box();
        const auto lo = cellBox.low();
        const auto hi = cellBox.high();
        const int ni = hi[0]-lo[0]+1;  // cells in i (radial march)
        const int nj = hi[1]-lo[1]+1;  // cells in j
        const int nk = hi[2]-lo[2]+1;  // cells in k
        if (ni<=0 || nj<=0 || nk<=0) { ++patchOrdinal; continue; }

        // Quick reject: check node-envelope radii against target
        double rmin = +1e300, rmax = -1e300;
        {
            const Box mapBox = mapPatch.box();
            for (int k=lo[2]; k<=hi[2]+1; ++k)
            for (int j=lo[1]; j<=hi[1]+1; ++j)
            for (int i=lo[0]; i<=hi[0]+1; ++i) {
                Point pn(i,j,k);
                if (!mapBox.containsPoint(pn)) continue;
                const double x = mapPatch(pn, Xc);
                const double y = mapPatch(pn, Yc);
                const double z = mapPatch(pn, Zc);
                const double r = hypot3(x,y,z);
                if (r < rmin) rmin = r;
                if (r > rmax) rmax = r;
            }
        }
        if (rmin > a_radius + tol || rmax < a_radius - tol) { ++patchOrdinal; continue; }

        const int blockId = a_data.layout().block(it);
        (void)blockId; // (not needed here, but kept for symmetry with plane slicer)

        // Node grid for Tecplot geometry: use (j,k) since we march along i (radial)
        const int Njn = nj + 1, Nkn = nk + 1;
        auto linN = [&](int j,int k){ return k*Njn + j; }; // nodes (j,k)
        auto linC = [&](int j,int k){ return k*nj  + j;  }; // cells (j,k)

        std::vector<double> X(Njn*Nkn, NaN), Y(Njn*Nkn, NaN), Z(Njn*Nkn, NaN);

        const Box  boxMap  = mapPatch.box();
        const auto mhi = boxMap.high();

        auto nodeClamped = [&](int ii, int jj, int kk) {
            int ci = std::min(ii, mhi[0]);
            int cj = std::min(jj, mhi[1]);
            int ck = std::min(kk, mhi[2]);
            return Point(ci, cj, ck);
        };

        // ---------- NODE GEOMETRY: intersect along i for each (j,k) node-line ----------
        for (int kn=0; kn<Nkn; ++kn) {
            for (int jn=0; jn<Njn; ++jn) {
                const int gj = lo[1] + jn;                 // node j (may be hi[1]+1)
                const int gk = lo[2] + kn;                 // node k (may be hi[2]+1)
                const int mj = (gj > mhi[1]) ? mhi[1] : gj;
                const int mk = (gk > mhi[2]) ? mhi[2] : gk;

                bool hit=false;
                for (int i=0; i<ni; ++i) { // segment between i and i+1
                    const int ii  = lo[0] + i;
                    const int ii1 = lo[0] + std::min(i+1, ni);
                    Point p0 = nodeClamped(ii , mj, mk);
                    Point p1 = nodeClamped(ii1, mj, mk);
                    if (!boxMap.containsPoint(p0) || !boxMap.containsPoint(p1)) continue;

                    const double x0 = mapPatch(p0, Xc), x1 = mapPatch(p1, Xc);
                    const double y0 = mapPatch(p0, Yc), y1 = mapPatch(p1, Yc);
                    const double z0 = mapPatch(p0, Zc), z1 = mapPatch(p1, Zc);
                    const double r0 = hypot3(x0,y0,z0);
                    const double r1 = hypot3(x1,y1,z1);

                    const bool cross = ( (r0 - a_radius)<=+tol && (r1 - a_radius)>=-tol ) ||
                                       ( (r0 - a_radius)>=-tol && (r1 - a_radius)<=+tol );
                    if (!cross) continue;

                    const double denom = (r1 - r0);
                    const double t = (std::fabs(denom) < 1e-30) ? 0.0 : ((a_radius - r0)/denom);

                    const int L = linN(jn, kn);
                    X[L] = lerp(x0, x1, t);
                    Y[L] = lerp(y0, y1, t);
                    Z[L] = lerp(z0, z1, t);
                    // Numerically nudge exactly onto the sphere (optional):
                    // const double s = a_radius / hypot3(X[L],Y[L],Z[L]); X[L]*=s; Y[L]*=s; Z[L]*=s;

                    hit = true; break;
                }
                (void)hit; // stays NaN if truly no intersection on this line
            }
        }

        // ---------- CELL-CENTER VARIABLES: interpolate along i at fixed (j,k) ----------
        // Average node map to cell-center (i,j,k) using the 8 corners
        auto avg8_at = [&](int i,int j,int k,int comp)->double{
            Point b00 = nodeClamped(lo[0]+i  , lo[1]+j  , lo[2]+k);
            Point b10 = nodeClamped(lo[0]+i+1, lo[1]+j  , lo[2]+k);
            Point b01 = nodeClamped(lo[0]+i  , lo[1]+j+1, lo[2]+k);
            Point b11 = nodeClamped(lo[0]+i+1, lo[1]+j+1, lo[2]+k);
            Point t00 = nodeClamped(lo[0]+i  , lo[1]+j  , lo[2]+k+1);
            Point t10 = nodeClamped(lo[0]+i+1, lo[1]+j  , lo[2]+k+1);
            Point t01 = nodeClamped(lo[0]+i  , lo[1]+j+1, lo[2]+k+1);
            Point t11 = nodeClamped(lo[0]+i+1, lo[1]+j+1, lo[2]+k+1);
            return 0.125*( mapPatch(b00,comp)+mapPatch(b10,comp)+mapPatch(b01,comp)+mapPatch(b11,comp)
                         + mapPatch(t00,comp)+mapPatch(t10,comp)+mapPatch(t01,comp)+mapPatch(t11,comp) );
        };

        // Helper to get center radius at (i,j,k)
        auto center_r = [&](int i,int j,int k)->double{
            const double xc = avg8_at(i,j,k, Xc);
            const double yc = avg8_at(i,j,k, Yc);
            const double zc = avg8_at(i,j,k, Zc);
            return hypot3(xc,yc,zc);
        };

        std::vector<std::vector<double>> VarsCell(C, std::vector<double>(nj*nk, NaN));

        for (int k=0; k<nk; ++k) {
            for (int j=0; j<nj; ++j) {
                bool have = false;
                for (int i=0; i<ni-1; ++i) {
                    const double r0 = center_r(i  , j, k);
                    const double r1 = center_r(i+1, j, k);

                    const bool cross = ( (r0 - a_radius)<=+tol && (r1 - a_radius)>=-tol ) ||
                                       ( (r0 - a_radius)>=-tol && (r1 - a_radius)<=+tol );
                    if (!cross) continue;

                    const double denom = (r1 - r0);
                    const double tcc   = (std::fabs(denom) < 1e-30) ? 0.0 : ((a_radius - r0)/denom);

                    Point cp0(lo[0]+i  , lo[1]+j, lo[2]+k);
                    Point cp1(lo[0]+i+1, lo[1]+j, lo[2]+k);
                    for (unsigned c=0; c<C; ++c) {
                        const double v0 = static_cast<double>(dataPatch(cp0, c));
                        const double v1 = static_cast<double>(dataPatch(cp1, c));
                        VarsCell[c][linC(j,k)] = lerp(v0, v1, tcc);
                    }
                    have = true;
                    break;
                }
                (void)have; // remains NaN where sphere truly does not cut the column
            }
        }

        // ---- Emit ZONE for this patch (structured (j,k)) ----
        zbuf << "ZONE T=\"B" << a_data.layout().block(it) << "_P" << patchOrdinal
             << "\" DATAPACKING=BLOCK, VARLOCATION=([4-" << (3+(int)C)
             << "]=CELLCENTERED), I=" << (nj+1) << ", J=" << (nk+1) << "\n";

        // Dump helpers
        auto dumpBlockN = [&](const std::vector<double>& A){
            for (int kk=0; kk<Nkn; ++kk) {
                for (int jj=0; jj<Njn; ++jj) zbuf << A[linN(jj,kk)] << " ";
                zbuf << "\n";
            }
        };
        auto dumpBlockC = [&](const std::vector<double>& A){
            for (int kk=0; kk<nk; ++kk) {
                for (int jj=0; jj<nj; ++jj) zbuf << A[linC(jj,kk)] << " ";
                zbuf << "\n";
            }
        };

        dumpBlockN(X);
        dumpBlockN(Y);
        dumpBlockN(Z);
        for (unsigned c=0; c<C; ++c) dumpBlockC(VarsCell[c]);

        ++patchOrdinal;
    }

    // ---- Gather and write single file on rank 0 ----
    std::string localStr = zbuf.str();
    std::vector<char> localBuf(localStr.begin(), localStr.end());
    int localN = (int)localBuf.size();

    int rank = 0, nprocs = 1;
#ifdef PR_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    std::vector<int> counts, displs;
    if (rank == 0) counts.resize(nprocs);
    MPI_Gather(&localN, 1, MPI_INT, (rank==0?counts.data():nullptr), 1, MPI_INT, 0, comm);

    std::vector<char> big;
    if (rank == 0) {
        displs.resize(nprocs,0);
        int tot=0; for (int p=0;p<nprocs;++p){ displs[p]=tot; tot+=counts[p]; }
        big.resize(tot);
        MPI_Gatherv(localBuf.data(), localN, MPI_CHAR, big.data(), counts.data(), displs.data(), MPI_CHAR, 0, comm);

        std::ofstream ofs(a_filename, std::ios::out);
        ofs.setf(std::ios::scientific);
        ofs << std::setprecision(6);
        ofs << "TITLE = \"t=" << a_time << " step="<< a_iter<<"\"\n";
        ofs << "VARIABLES=\"X\" \"Y\" \"Z\"";
        for (const auto& nm : a_varNames) ofs << " \"" << nm << "\"";
        ofs << "\n";
        ofs << "DATASETAUXDATA PhysTime = \"" << std::setprecision(12) << a_physTime << "\"\n";
        ofs << std::setprecision(6);
        ofs.write(big.data(), (std::streamsize)big.size());
        ofs.close();
    } else {
        MPI_Gatherv(localBuf.data(), localN, MPI_CHAR, nullptr, nullptr, nullptr, MPI_CHAR, 0, comm);
    }
#else
    std::ofstream ofs(a_filename, std::ios::out);
    ofs.setf(std::ios::scientific);
    ofs << std::setprecision(6);
    ofs << "TITLE = \"t=" << a_time << " step="<< a_iter<<"\"\n";
    ofs << "VARIABLES=\"X\" \"Y\" \"Z\"";
    for (const auto& nm : a_varNames) ofs << " \"" << nm << "\"";
    ofs << "\n";
    ofs << "DATASETAUXDATA PhysTime = \"" << std::setprecision(12) << a_physTime << "\"\n";
    ofs << std::setprecision(6);
    ofs << localStr;
    ofs.close();
#endif
}

template<typename BOX_OP>
void Write_W(MBLevelBoxData<double, NUMCOMPS, HOST> &a_JU,
             BOX_OP &eulerOp,
             MBInterpOp &a_iop,
             int iter,
             double time = 0.0,
             double dt = 0.0,
             bool write_data = false,
             bool write_slice = false)
{
  PR_TIME("Write_W");
  double gamma = ParseInputs::get_gamma();
  HDF5Handler h5;
  a_JU.exchange();
  h5.setTime(time);
	h5.setTimestep(dt);
  h5.setIter(iter);
  typedef BoxOp_EulerCubedSphere<double, MBMap_CubedSphereShell<HOST>, HOST> OP;
  auto layout = a_JU.layout();
  const auto& map = *OP::levelGeometry(layout).map;
  MBLevelBoxData<double, NUMCOMPS, HOST> W_sph_out(layout, Point::Zeros());
  MBLevelBoxData<double, 1, HOST> dVolrLev(layout, OP::ghost() + Point::Basis(0, 2));
  MBLevelBoxData<double, 1, HOST> radiusLev(layout, OP::ghost() + 2);
  for (auto dit : layout)
  {
    BoxData<double, DIM, HOST> Dr(dVolrLev[dit].box());
    BoxData<double, DIM, HOST> adjDr(dVolrLev[dit].box());
    eulerOp[dit].radialMetrics(radiusLev[dit], Dr, adjDr, dVolrLev[dit], Dr.box());
  }
  Point size = layout.domain().getBlock(0).sizes();
  double dxiPerp = 1.0/size[1];
  
  bool infOrNan = false;
  int scale_data_with_r = ParseInputs::get_scale_data_with_r(); 
  int init_condition_type = ParseInputs::get_init_condition_type();
  bool magneticTest = isConstantMagneticFieldTest(init_condition_type);
  double testBx0 = testProblemBackgroundBx(init_condition_type);
  Array<std::string,NUMCOMPS+2> names = {"density", "VxPerp", "VyPerp", "VzPerp", "P", "BxPerp", "ByPerp", "BzPerp","Vr","Br*r^2"};//,"e^entropy"};
  if (scale_data_with_r == 1){
    names[0] = "density_scaled";
    names[4] = "P_scaled";
    names[5] = "BxPerp_scaled";
    names[6] = "ByPerp_scaled";
    names[7] = "BzPerp_scaled";
    names[9] = "Br_scaled"; 
  }
  Array<double, DIM> dx;
  const bool secondOrder = ParseInputs::get_spatial_order() == 2;
  for (auto dit : layout)
  {
    // The fourth-order semispherical split consumes three angular donor
    // layers. O2 has only two and uses a cell-center basis instead; recover
    // its output vectors from the selected-order primitives below.
    if (secondOrder) continue;
    auto& JU_i = a_JU[dit];
    BoxData<double,1,HOST> deta = CubedSphereShell::detA<double,HOST>(JU_i.box(),dxiPerp,0); 
    BoxData<double,NUMCOMPS+2,HOST> USemiSph(layout[dit].grow(Point::Basis(0,2)));
    BoxData<double,NUMCOMPS+2,HOST> Wout_i(layout[dit]);
    USemiSph.setToZero();
    unsigned int block = layout.block(dit);
    Box blockBox = layout.getBlock(block).domain().box();
    auto& dVolr_i = dVolrLev[dit];
    Box bx_i = layout[dit];
    CubedSphereShell::
      consToSemiSphNGEuler(USemiSph,JU_i,dVolr_i,bx_i,
                           blockBox,dxiPerp,block,4);
    dx = eulerOp[dit].dx();
    double half = 0.5;
    BoxData<double, DIM, HOST> XCart = forall_p<double,DIM,HOST>
            (f_cubedSphereMap3,radiusLev[dit].box(),radiusLev[dit],dx,half,half,block);  
    forallInPlace([ ] PROTO_LAMBDA
    (Var<double,NUMCOMPS+2,HOST>& a_Wout,
     Var<double,NUMCOMPS,HOST>& a_W_sph_out,
     const Var<double,NUMCOMPS+2,HOST>& a_USemiSph,
     double a_gamma,
     const Var<double,1,HOST>& a_radius,
     const Var<double,DIM,HOST>& a_Xcart,
     int a_scale_data_with_r,
     bool a_magneticTest,
     double a_Bx0)
    {
      // Primitive variables with vec_non_rad as vec_x_non, vec_y_non, vec_z_non
      a_Wout(0) = a_USemiSph(0);
      a_Wout(1) = a_USemiSph(1) / a_Wout(0);
      a_Wout(2) = a_USemiSph(2) / a_Wout(0);
      a_Wout(3) = a_USemiSph(3) / a_Wout(0);
      a_Wout(5) = a_USemiSph(5);
      a_Wout(6) = a_USemiSph(6);
      a_Wout(7) = a_USemiSph(7);
      a_Wout(8) = a_USemiSph(8) / a_Wout(0);
      a_Wout(9) = a_USemiSph(9);
      double totalEnergy = a_USemiSph(4);
      if (a_magneticTest)
      {
        // Restore the uniform energy and magnetic background before
        // primitive recovery and output conversion.
        totalEnergy += c_TEST_P0/(a_gamma - 1.0)
            + a_Bx0*a_Bx0/(8.0*M_PI);
        const double x = a_Xcart(0);
        const double y = a_Xcart(1);
        const double z = a_Xcart(2);
        const double r = sqrt(x*x + y*y + z*z);
        const double Br0 = a_Bx0*x/r;
        a_Wout(5) += a_Bx0 - Br0*x/r;
        a_Wout(6) += -Br0*y/r;
        a_Wout(7) += -Br0*z/r;
        a_Wout(9) += Br0;
      }
      double ke = a_Wout(0)*(a_Wout(1)*a_Wout(1)
                         + a_Wout(2)*a_Wout(2)
                         + a_Wout(3)*a_Wout(3) + a_Wout(8)*a_Wout(8))*.5;
      double me = (a_Wout(5)*a_Wout(5)
              + a_Wout(6)*a_Wout(6)
              + a_Wout(7)*a_Wout(7) + a_Wout(9)*a_Wout(9))/(8.0*M_PI);
      a_Wout(4) = (totalEnergy - ke - me)*(a_gamma - 1.0);


      a_W_sph_out(iRHO) = a_Wout(0);
      a_W_sph_out(iP) = a_Wout(4);
      // Convert non rad cartesian to spherical velocities
      double x = a_Xcart(0);
      double y = a_Xcart(1);
      double z = a_Xcart(2);
      double r = sqrt(x*x + y*y + z*z);
      double theta = acos(z/r);
      double phi = atan2(y,x);
      // Convert Cartesian to spherical velocities
      a_W_sph_out(iVX) = a_Wout(8);
      a_W_sph_out(iVY) = cos(theta)*cos(phi)*a_Wout(1) + cos(theta)*sin(phi)*a_Wout(2) - sin(theta)*a_Wout(3);
      a_W_sph_out(iVZ) = -sin(phi)*a_Wout(1) + cos(phi)*a_Wout(2);
      // Convert Cartesian to spherical magnetic fields
      a_W_sph_out(iBX) = a_Wout(9);
      a_W_sph_out(iBY) = cos(theta)*cos(phi)*a_Wout(5) + cos(theta)*sin(phi)*a_Wout(6) - sin(theta)*a_Wout(7);
      a_W_sph_out(iBZ) = -sin(phi)*a_Wout(5) + cos(phi)*a_Wout(6);

      if (a_scale_data_with_r == 1){
        a_W_sph_out(iRHO) = a_W_sph_out(iRHO)*a_radius(0)*a_radius(0)/c_AU/c_AU/c_MP; // /cm^3 , scaled by r^2;
        a_W_sph_out(iVX) = a_W_sph_out(iVX)/1e5; // km/s;
        a_W_sph_out(iVY) = a_W_sph_out(iVY)/1e5; // km/s;
        a_W_sph_out(iVZ) = a_W_sph_out(iVZ)/1e5; // km/s;
        a_W_sph_out(iP) = a_W_sph_out(iP)*pow(a_radius(0)/c_AU,a_gamma*2)*1e12;    
        // scaled by r*(2*gamma)
        if (a_Wout(iP) < 0.) a_Wout(iP) = 0.;
        a_W_sph_out(iBX) = a_W_sph_out(iBX)*(a_radius(0)*a_radius(0)/c_AU/c_AU)*1e6; // microGauss
        a_W_sph_out(iBY) = a_W_sph_out(iBY)*(a_radius(0)/c_AU)*1e6; // microGauss
        a_W_sph_out(iBZ) = a_W_sph_out(iBZ)*(a_radius(0)/c_AU)*1e6; // microGauss
      }
      
    }, Wout_i, W_sph_out[dit], USemiSph, gamma, radiusLev[dit], XCart,
       scale_data_with_r, magneticTest, testBx0);

#if 0    
    forallInPlace([ ] PROTO_LAMBDA
      (Var<double,NUMCOMPS,HOST>& l_U,
        const Var<double,NUMCOMPS,HOST>& l_JU,
        const Var<double,1,HOST>& l_deta,
        const Var<double,1,HOST>& l_dVolr)
      {                    
        for (int comp = 0; comp < NUMCOMPS; comp++)
          {
            l_U(comp) = l_JU(comp)/(l_deta(0)*l_dVolr(0));
          } 
      },Uout[dit],JU_i,deta,dVolr_i);
#endif

#ifndef PR_DISABLE_INIT_CHECKS
    // bool infOrNanBlock = Wout_i.containsInfOrNAN();
    // if (infOrNanBlock)
    //   {
    //     h5.writePatch(1.0,Wout_i,"MHD_IO_InfOrNAN"+to_string(block));
    //     infOrNan = infOrNan || infOrNanBlock;
    //     cout << "Inf or Nan at Point " << Wout_i.findInfOrNAN()
    //          << ", block = " << block << endl;
    //   }
#endif
#if 0
    if (procID() == 0)
    {
      Box bx = layout[dit];
      std::string output1D_file = ParseInputs::get_data_file_prefix()+"index" + "1D_" + std::to_string(iter)+".curve";
      std::ofstream ofs (output1D_file, std::ofstream::out);
      Point ptcen = bx.high()/2;
      int highxi0 = bx.high()[0];
      ptcen[0] = 0;
      
      for (int comp = 0; comp < NUMCOMPS+3; comp++)
        {
          ofs << "#" + names[comp] << endl;
          for (int ind = 0; ind < highxi0; ind++)
            {
              Point ptindex = ptcen;
              ptindex[0] = ind;
              Point ptindex_p = ptcen;
              ptindex_p[0] = ind+1;
              double r = (radiusLev[dit](ptindex,0) + radiusLev[dit](ptindex_p,0))*.5;
              if (comp < NUMCOMPS+2)
                {
                  double val = Wout_i(ptindex,comp);
                  ofs << r << " " << val << endl;
                }
              else if (comp == NUMCOMPS+2)
                {
                  double p = Wout_i(ptindex,4);
                  double rho = Wout_i(ptindex,0);
                  double val = p/pow(rho,gamma);
                  ofs << r << " " << val << endl;
                }
            }
          ofs << endl;
        }
    }
#endif
  }

  // Recover density and pressure through the same selected-order primitive
  // path used by the spatial operator. O2 also uses these primitives for
  // vectors. O4 retains its semispherical presentation above, but nonlinear
  // pressure recovery from those cell averages is inaccurate in low-beta MHD.
  MBLevelBoxData<double, NUMCOMPS, HOST> USphEvolution(layout, OP::ghost());
  a_JU.copyTo(USphEvolution);
  OP::prepareState(USphEvolution,a_iop,dVolrLev,time);

  for (auto dit : layout)
  {
    unsigned int block = layout.block(dit);
    BoxData<double,NUMCOMPS,HOST> W;
    BoxData<double,NUMCOMPS,HOST> WBar;
    eulerOp[dit].sphToPrim(W,WBar,USphEvolution[dit],block);
    if (secondOrder)
    {
      auto basis = HelioCubed::secondOrderCellBasis<double,HOST>(
          layout[dit],dxiPerp,block);
      BoxData<double,NUMCOMPS,HOST> cart;
      HelioCubed::nativePrimitivesToCartesian(cart,W,basis);
      forallInPlace([] PROTO_LAMBDA(
          Var<double,NUMCOMPS,HOST>& out,
          const Var<double,NUMCOMPS,HOST>& cart,
          const Var<double,DIM,HOST,DIM>& basis,
          const Var<double,1,HOST>& radius,
          int scaled)
        {
          // The first basis column is the unit radial vector at the same
          // cell center. Project the full Cartesian vectors onto the usual
          // orthonormal (r, theta, phi) directions used by output files.
          const double nx = basis(0,0), ny = basis(1,0), nz = basis(2,0);
          const double phi = atan2(ny,nx);
          const double cp = cos(phi), sp = sin(phi);
          const double st = sqrt(nx*nx + ny*ny);
          for (int start : {iVX,iBX})
          {
            const double x = cart(start), y = cart(start+1), z = cart(start+2);
            out(start) = nx*x + ny*y + nz*z;
            out(start+1) = nz*(cp*x + sp*y) - st*z;
            out(start+2) = -sp*x + cp*y;
          }
          if (scaled == 1)
          {
            const double rOverAU = radius(0)/c_AU;
            for (int d = 0; d < DIM; ++d) out(iVX+d) *= 1.0e-5;
            out(iBX) *= rOverAU*rOverAU*1.0e6;
            out(iBY) *= rOverAU*1.0e6;
            out(iBZ) *= rOverAU*1.0e6;
          }
        },W_sph_out[dit],cart,basis,radiusLev[dit],scale_data_with_r);
    }
    forallInPlace([] PROTO_LAMBDA(
        Var<double,NUMCOMPS,HOST>& a_output,
        const Var<double,NUMCOMPS,HOST>& a_W,
        const Var<double,1,HOST>& a_radius,
        double a_gamma,
        int a_scale_data_with_r)
      {
        a_output(iRHO) = a_W(iRHO);
        a_output(iP) = a_W(iP);
        if (a_scale_data_with_r == 1)
        {
          const double rOverAU = a_radius(0)/c_AU;
          a_output(iRHO) *= rOverAU*rOverAU/c_MP;
          a_output(iP) *= pow(rOverAU,2.0*a_gamma)*1.0e12;
        }
      },W_sph_out[dit],W,radiusLev[dit],gamma,scale_data_with_r);
  }

  const auto outputNames = HelioCubed::outputPrimitiveNames();
  if (ParseInputs::get_output_coordinates() == 1)
  {
    // The established writer has already formed and scaled spherical fields.
    // Rotate those presentation values without modifying the evolved state.
    for (auto dit : layout)
    {
      const auto block = layout.block(dit);
      const auto dx = eulerOp[dit].dx();
      auto position = forall_p<double,DIM,HOST>(f_cubedSphereMap3,
          layout[dit],radiusLev[dit],dx,0.5,0.5,block);
      forallInPlace([] PROTO_LAMBDA(
          Var<double,NUMCOMPS,HOST>& out,
          const Var<double,DIM,HOST>& position)
        {
          const double radius = sqrt(position(0)*position(0)
              + position(1)*position(1) + position(2)*position(2));
          HelioCubed::sphericalOutputToCartesian(out,
              position(0)/radius,position(1)/radius,position(2)/radius);
        },W_sph_out[dit],position);
    }
  }

  if (write_data && ParseInputs::get_output_divB() == 1){
    MBLevelBoxData<double, 1, HOST> divB(layout, Point::Zeros());
    for (auto dit : layout)
    {
      eulerOp[dit].computeDivB(divB[dit],USphEvolution[dit]);
    }

    // Keep div(B) in physical cgs units (G/cm); scale_data_with_r only
    // changes the presentation units of the primitive MHD fields.
    MBLevelBoxData<double, NUMCOMPS+1, HOST> W_sph_divB_out(
        layout,Point::Zeros());
    for (auto dit : layout)
    {
      forallInPlace([ ] PROTO_LAMBDA
          (Var<double,NUMCOMPS+1,HOST>& a_output,
           const Var<double,NUMCOMPS,HOST>& a_W,
           const Var<double,1,HOST>& a_divB)
          {
            for (int comp = 0; comp < NUMCOMPS; comp++)
            {
              a_output(comp) = a_W(comp);
            }
            a_output(NUMCOMPS) = a_divB(0);
          },W_sph_divB_out[dit],W_sph_out[dit],divB[dit]);
    }

    std::string output_file = ParseInputs::get_data_file_prefix() + "." + std::to_string(iter);
    auto names = outputNames;
    names.push_back("divB");
    h5.writeMBLevel(names,map,W_sph_divB_out,output_file);
    pout(0) << "Data file with divB written at iteration " << iter << endl;
  }
  else if (write_data){
    std::string output_file = ParseInputs::get_data_file_prefix() + "." + std::to_string(iter); 
    h5.writeMBLevel(outputNames, map, W_sph_out, output_file);
    pout(0) << "Data file written at iteration " << iter << endl;
    // Print this for more detailed output
    // h5.writeMBLevel({"density", "VxPerp", "VyPerp", "VzPerp", "P", "BxPerp", "ByPerp", "BzPerp","Vr","Br"}, map, Wout, "W_" + to_string(iter));
  }

  if (write_slice && (ParseInputs::get_slice_X() || ParseInputs::get_slice_Y()
      || ParseInputs::get_slice_Z() || ParseInputs::get_slice_R())){
    double physical_time = getPhysTime(time);


    Point ghost  = Point(0, 3, 3);      // base ghost
    Point offset = Point(0, 2, 2);      // extra padding for interpolation
    MBLevelBoxData<double, NUMCOMPS, HOST> W_sph_in(layout, ghost + offset);
    MBLevelBoxData<double, NUMCOMPS, HOST> W_sph_out_ghost(layout, ghost);

    // Slice sampling needs three angular layers even when the evolution
    // operator uses only two. Give it matching interpolation and geometry.
    MBInterpOp* sliceInterp = &a_iop;
    const typename OP::MapT* sliceMap = &map;
    if (ParseInputs::get_spatial_order() == 2)
    {
      struct SliceInterpolation
      {
        MBDisjointBoxLayout layout;
        MBInterpOp op;
        typename OP::MapT map;
        SliceInterpolation(const MBDisjointBoxLayout& a_layout, Point a_ghost)
          : layout(a_layout), op(CubedSphereShell::MHDInterpOp<HOST>(a_layout,a_ghost,4)),
            map(CubedSphereShell::Map(a_layout,Point::Ones(3))) {}
      };
      static std::unique_ptr<SliceInterpolation> interpolation;
      if (!interpolation || &interpolation->layout.partition() != &layout.partition())
        interpolation = std::make_unique<SliceInterpolation>(layout,ghost+offset);
      sliceInterp = &interpolation->op;
      sliceMap = &interpolation->map;
    }

    W_sph_in.setVal(0.0);
    W_sph_out_ghost.setVal(0.0);

    W_sph_out.copyTo(W_sph_in);
    W_sph_out.copyTo(W_sph_out_ghost);

    W_sph_in.exchange();
    W_sph_in.copyTo(W_sph_out_ghost);
    sliceInterp->apply(W_sph_out_ghost, W_sph_in);
    W_sph_out_ghost.exchange0();


    if (ParseInputs::get_slice_X()){
      std::string tecplot_slice_file_x = ParseInputs::get_data_file_prefix() + ".x." + std::to_string(iter) + ".dat";
      writeTecplotSlice_plane(*sliceMap, W_sph_out_ghost, outputNames, tecplot_slice_file_x, physical_time,time,iter, 0, 0.0);
    }
    if (ParseInputs::get_slice_Y()){
      std::string tecplot_slice_file_y = ParseInputs::get_data_file_prefix() + ".y." + std::to_string(iter) + ".dat";
      writeTecplotSlice_plane(*sliceMap, W_sph_out_ghost, outputNames, tecplot_slice_file_y, physical_time,time,iter, 1, 0.0);
    }
    if (ParseInputs::get_slice_Z()){
      std::string tecplot_slice_file_z = ParseInputs::get_data_file_prefix() + ".z." + std::to_string(iter) + ".dat";
      writeTecplotSlice_plane(*sliceMap, W_sph_out_ghost, outputNames, tecplot_slice_file_z, physical_time,time,iter, 2, 0.0);
    }
    if (ParseInputs::get_slice_R()){
      std::string tecplot_spherical_slice_file = ParseInputs::get_data_file_prefix() + ".r." + std::to_string(iter) + ".dat";
      writeTecplotSlice_spherical(*sliceMap, W_sph_out_ghost, outputNames, tecplot_spherical_slice_file, physical_time, time, iter, /*radius=*/1.5e13);
    }
    pout(0) << "Tecplot slice files written at iteration " << iter << endl;                                     
  }
  
}

// Store the number of chronologically inserted events. Rank zero adds the scalar
// attribute after all ranks have closed the parallel data writer.
inline void CheckCMECheckpointHDF5(bool a_success, const char* a_operation)
{
  if (a_success) return;
  std::fprintf(stderr, "CME checkpoint: HDF5 operation failed: %s\n", a_operation);
  std::fflush(stderr);
#ifdef PR_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  std::abort();
}
#define CME_CHECKPOINT_HDF5(operation) \
    CheckCMECheckpointHDF5((operation), #operation)

inline void WriteCheckpointCMEState(std::string a_filename, int a_inserted)
{
  barrier();
  if (procID() == 0)
  {
    auto file = H5Fopen(a_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    CME_CHECKPOINT_HDF5(file >= 0);
    auto level = H5Gopen(file, "/level_0", H5P_DEFAULT);
    CME_CHECKPOINT_HDF5(level >= 0);
    auto space = H5Screate(H5S_SCALAR);
    CME_CHECKPOINT_HDF5(space >= 0);
    auto attr = H5Acreate2(level, "cme_inserted_count", H5T_NATIVE_INT, space,
                          H5P_DEFAULT, H5P_DEFAULT);
    CME_CHECKPOINT_HDF5(attr >= 0);
    int inserted = a_inserted;
    CME_CHECKPOINT_HDF5(H5Awrite(attr, H5T_NATIVE_INT, &inserted) >= 0);
    CME_CHECKPOINT_HDF5(H5Aclose(attr) >= 0);
    CME_CHECKPOINT_HDF5(H5Sclose(space) >= 0);
    CME_CHECKPOINT_HDF5(H5Gclose(level) >= 0);
    CME_CHECKPOINT_HDF5(H5Fclose(file) >= 0);
  }
  barrier();
}

// -1 identifies older checkpoints without event metadata. The driver then
// infers the event state from the restart time, as the FullSphere driver does.
inline int ReadCheckpointCMEState(std::string a_filename)
{
  if (a_filename.substr(a_filename.find_last_of('.') + 1) != "hdf5")
    a_filename += ".hdf5";
  int inserted = -1;
  barrier();
  if (procID() == 0)
  {
    auto file = H5Fopen(a_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CME_CHECKPOINT_HDF5(file >= 0);
    auto level = H5Gopen(file, "/level_0", H5P_DEFAULT);
    CME_CHECKPOINT_HDF5(level >= 0);
    const char* attribute = "cme_inserted_count";
    auto exists = H5Aexists(level, attribute);
    CME_CHECKPOINT_HDF5(exists >= 0);
    if (exists == 0) {
      attribute = "cme_inserted";
      exists = H5Aexists(level, attribute);
      if (exists > 0 && ParseInputs::get_CMEs().size() > 1)
        CheckCMECheckpointHDF5(false,"A legacy single-CME checkpoint cannot restore an indexed CME schedule");
    }
    CME_CHECKPOINT_HDF5(exists >= 0);
    if (exists > 0)
    {
      auto attr = H5Aopen(level, attribute, H5P_DEFAULT);
      CME_CHECKPOINT_HDF5(attr >= 0);
      CME_CHECKPOINT_HDF5(H5Aread(attr, H5T_NATIVE_INT, &inserted) >= 0);
      CME_CHECKPOINT_HDF5(inserted >= 0);
      CME_CHECKPOINT_HDF5(H5Aclose(attr) >= 0);
    }
    CME_CHECKPOINT_HDF5(H5Gclose(level) >= 0);
    CME_CHECKPOINT_HDF5(H5Fclose(file) >= 0);
  }
#ifdef PR_MPI
  MPI_Bcast(&inserted, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
  return inserted;
}

// Proto's multiblock reader requires the file and destination ghost widths
// to agree. Preserve checkpoint compatibility when the selected order or
// viscosity stencil changes the evolution halo.
inline void ReadCheckpointState(HDF5Handler& a_h5,
    MBLevelBoxData<double,NUMCOMPS,HOST>& a_state, std::string a_filename)
{
  if (a_filename.substr(a_filename.find_last_of('.') + 1) != "hdf5")
    a_filename += ".hdf5";
  Point storedGhost = Point::Zeros();
  int widths[DIM] = {};
  if (procID() == 0)
  {
    auto file = H5Fopen(a_filename.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
    CME_CHECKPOINT_HDF5(file >= 0);
    auto meta = H5Gopen(file,"/level_0/data_attributes",H5P_DEFAULT);
    CME_CHECKPOINT_HDF5(meta >= 0);
    auto attribute = H5Aopen(meta,"outputGhost",H5P_DEFAULT);
    CME_CHECKPOINT_HDF5(attribute >= 0);
    hid_t pointType;
    HDF5Handler::getH5DataType<Point>(&pointType);
    CME_CHECKPOINT_HDF5(H5Aread(attribute,pointType,&storedGhost) >= 0);
    CME_CHECKPOINT_HDF5(H5Aclose(attribute) >= 0);
    CME_CHECKPOINT_HDF5(H5Gclose(meta) >= 0);
    CME_CHECKPOINT_HDF5(H5Fclose(file) >= 0);
    for (int d=0;d<DIM;++d) widths[d]=storedGhost[d];
  }
#ifdef PR_MPI
  MPI_Bcast(widths,DIM,MPI_INT,0,MPI_COMM_WORLD);
#endif
  for (int d=0;d<DIM;++d)
  {
    CME_CHECKPOINT_HDF5(widths[d] >= 0);
    storedGhost[d]=widths[d];
  }
  if (storedGhost == a_state.ghost()[0])
  {
    a_h5.readMBLevel(a_state,a_filename);
    return;
  }
  Array<Point,DIM+1> ghost;
  ghost.fill(Point::Zeros());
  ghost[0]=storedGhost;
  MBLevelBoxData<double,NUMCOMPS,HOST> saved(a_state.layout(),ghost);
  a_h5.readMBLevel(saved,a_filename);
  for (auto index:a_state.layout())
  {
    a_state[index].setVal(0.);
    saved[index].copyTo(a_state[index],a_state.layout()[index]);
  }
}

#undef CME_CHECKPOINT_HDF5

  void Write_Checkpoint(MBLevelBoxData<double, NUMCOMPS, HOST> &a_JU,
             int iter,
             int restart_step,
             double time = 0.0,
             double dt = 0.0,
             int cme_inserted = 0)
{
  PR_TIME("Write_Checkpoint");
  HDF5Handler h5;
  std::string check_file_name = ParseInputs::get_checkpoint_file_prefix() + "_" + std::to_string(iter);
  h5.setTime(time);
  h5.setTimestep(dt);
  h5.setIter(iter);
  h5.writeMBLevel(a_JU, check_file_name);
  WriteCheckpointCMEState(check_file_name + ".hdf5", cme_inserted);
  pout(0) << "Checkpoint file written at iteration " << iter << endl;
  int checkpoint_cadence = ParseInputs::get_checkpoint_cadence();
  int iter_to_delete = iter - (checkpoint_cadence*ParseInputs::get_max_checkpoint_files());
  if (iter_to_delete > restart_step)
  {
    std::string filename_to_delete=ParseInputs::get_checkpoint_file_prefix()+"_"+std::to_string(iter_to_delete)+".hdf5";
    const char* str = filename_to_delete.c_str();
    if (procID() == 0) std::remove(str);
  }
}

// Returns interpolated value at x from parallel arrays ( xData, yData )
//   Assumes that xData has at least two elements, is sorted and is strictly monotonic increasing
//   boolean argument extrapolate determines behaviour beyond ends of array (if needed)
double interpolate(const vector<double> &xData, const vector<double> &yData, double x, bool extrapolate)
{
  int size = xData.size();

  int i = 0;                // find left end of interval for interpolation
  if (x >= xData[size - 2]) // special case: beyond right end
  {
    i = size - 2;
  }
  else
  {
    auto upper = std::lower_bound(xData.begin() + 1, xData.end(), x);
    i = static_cast<int>(upper - xData.begin()) - 1;
  }
  double xL = xData[i], yL = yData[i], xR = xData[i + 1], yR = yData[i + 1]; // points on either side (unless beyond ends)
  if (!extrapolate)                                                          // if beyond ends of array and not extrapolating
  {
    if (x < xL)
      yR = yL;
    if (x > xR)
      yL = yR;
  }

  double dydx = (yR - yL) / (xR - xL); // gradient

  return yL + dydx * (x - xL); // linear interpolation
}

namespace HelioCubed
{
struct ProbeTrajectory
{
  vector<double> time, radius, latitude, longitude;

  explicit ProbeTrajectory(const string& filename)
  {
    std::ifstream file(filename);
    if (!file.is_open()) MayDay<void>::Abort(("Unable to open probe trajectory: " + filename).c_str());
    string line;
    while (std::getline(file, line))
    {
      if (line.find_first_not_of(" \t\r\n") == string::npos) continue;
      double yearValue, dayValue, r, lat, lon;
      std::istringstream row(line);
      if (!(row >> yearValue >> dayValue >> r >> lat >> lon))
        MayDay<void>::Abort(("Invalid probe trajectory row in: " + filename).c_str());
      // Retain the existing whole-day interpretation of trajectory DOY values.
      const int year = static_cast<int>(yearValue);
      const int day = static_cast<int>(dayValue);
      const double epoch = year + (day - 1)/(gregorianLeapYear(year) ? 366.0 : 365.0);
      if (!time.empty() && epoch <= time.back())
        MayDay<void>::Abort(("Probe trajectory times must increase: " + filename).c_str());
      time.push_back(epoch);
      radius.push_back(r);
      latitude.push_back(lat);
      longitude.push_back(lon);
    }
    if (time.size() < 2)
      MayDay<void>::Abort(("Probe trajectory needs at least two rows: " + filename).c_str());
  }
};

inline const ProbeTrajectory& probeTrajectory(const string& filename)
{
  // Trajectories are run inputs; load each distinct file once on each rank.
  static std::map<string, ProbeTrajectory> cache;
  auto entry = cache.find(filename);
  if (entry == cache.end())
    entry = cache.emplace(filename, ProbeTrajectory(filename)).first;
  return entry->second;
}

struct ProbeGeometry
{
  MBDisjointBoxLayout layout;
  std::map<std::pair<int,int>, double> angularDeterminants;

  explicit ProbeGeometry(const MBDisjointBoxLayout& a_layout) : layout(a_layout) {}

  double determinant(Point point, double spacing)
  {
    auto key = std::make_pair(point[1], point[2]);
    auto entry = angularDeterminants.find(key);
    if (entry == angularDeterminants.end())
    {
      // detA is independent of radius and block. Evaluate only sampled cells.
      Point angular(0, point[1], point[2]);
      auto value = CubedSphereShell::detA<double,HOST>(Box(angular, angular), spacing, 0);
      entry = angularDeterminants.emplace(key, value(angular, 0)).first;
    }
    return entry->second;
  }
};

inline ProbeGeometry& probeGeometry(const MBDisjointBoxLayout& layout)
{
  static std::unique_ptr<ProbeGeometry> cache;
  if (!cache || &cache->layout.partition() != &layout.partition())
    cache = std::make_unique<ProbeGeometry>(layout);
  return *cache;
}
}


template<typename BOX_OP>
void Probe(const MBLevelBoxData<double, NUMCOMPS, HOST> &a_JU,
          const MBLevelMap<MBMap_CubedSphereShell<HOST>, HOST>& a_map,
          BOX_OP &a_eulerOp,
          MBInterpOp &a_iop,
          int a_iter,
          double a_time,
          Array<double, DIM> a_dx,
          bool give_space = false,
          string probe_file = "default",
          string Probe_data_file = "default",
          const MBLevelBoxData<double, 1, HOST>* a_dVolrLev = nullptr)

{
  PR_TIME("Probe");
  int pid = procID();
  ofstream outputFile;
  auto openOutput = [&]()
  {
    if (!outputFile.is_open()) outputFile.open(Probe_data_file, std::ios::app);
  };
  if (pid == 0 && give_space)
  {
    openOutput();
    outputFile << endl;
  }
  double gamma = ParseInputs::get_gamma();
  const auto& trajectory = HelioCubed::probeTrajectory(probe_file);
  double physical_time = getPhysTime(a_time);
  double r_now = interpolate(trajectory.time, trajectory.radius, physical_time, false);
  double latitude_now = interpolate(trajectory.time, trajectory.latitude, physical_time, false);
  double longitude_now = interpolate(trajectory.time, trajectory.longitude, physical_time, false);
  r_now *= c_AU;
  latitude_now = (90.0 - latitude_now);   // To co-latitude
  longitude_now = longitude_now + 181.02; // HGI to Carrington
  if (longitude_now >= 360.0)
    longitude_now -= 360;
  // longitude_now = longitude_now - 1.792951949248379e+02; // HGI to Carrington
  if (longitude_now < 0.0)
    longitude_now += 360;

  double x_n = sin(latitude_now * M_PI / 180) * cos(longitude_now * M_PI / 180);
  double y_n = sin(latitude_now * M_PI / 180) * sin(longitude_now * M_PI / 180);
  double z_n = cos(latitude_now * M_PI / 180);

  double x = r_now * x_n;
  double y = r_now * y_n;
  double z = r_now * z_n;

  const double theta = latitude_now * M_PI / 180;
  const double phi = longitude_now * M_PI / 180;

  const Array<double, DIM> Xcart = {x, y, z};
  Array<double, DIM> Xi;
  const auto& layout = a_map.layout();
  auto& geometry = HelioCubed::probeGeometry(layout);
  // Select the physical panel independently of the rank's local patches.
  // The first covering panel owns seam points, so each probe writes one row.
  int sampleBlock = -1;
  for (int block = 0; block < 6; ++block)
  {
    f_cubedSphereInverseMap(Xi, Xcart, block, cubedSphereShellGeometry());
    bool inside = true;
    for (int d = 0; d < DIM; ++d)
      if (!std::isfinite(Xi[d]) || Xi[d] < -1e-12 || Xi[d] > 1 + 1e-12)
        inside = false;
    if (!inside) continue;
    for (int d = 0; d < DIM; ++d) Xi[d] = std::clamp(Xi[d], 0.0, 1.0);
    sampleBlock = block;
    break;
  }
  if (sampleBlock < 0) return;
  const Point size = layout.domain().getBlock(sampleBlock).sizes();

  double eta0 = Xi[0];
  double eta1 = Xi[1];
  double eta2 = Xi[2];

  double pt0 = eta0 / a_dx[0];
  double pt1 = eta1 / a_dx[1];
  double pt2 = eta2 / a_dx[2];

  int pt0_nearest, pt1_nearest, pt2_nearest;
  int pt0_neigbor, pt1_neigbor, pt2_neigbor;

  pt0_nearest = std::min(int(floor(pt0)), size[0] - 1);
  pt1_nearest = std::min(int(floor(pt1)), size[1] - 1);
  pt2_nearest = std::min(int(floor(pt2)), size[2] - 1);

  pt0_neigbor = pt0_nearest + (pt0 < pt0_nearest + 0.5 ? -1 : 1);
  pt1_neigbor = pt1_nearest + (pt1 < pt1_nearest + 0.5 ? -1 : 1);
  pt2_neigbor = pt2_nearest + (pt2 < pt2_nearest + 0.5 ? -1 : 1);

  double probed_values[NUMCOMPS];
  double probed_values_primitive[NUMCOMPS];
  Point index_cc(pt0_nearest, pt1_nearest, pt2_nearest);
  Point index_n0(pt0_neigbor, pt1_nearest, pt2_nearest);
  Point index_n1(pt0_nearest, pt1_neigbor, pt2_nearest);
  Point index_n2(pt0_nearest, pt1_nearest, pt2_neigbor);


  double dxiPerp = 1.0/size[1];

  for (auto dit : layout)
  {
    auto& JU_i = a_JU[dit];
    const Box dbx0 = JU_i.box();
    const Box dbx1 = layout[dit];
    if (layout.block(dit) == sampleBlock && dbx1.containsPoint(index_cc))
    {
      BoxData<double,1,HOST> localVolume;
      if (!a_dVolrLev)
      {
        // Keep the original call interface usable without allocating a level.
        Box sampleBox(index_cc-Point::Ones(), index_cc+Point::Ones());
        localVolume.define(sampleBox);
        BoxData<double,1,HOST> radius(sampleBox);
        BoxData<double,DIM,HOST> Dr(sampleBox), adjDr(sampleBox);
        a_eulerOp[dit].radialMetrics(radius, Dr, adjDr, localVolume, sampleBox);
      }
      const auto& volume = a_dVolrLev ? (*a_dVolrLev)[dit] : localVolume;
      const std::array<Point,4> samples = {index_cc, index_n0, index_n1, index_n2};
      std::array<std::array<double,NUMCOMPS>,4> values{};
      for (int sample = 0; sample < 4; ++sample)
      {
        Point point = samples[sample];
        if (!dbx0.containsPoint(point)) continue;
        const double J = geometry.determinant(point, dxiPerp)*volume(point, 0);
        for (int component = 0; component < NUMCOMPS; ++component)
          values[sample][component] = JU_i(point, component)/J;
      }
      for (int i = 0; i < NUMCOMPS; i++)
      {
        probed_values[i] = values[0][i];
        if (dbx0.containsPoint(index_n0))
          probed_values[i] += (values[1][i] - values[0][i]) * HelioCubed::probeInterpolationWeight(pt0, pt0_nearest, pt0_neigbor);
        if (dbx0.containsPoint(index_n1))
          probed_values[i] += (values[2][i] - values[0][i]) * HelioCubed::probeInterpolationWeight(pt1, pt1_nearest, pt1_neigbor);
        if (dbx0.containsPoint(index_n2))
          probed_values[i] += (values[3][i] - values[0][i]) * HelioCubed::probeInterpolationWeight(pt2, pt2_nearest, pt2_neigbor);
      }

      double v2 = 0.0;
      double B2 = 0.0;
      probed_values_primitive[0] = probed_values[0];

      for (int i = 1; i <= DIM; i++)
      {
        double v, B;
        v = probed_values[i] / probed_values[0];
        B = probed_values[DIM + 1 + i];
        probed_values_primitive[i] = v;
        probed_values_primitive[DIM + 1 + i] = B;
        v2 += v * v;
        B2 += B * B;
      }

      probed_values_primitive[DIM + 1] = (probed_values[DIM + 1] - .5 * probed_values[0] * v2 - B2 / 8.0 / M_PI) * (gamma - 1.0);

      double rho = probed_values_primitive[0] / c_MP; // /cm^3;
      double Vx = probed_values_primitive[1] / 1e5;   // km/s;
      double Vy = probed_values_primitive[2] / 1e5;   // km/s;
      double Vz = probed_values_primitive[3] / 1e5;   // km/s;
      double p = probed_values_primitive[4] * 1e12;   // picodyne/cm*2
      double Bx = probed_values_primitive[5] * 1e6;   // microGauss
      double By = probed_values_primitive[6] * 1e6;   // microGauss
      double Bz = probed_values_primitive[7] * 1e6;   // microGauss

      const auto velocity = HelioCubed::projectProbeVectorToRTN(Vx, Vy, Vz, theta, phi);
      const auto magnetic = HelioCubed::projectProbeVectorToRTN(Bx, By, Bz, theta, phi);

      // if (probed_values[0] != 0) {
      openOutput();
      outputFile << setw(16) << setprecision(12) << physical_time
                 << setw(11) << setprecision(4) << r_now / c_AU
                 << setw(11) << setprecision(4) << 90 - latitude_now
                 << setw(11) << setprecision(4) << longitude_now
                 << setw(11) << setprecision(4) << rho
                 << setw(11) << setprecision(4) << velocity[0]
                 << setw(11) << setprecision(4) << velocity[1]
                 << setw(11) << setprecision(4) << velocity[2]
                 << setw(11) << setprecision(4) << p
                 << setw(11) << setprecision(4) << magnetic[0]
                 << setw(11) << setprecision(4) << magnetic[1]
                 << setw(11) << setprecision(4) << magnetic[2]
                 << endl;
      // }
    }
  }

  outputFile.close();
}
