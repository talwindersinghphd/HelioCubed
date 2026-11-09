#include "MHD_Initialize.H"
#include "MHD_Mapping.H"
#include "MHD_Set_Boundary_Values.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

Parsefrominputs inputs;

void close(double actual, double expected, const char* description)
{
    MHD::require(std::isfinite(actual) &&
            std::abs(actual-expected) <= 1e-11*std::max(1.0,std::abs(expected)),
            "%s: got %.17g, expected %.17g", description, actual, expected);
}

// Compare conserved states against the full Galilean momentum/energy transform.
// Density, magnetic fields, and turbulence quantities must be unchanged.
void check_conversion(const BoxData<double,NUMCOMPS>& inertial,
                      const BoxData<double,NUMCOMPS>& rotating,
                      const BoxData<double,DIM>& coords,
                      const Point& p, bool cartesian)
{
    const double speed = inputs.solar_angular_velocity()*coords(p,0)*
            std::sin(coords(p,1))/inputs.velocity_scale;
    const std::array<double,DIM> boost = cartesian ?
            std::array<double,DIM>{speed*std::sin(coords(p,2)),
                    -speed*std::cos(coords(p,2)),0.0} :
            std::array<double,DIM>{0.0,0.0,-speed};
    const double rho = inertial(p,iRHO);
    double energy = inertial(p,iE);
    double kinetic_i = 0.0, kinetic_c = 0.0;
    for (int d = 0; d < DIM; ++d)
    {
        const double momentum = inertial(p,iMOMX+d);
        close(rotating(p,iMOMX+d), momentum+rho*boost[d], "converted momentum");
        energy += momentum*boost[d]+0.5*rho*boost[d]*boost[d];
        kinetic_i += 0.5*momentum*momentum/rho;
        kinetic_c += 0.5*rotating(p,iMOMX+d)*rotating(p,iMOMX+d)/rho;
    }
    close(rotating(p,iE), energy, "converted total energy");
    close(rotating(p,iE)-kinetic_c, inertial(p,iE)-kinetic_i,
            "thermal plus magnetic/turbulent energy is preserved");
    for (int c = 0; c < NUMCOMPS; ++c)
        if (c < iMOMX || c > iE)
            MHD::require(rotating(p,c) == inertial(p,c),
                    "Frame conversion changed component %i",c);
}

void check_analytic_initial_state()
{
    const Box box(Point::Zeros(),Point(0,0,0));
    const Point p = Point::Zeros();
    BoxData<double,DIM> coords(box);
    BoxData<double,NUMCOMPS> inertial(box),rotating(box);
    for (double radius : {0.08*c_AU,0.1*c_AU,1.2*c_AU})
    for (double theta : {0.0,0.5,c_PI/2.0,c_PI})
    for (int spherical : {0,1})
    {
        inputs.initialize_in_spherical_coords = spherical;
        coords(p,0)=radius;
        coords(p,1)=theta;
        coords(p,2)=1.37;
        inputs.corotating_frame=0;
        MHD_Initialize::InitializeState_Spherical_2O(inertial,coords,inputs.gamma);
        inputs.corotating_frame=1;
        MHD_Initialize::InitializeState_Spherical_2O(rotating,coords,inputs.gamma);
        check_conversion(inertial,rotating,coords,p,false);
    }
    inputs.initialize_in_spherical_coords=1;
}

void check_inner_boundary(bool file_boundary,double azimuthal_velocity)
{
    const Box domain(Point::Zeros(),Point(7,7,15));
    const std::array<bool,DIM> periodic={true,true,true};
    MHDLevelDataState state(ProblemDomain(domain,periodic),Point::Ones(4),
            1.0/8.0,1.0/8.0,1.0/16.0,inputs.gamma);
    state.m_time=0.0;
    inputs.sph_inner_BC_hdf5=file_boundary;
    const std::array<double,8> map={700.0,5e7,2.3e6,azimuthal_velocity,
            7.8e4,500.0,10.0,-5.0};
    state.m_BC.setToZero();
    for (auto dit : state.m_BC)
    for (auto p : state.m_BC[dit].box())
    for (int c=0;c<8;++c) state.m_BC[dit](p,c)=map[c];

    LevelBoxData<double,NUMCOMPS> inertial(state.m_dbl,Point::Ones(NGHOST));
    LevelBoxData<double,NUMCOMPS> rotating(state.m_dbl,Point::Ones(NGHOST));
    LevelBoxData<double,NUMCOMPS> repeated(state.m_dbl,Point::Ones(NGHOST));
    for (auto dit : inertial)
    {
        BoxData<double,DIM> coords(inertial[dit].box());
        MHD_Mapping::get_sph_coords_cc(coords,coords.box(),state.m_dx,state.m_dy,state.m_dz);
        BoxData<double,NUMCOMPS> spherical(coords.box());
        inputs.corotating_frame=0;
        MHD_Initialize::InitializeState_Spherical_2O(spherical,coords,inputs.gamma);
        MHD_Mapping::Spherical_to_Cartesian(inertial[dit],spherical,coords);
        inertial[dit].copyTo(rotating[dit]);
    }
    inputs.corotating_frame=0;
    MHD_Set_Boundary_Values::Set_Boundary_Values_Spherical_2O(inertial,state);
    inputs.corotating_frame=1;
    MHD_Set_Boundary_Values::Set_Boundary_Values_Spherical_2O(rotating,state);
    for (auto dit : rotating) rotating[dit].copyTo(repeated[dit]);
    MHD_Set_Boundary_Values::Set_Boundary_Values_Spherical_2O(repeated,state);
    for (auto dit : inertial)
    {
        BoxData<double,DIM> coords(inertial[dit].box());
        MHD_Mapping::get_sph_coords_cc(coords,coords.box(),state.m_dx,state.m_dy,state.m_dz);
        for (auto p : inertial[dit].box())
        {
            if (p[0]>=0) continue;
            check_conversion(inertial[dit],rotating[dit],coords,p,true);
            for (int c=0;c<NUMCOMPS;++c)
                MHD::require(rotating[dit](p,c) == repeated[dit](p,c),
                        "Repeated boundary filling applied a second frame conversion");
            for (int c=0;c<8;++c)
                MHD::require(state.m_BC[dit](p,c) == map[c],
                        "Boundary filling modified the supplied inertial map");
        }
    }
}

int main(int argc,char** argv)
{
#ifdef PR_MPI
    MPI_Init(&argc,&argv);
#endif
    inputs.density_scale=700.0;
    inputs.velocity_scale=3e7;
    inputs.gamma=1.5;
    inputs.r_in=0.1;
    inputs.r_out=0.3;
    inputs.C_rad=1.0;
    for (double period : {25.38,20.0})
    {
        inputs.solar_rotation_period=period;
        check_analytic_initial_state();
        check_inner_boundary(true,0.0);
        check_inner_boundary(true,2.5e6);
        check_inner_boundary(false,0.0);
    }
    if (procID()==0)
        std::cout << "Automatic boundary/initial frame conversion checks passed (TURB="
                  << TURB << ")." << std::endl;
#ifdef PR_MPI
    MPI_Finalize();
#endif
}
