#pragma once

#include "Proto.H"
#include "MHDOp.H"

namespace MHD_Rotation
{
	/**
	 * Add centrifugal and Coriolis sources for rotation about +z.
	 * RHS and Cartesian primitive velocities are nondimensional; spherical
	 * coordinates contain radius in cm and angles in radians. The rotation rate
	 * is in rad/s, and the velocity scale is in cm/s.
	 */
	void add_rotation_source(
	        BoxData<double,NUMCOMPS>& a_Rhs,
	        const BoxData<double,NUMCOMPS>& a_W_cart,
	        const BoxData<double,DIM>& a_x_sph,
	        double a_velocity_scale,
	        double a_rotation_rate);
}
