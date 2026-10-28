#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "MHD_Constants.H"

namespace HelioCubed
{
struct JmapParameters
{
    std::string filename, observer, reference;
    double positionAngle=90; // Solar north through solar east, in degrees.
};

using SightVector=std::array<double,3>;
inline double sightDot(const SightVector& a,const SightVector& b)
{return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}

// Finite, limb-darkened solar disk: Xiong et al. (2013), equations 1--6,
// doi:10.1007/s11207-012-0047-0. Returns z^2 G and z^2 G_P in cm^2.
// Integrating ne * kernel * ds gives radiance / disk-center solar radiance.
// There is no observer-distance inverse-square factor in a radiance integral.
inline std::array<double,2> thomsonKernel(double radius,double cosChi,double limb=0.63)
{
    if(!(radius>c_SR) || !std::isfinite(radius))
        throw std::invalid_argument("Thomson scattering needs radius above the photosphere");
    const double s=c_SR/radius,q=s*s,c=std::sqrt(1-q);
    const double A=c*q,C=q/(1+c)*(1+(1+c+c*c)/3);
    double B,D;
    if(s<0.1)
    {
        // Avoid subtracting O(1) terms to recover O(s^2) coefficients at 1 AU.
        B=q*(2./3+q*(-4./15+q*(-2./35+q*(-8./315-q*10./693))));
        D=q*(2./3+q*q*(2./105+q*(4./315+q*2./231)));
    }
    else
    {
        const double f=(1-q)*std::atanh(s)/s;
        B=(-1+3*q+f*(1+3*q))/8;
        D=(5+q-f*(5-q))/8;
    }
    const double tangent=(1-limb)*C+limb*D;
    const double polarized=std::max(0.,1-cosChi*cosChi)*((1-limb)*A+limb*B);
    const double factor=c_PI*7.95e-26/2;
    return {factor*(2*tangent-polarized),factor*polarized};
}

inline SightVector sightDirection(const SightVector& observer,double elongation,double positionAngle,
    const SightVector* reference=nullptr)
{
    const double radius=std::sqrt(sightDot(observer,observer));
    const double longitude=std::atan2(observer[1],observer[0]);
    const double latitude=std::asin(observer[2]/radius);
    const SightVector north={-sin(latitude)*cos(longitude),-sin(latitude)*sin(longitude),cos(latitude)};
    const SightVector east={sin(longitude),-cos(longitude),0};
    const double e=elongation*c_PI/180,p=positionAngle*c_PI/180;
    SightVector tangent;
    for(int d=0;d<3;++d) tangent[d]=cos(p)*north[d]+sin(p)*east[d];
    if(reference)
    {
        const double projection=sightDot(observer,*reference)/(radius*radius);
        for(int d=0;d<3;++d) tangent[d]=(*reference)[d]-projection*observer[d];
        const double norm=sqrt(sightDot(tangent,tangent));
        if(norm<1e-10*sqrt(sightDot(*reference,*reference)))
            throw std::invalid_argument("J-map observer and reference are collinear with the Sun");
        for(double& value : tangent) value/=norm;
    }
    SightVector direction;
    for(int d=0;d<3;++d)
        direction[d]=-cos(e)*observer[d]/radius+sin(e)*tangent[d];
    return direction;
}

// Intersect a forward ray with the modeled shell. Exclude the inner cavity;
// if the opaque photosphere is crossed, discard everything behind it too.
inline std::vector<std::array<double,2>> sightSegments(const SightVector& observer,
    const SightVector& direction,double inner,double outer)
{
    const double along=sightDot(observer,direction);
    const double impact2=std::max(0.,sightDot(observer,observer)-along*along);
    if(impact2>=outer*outer) return {};
    const double root=sqrt(outer*outer-impact2);
    double low=std::max(0.,-along-root),high=-along+root;
    const double photosphere2=double(c_SR)*c_SR;
    if(impact2<photosphere2 && -along>0)
        high=std::min(high,-along-sqrt(photosphere2-impact2));
    if(high<=low) return {};
    if(impact2>=inner*inner) return {{low,high}};
    const double hole=sqrt(inner*inner-impact2),a=-along-hole,b=-along+hole;
    std::vector<std::array<double,2>> segments;
    if(low<std::min(high,a)) segments.push_back({low,std::min(high,a)});
    if(std::max(low,b)<high) segments.push_back({std::max(low,b),high});
    return segments;
}
}
