#pragma once

#include "MHD_JSON.H"
#include "MHD_Time.H"
#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>

// The continuous dGCS representation, in its exported native coordinates.
// Neither surface triangles nor field-line samples define the interior field.
namespace ImportedCME {
constexpr double pi=3.1415926535897932384626433832795;
constexpr double nominalSolarRadius=69570000000.;
using Vec=std::array<double,3>;
using Mat=std::array<Vec,3>;
inline double dot(const Vec& a,const Vec& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline double norm(const Vec& a) { return std::sqrt(dot(a,a)); }
inline Vec product(const Mat& a,const Vec& b) { return {dot(a[0],b),dot(a[1],b),dot(a[2],b)}; }
inline Vec transposeProduct(const Mat& a,const Vec& b) {
    Vec v{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) v[i]+=a[j][i]*b[j]; return v;
}
inline Mat product(const Mat& a,const Mat& b) {
    Mat c{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) c[i][j]+=a[i][k]*b[k][j]; return c;
}
inline double determinant(const Mat& a) {
    return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-
           a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+
           a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
}
inline Vec solve(const Mat& a,const Vec& b) {
    const double d=determinant(a);
    if (!(d>0) || !std::isfinite(d)) throw std::runtime_error("Invalid imported CME Jacobian");
    Vec x{};
    for(int j=0;j<3;++j) { Mat m=a; for(int i=0;i<3;++i) m[i][j]=b[i]; x[j]=determinant(m)/d; }
    return x;
}
inline bool close(double a,double b,double tolerance=1e-10) {
    return std::abs(a-b)<=tolerance*std::max({1.,std::abs(a),std::abs(b)});
}
inline bool same(const CMEJson::Value& a,const CMEJson::Value& b) {
    using J=CMEJson::Value;
    if(a.kind!=b.kind) return false;
    if(a.kind==J::Number) return close(a.number,b.number,1e-12);
    if(a.kind==J::String) return a.string==b.string;
    if(a.kind==J::Boolean) return a.boolean==b.boolean;
    if(a.kind==J::Array) {
        if(a.array.size()!=b.array.size()) return false;
        for(size_t i=0;i<a.array.size();++i) if(!same(a.array[i],b.array[i])) return false;
    }
    if(a.kind==J::Object) {
        if(a.object.size()!=b.object.size()) return false;
        for(const auto& kv:a.object) if(!b.has(kv.first) || !same(kv.second,b.at(kv.first))) return false;
    }
    return true;
}
inline Vec vector(const CMEJson::Value& v) {
    const auto& a=v.items(); if(a.size()!=3) throw std::invalid_argument("CME vector needs three components");
    return {a[0].real(),a[1].real(),a[2].real()};
}
inline Mat matrix(const CMEJson::Value& v) {
    const auto& a=v.items(); if(a.size()!=3) throw std::invalid_argument("CME matrix needs three rows");
    return {vector(a[0]),vector(a[1]),vector(a[2])};
}
inline CalendarEpoch epoch(const std::string& text) {
    if(text.size()<20 || text.back()!='Z' || text[4]!='-' || text[7]!='-'
       || text.find_first_of(" \t\r\n#")!=std::string::npos)
        throw std::invalid_argument("Export epoch must use YYYY-MM-DDTHH:MM:SSZ (UTC; fractional seconds are allowed)");
    return parseCalendarEpoch(text,"export epoch");
}
inline Mat orientation(double latitude,double longitude,double tilt) {
    const double a=latitude*pi/180,b=longitude*pi/180,t=tilt*pi/180;
    const Mat rz={Vec{std::cos(t),-std::sin(t),0},Vec{std::sin(t),std::cos(t),0},Vec{0,0,1}};
    const Mat ry={Vec{std::cos(a),0,std::sin(a)},Vec{0,1,0},Vec{-std::sin(a),0,std::cos(a)}};
    const Mat rx={Vec{1,0,0},Vec{0,std::cos(b),-std::sin(b)},Vec{0,std::sin(b),std::cos(b)}};
    const Mat p={Vec{0,0,1},Vec{0,-1,0},Vec{1,0,0}};
    return product(p,product(rx,product(ry,rz)));
}
inline void bernstein(int n,double t,std::array<double,10>& values) {
    values.fill(0); values[0]=1;
    for(int degree=1;degree<=n;++degree) {
        for(int i=degree;i>=0;--i) values[i]=(i<=degree-1 ? (1-t)*values[i] : 0)+(i ? t*values[i-1] : 0);
    }
}
inline double spectralNorm(const Mat& m) {
    Mat a{};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) a[i][j]+=m[k][i]*m[k][j];
    for(int sweep=0;sweep<20;++sweep) for(int p=0;p<3;++p) for(int q=p+1;q<3;++q) {
        if(std::abs(a[p][q])<1e-16*std::max({1e-300,std::abs(a[p][p]),std::abs(a[q][q])})) continue;
        const double tau=(a[q][q]-a[p][p])/(2*a[p][q]);
        const double t=std::copysign(1./(std::abs(tau)+std::hypot(1.,tau)),tau),c=1/std::sqrt(1+t*t),s=t*c;
        a[p][p]-=t*a[p][q]; a[q][q]+=t*a[p][q]; a[p][q]=a[q][p]=0;
        for(int k=0;k<3;++k) if(k!=p && k!=q) {
            const double kp=a[k][p],kq=a[k][q];
            a[k][p]=a[p][k]=c*kp-s*kq; a[k][q]=a[q][k]=s*kp+c*kq;
        }
    }
    return std::sqrt(std::max({0.,a[0][0],a[1][1],a[2][2]}))*(1+1e-13);
}

struct Map {
    std::array<int,3> grid{2,2,2};
    std::vector<Vec> coefficients=std::vector<Vec>(8);
    double scale=1,bound=0;
    Vec lower{-1,-1,0},upper{1,1,1}; // Convex-hull bounds of the mapped box.
    size_t index(int i,int j,int k) const { return (i*grid[1]+j)*grid[2]+k; }
    Vec evaluate(const Vec& q,Mat* jacobian=nullptr) const {
        std::array<std::array<double,10>,3> b{},db{};
        for(int axis=0;axis<3;++axis) {
            const double extent=axis==2 ? 1. : 2.;
            const double t=std::clamp((q[axis]/scale+(axis==2?0:1))/extent,0.,1.);
            bernstein(grid[axis]-1,t,b[axis]);
            if(jacobian) {
                std::array<double,10> base{}; bernstein(grid[axis]-2,t,base);
                for(int i=0;i<grid[axis];++i) db[axis][i]=(grid[axis]-1)/extent*((i ? base[i-1] : 0)-(i<grid[axis]-1 ? base[i] : 0));
            }
        }
        Vec result=q; if(jacobian) *jacobian={Vec{1,0,0},Vec{0,1,0},Vec{0,0,1}};
        for(int i=0;i<grid[0];++i) for(int j=0;j<grid[1];++j) for(int k=0;k<grid[2];++k) {
            const auto& c=coefficients[index(i,j,k)];
            const double weight=b[0][i]*b[1][j]*b[2][k];
            for(int d=0;d<3;++d) result[d]+=scale*weight*c[d];
            if(jacobian) {
                const Vec weights={db[0][i]*b[1][j]*b[2][k],b[0][i]*db[1][j]*b[2][k],b[0][i]*b[1][j]*db[2][k]};
                for(int d=0;d<3;++d) for(int a=0;a<3;++a) (*jacobian)[d][a]+=weights[a]*c[d];
            }
        }
        return result;
    }
    bool inverse(const Vec& target,Vec& q) const {
        for(int d=0;d<3;++d) if(target[d]<lower[d]-1e-10*scale || target[d]>upper[d]+1e-10*scale) return false;
        auto clip=[&](Vec v) { for(int d=0;d<3;++d) v[d]=std::clamp(v[d],d==2?0.:-scale,scale); return v; };
        q=clip(target);
        for(int iteration=0;iteration<200;++iteration) {
            Mat a; const Vec mapped=evaluate(q,&a); Vec residual{};
            for(int d=0;d<3;++d) residual[d]=mapped[d]-target[d];
            if(norm(residual)<2e-11*scale) return true;
            const Vec step=solve(a,residual); Vec proposed=q;
            for(int d=0;d<3;++d) proposed[d]-=step[d];
            proposed=clip(proposed);
            Vec trial=evaluate(proposed); for(int d=0;d<3;++d) trial[d]-=target[d];
            if(norm(trial)<norm(residual)) q=proposed;
            else { for(int d=0;d<3;++d) q[d]-=residual[d]; q=clip(q); }
        }
        Vec r=evaluate(q); for(int d=0;d<3;++d) r[d]-=target[d];
        return norm(r)<=2e-10*scale;
    }
    double derivativeBound(double factor=1) const {
        Mat maxima{};
        for(int i=0;i<grid[0];++i) for(int j=0;j<grid[1];++j) for(int k=0;k<grid[2];++k) {
            std::array<int,3> at{i,j,k};
            for(int a=0;a<3;++a) if(at[a]+1<grid[a]) {
                auto next=at; ++next[a];
                for(int d=0;d<3;++d) maxima[d][a]=std::max(maxima[d][a],std::abs(coefficients[index(next[0],next[1],next[2])][d]-coefficients[index(i,j,k)][d])*(grid[a]-1)*factor/(a==2?1.:2.));
            }
        }
        return spectralNorm(maxima);
    }
    std::array<Map,2> split(int axis) const {
        std::array<Map,2> children{*this,*this};
        for(int i=0;i<grid[0];++i) for(int j=0;j<grid[1];++j) for(int k=0;k<grid[2];++k) {
            std::array<int,3> at{i,j,k}; if(at[axis]) continue;
            std::array<Vec,10> row{}; const int n=grid[axis];
            for(int l=0;l<n;++l) { at[axis]=l; row[l]=coefficients[index(at[0],at[1],at[2])]; }
            for(int level=0;level<n;++level) {
                at[axis]=level; children[0].coefficients[index(at[0],at[1],at[2])]=row[0];
                at[axis]=n-1-level; children[1].coefficients[index(at[0],at[1],at[2])]=row[n-level-1];
                for(int l=0;l<n-level-1;++l) for(int d=0;d<3;++d) row[l][d]=.5*(row[l][d]+row[l+1][d]);
            }
        }
        return children;
    }
    void prepare(const CMEJson::Value& data,double length) {
        scale=length;
        const std::string format=data.at("format").text();
        if((format!="gcs-deformation-v1" && format!="gcs-deformation-v2") || data.at("basis").text()!="tensor_Bernstein")
            throw std::invalid_argument("Unsupported dGCS deformation format/basis");
        const auto units=data.at("coefficient_unit").text();
        if(units!="fraction_of_reference_height" && units!="fraction_of_deformation_scale")
            throw std::invalid_argument("Unsupported deformation coefficient units");
        const auto& dims=data.at("grid").items(); if(dims.size()!=3) throw std::invalid_argument("Invalid deformation grid");
        for(int d=0;d<3;++d) { grid[d]=dims[d].integer(); if(grid[d]<2 || grid[d]>10) throw std::invalid_argument("Deformation grid must have 2..10 nodes per axis"); }
        const auto& rows=data.at("coefficients").items();
        if(rows.size()!=size_t(grid[0])) throw std::invalid_argument("Deformation coefficient dimensions disagree");
        coefficients.clear();
        for(const auto& row:rows) {
            if(row.items().size()!=size_t(grid[1])) throw std::invalid_argument("Deformation coefficient dimensions disagree");
            for(const auto& col:row.items()) {
                if(col.items().size()!=size_t(grid[2])) throw std::invalid_argument("Deformation coefficient dimensions disagree");
                for(const auto& value:col.items()) coefficients.push_back(vector(value));
            }
        }
        if(format=="gcs-deformation-v2") {
            if(data.at("anchor_mode").text()!="sun_center" || norm(vector(data.at("anchored_point")))!=0)
                throw std::invalid_argument("Unsupported deformation anchor");
        } else if(data.at("anchored_layer").text()!="native z=0") throw std::invalid_argument("Unsupported deformation anchor");
        std::array<double,10> bx{},by{}; bernstein(grid[0]-1,.5,bx); bernstein(grid[1]-1,.5,by); Vec root{};
        for(int i=0;i<grid[0];++i) for(int j=0;j<grid[1];++j) for(int d=0;d<3;++d) {
            const double c=coefficients[index(i,j,0)][d]; root[d]+=bx[i]*by[j]*c;
            if(format=="gcs-deformation-v1" && std::abs(c)>1e-12) throw std::invalid_argument("Deformation moves its anchored base plane");
        }
        if(norm(root)>2e-12) throw std::invalid_argument("Deformation moves the Sun-center anchor");
        std::vector<Map> cells{*this};
        for(int depth=0;depth<2;++depth) for(int axis=0;axis<3;++axis) {
            std::vector<Map> next; next.reserve(cells.size()*2);
            for(const auto& cell:cells) { auto children=cell.split(axis); next.push_back(std::move(children[0])); next.push_back(std::move(children[1])); }
            cells=std::move(next);
        }
        double refined=0; for(const auto& cell:cells) refined=std::max(refined,cell.derivativeBound(4));
        bound=std::min(derivativeBound(),refined);
        if(bound>.85+1e-12 || !close(bound,data.at("derivative_bound").real(),2e-9) || !close(data.at("safety_limit").real(),.85))
            throw std::invalid_argument("Deformation fails its independently recomputed whole-volume certificate");
        lower.fill(std::numeric_limits<double>::infinity()); upper.fill(-std::numeric_limits<double>::infinity());
        for(int i=0;i<grid[0];++i) for(int j=0;j<grid[1];++j) for(int k=0;k<grid[2];++k) {
            Vec q={scale*(-1.+2.*i/(grid[0]-1)),scale*(-1.+2.*j/(grid[1]-1)),scale*k/(grid[2]-1)};
            for(int d=0;d<3;++d) { q[d]+=scale*coefficients[index(i,j,k)][d]; lower[d]=std::min(lower[d],q[d]); upper[d]=std::max(upper[d],q[d]); }
        }
    }
};

struct Section { Vec center{},e{},f{}; double radius=0,radiusS=0,normal=1,drift=0,rotation=0,s=0; };
struct Model {
    bool cone=false,magnetic=false;
    double height=0,alpha=0,kappa=0,latitude=0,longitude=0,tilt=0,solarRadius=nominalSolarRadius;
    double b=0,rho=0,denominator=1,joinLength=0,frontEnd=0,frontLength=0,length=0,leadingHeight=0;
    double poloidalFlux=0,toroidalFlux=0,twist=0,windingRatio=1;
    int helicity=1,polarity=1;
    Mat rotation{};
    Map map;
    CalendarEpoch fitEpoch;
    std::string shapeFile,magneticFile;
    static constexpr int intervals=4096;
    std::array<double,intervals+1> distances{};

    double speed(double phi) const {
        const double q=(rho+b*kappa*kappa*std::cos(phi))/denominator;
        const double qp=-b*kappa*kappa*std::sin(phi)/denominator;
        return std::hypot(q,qp);
    }
    double frontDistance(double phi) const {
        const double step=2*frontEnd/intervals;
        int i=std::clamp(int((phi+frontEnd)/step),0,intervals-1);
        const double low=-frontEnd+i*step,delta=phi-low;
        return distances[i]+delta*(speed(low)+4*speed(low+.5*delta)+speed(phi))/6;
    }
    Section front(double phi) const {
        const double co=std::cos(phi),si=std::sin(phi);
        const double q=(rho+b*kappa*kappa*co)/denominator,qp=-b*kappa*kappa*si/denominator;
        Section g; g.radius=kappa/denominator*std::hypot(rho+b*co,b*std::sqrt(denominator)*si);
        const double ell=std::hypot(q,qp);
        g.center={0,q*si,b+q*co}; g.e={0,si,co}; g.f={0,co,-si};
        g.normal=q/ell; g.drift=qp/ell; g.rotation=1/ell;
        g.radiusS=q*qp/(g.radius*ell); g.s=joinLength+frontDistance(phi); return g;
    }
    Section leg(double l,int side) const {
        const double ca=std::cos(alpha),sa=std::sin(alpha),slope=kappa/std::sqrt(denominator);
        Section g; g.center={0,side*l*sa,l*ca}; g.e={0,side*ca,-sa}; g.f={0,-sa,-side*ca};
        g.radius=l*slope; g.radiusS=-side*slope; g.s=side<0 ? l : length-l; return g;
    }
    static double coneRadius(double p) {
        const double p2=std::clamp(p,0.,1.)*std::clamp(p,0.,1.);
        return std::sqrt(std::max(0.,2*p2*(1-p2)/(std::sqrt(1+8*p2)+1+2*p2)));
    }
    Vec surface(int patch,double u,double angle) const {
        if(cone) { const double p=std::sin(pi*.5*u),r=height*std::tan(alpha)*coneRadius(p); return {r*std::cos(angle),r*std::sin(angle),height*p}; }
        Section g=patch==0 ? leg(joinLength*u,-1) : patch==1 ? front(frontEnd*(2*u-1)) : leg(joinLength*(1-u),1);
        Vec q=g.center; q[0]+=g.radius*std::sin(angle);
        for(int d=0;d<3;++d) q[d]+=g.radius*std::cos(angle)*g.e[d];
        return q;
    }
    void measureLeadingHeight() {
        leadingHeight=0;
        for(int patch=0;patch<(cone?1:3);++patch) {
            struct Start { double radius,u,theta; }; std::vector<Start> samples;
            for(int i=0;i<=64;++i) for(int j=0;j<96;++j) {
                const double u=i/64.,theta=2*pi*j/96;
                samples.push_back({norm(map.evaluate(surface(patch,u,theta))),u,theta});
            }
            std::sort(samples.begin(),samples.end(),[](const Start& a,const Start& b){return a.radius>b.radius;});
            std::vector<Start> starts;
            for(const auto& candidate:samples) {
                bool separate=true; for(const auto& old:starts) if(std::abs(candidate.u-old.u)<.08 && std::abs(std::remainder(candidate.theta-old.theta,2*pi))<.4) separate=false;
                if(separate) starts.push_back(candidate);
                if(starts.size()==6) break;
            }
            for(auto best:starts) {
                double du=1./64,dt=2*pi/96;
                for(int iteration=0;iteration<180 && du>1e-10;++iteration) {
                    auto next=best;
                    for(int i=-1;i<=1;++i) for(int j=-1;j<=1;++j) {
                        const double u=std::clamp(best.u+i*du,0.,1.),t=best.theta+j*dt;
                        double radius=norm(map.evaluate(surface(patch,u,t)));
                        if(radius>next.radius) next={radius,u,t};
                    }
                    if(next.radius>best.radius) best=next; else {du*=.5;dt*=.5;}
                }
                leadingHeight=std::max(leadingHeight,best.radius);
            }
        }
        if(!(leadingHeight>0) || !std::isfinite(leadingHeight)) throw std::invalid_argument("Imported shape has no finite leading height");
    }
    void prepareReference() {
        if(!(height>0 && alpha>0 && alpha<pi/2 && std::abs(latitude)<=90) || (!cone && !(kappa>0 && kappa<std::sin(alpha))))
            throw std::invalid_argument("Invalid cone/GCS reference geometry");
        if(!cone) {
            denominator=1-kappa*kappa; b=height*(1-kappa)/(1+std::sin(alpha)); rho=b*std::sin(alpha);
            joinLength=b*std::cos(alpha); frontEnd=pi/2+alpha;
            const double step=2*frontEnd/intervals; distances[0]=0;
            for(int i=0;i<intervals;++i) { const double low=-frontEnd+i*step; distances[i+1]=distances[i]+step*(speed(low)+4*speed(low+.5*step)+speed(low+step))/6; }
            frontLength=distances.back(); length=2*joinLength+frontLength;
        }
    }
    bool locate(const Vec& q,Section& g,double& radius,double& theta) const {
        if(cone) return q[2]>0 && q[2]<height && std::hypot(q[0],q[1])<=height*std::tan(alpha)*coneRadius(q[2]/height)*(1+1e-9);
        const double phi=std::atan2(q[1],q[2]-b);
        if(std::abs(phi)<=frontEnd) g=front(phi);
        else { int side=q[1]<0 ? -1:1; double l=side*q[1]*std::sin(alpha)+q[2]*std::cos(alpha); if(!(l>0 && l<joinLength)) return false; g=leg(l,side); }
        Vec delta{}; for(int d=0;d<3;++d) delta[d]=q[d]-g.center[d];
        const double outward=dot(delta,g.e); radius=std::hypot(outward,q[0]); theta=std::atan2(q[0],outward);
        return g.s>0 && g.s<length && radius<=g.radius*(1+1e-9);
    }
    Vec referenceField(const Section& g,double r,double theta) const {
        const double k2=twist*twist*g.radius*g.radius,r2=twist*twist*r*r,logarithm=std::log1p(k2);
        const double factor=k2>0 ? k2/logarithm : 1;
        const double bs=toroidalFlux/(pi*solarRadius*solarRadius*g.radius*g.radius)*factor/(1+r2);
        const double lambda=k2>0 ? std::log1p(r2)/logarithm : r*r/(g.radius*g.radius);
        const double radial=r>0 ? lambda*g.radius*g.radiusS*(1+r2)/(r*(1+k2)) : 0;
        const double delta=.1*std::min(joinLength,frontLength/2);
        auto smooth=[](double t){t=std::clamp(t,0.,1.);return t*t*t*(10+t*(-15+6*t));};
        const double weight=1+(windingRatio-1)*(smooth((g.s-joinLength+delta)/(2*delta))-smooth((g.s-joinLength-frontLength+delta)/(2*delta)));
        const double omega=helicity*twist*weight/(1+(windingRatio-1)*frontLength/length);
        const double co=std::cos(theta),si=std::sin(theta),metric=g.normal+r*co*g.rotation;
        if(!(metric>0)) throw std::runtime_error("Imported GCS section map folds");
        const double be=bs/metric*(g.drift+radial*co-omega*r*si),bx=bs/metric*(radial*si+omega*r*co);
        Vec field{}; for(int d=0;d<3;++d) field[d]=polarity*(bs*g.f[d]+be*g.e[d]+(d==0?bx:0)); return field;
    }
    bool sample(const Vec& positionCm,double expansion,Vec& field) const {
        Vec target=transposeProduct(rotation,positionCm);
        for(double& v:target) v/=solarRadius*expansion;
        Vec q; if(!map.inverse(target,q)) return false;
        Section g; double radius=0,theta=0; if(!locate(q,g,radius,theta)) return false;
        field={};
        if(magnetic) {
            Mat jacobian; map.evaluate(q,&jacobian); const double det=determinant(jacobian);
            if(!(det>0)) throw std::runtime_error("Imported CME has a nonpositive Jacobian");
            field=product(rotation,product(jacobian,referenceField(g,radius,theta)));
            for(double& v:field) v/=det*expansion*expansion;
        }
        return true;
    }
    void setLongitudeOffset(double degrees) {
        const double a=degrees*pi/180;
        const Mat rz={Vec{std::cos(a),-std::sin(a),0},Vec{std::sin(a),std::cos(a),0},Vec{0,0,1}};
        rotation=product(rz,rotation);
    }
    static std::shared_ptr<Model> read(const std::string& shapePath,const std::string& fieldPath,bool expectCone) {
        const auto shape=CMEJson::read(shapePath);
        auto m=std::make_shared<Model>(); m->cone=expectCone; m->shapeFile=shapePath;
        if(shape.at("format").text()!=(expectCone?"dgcs-cone-shape-v1":"gcs-deformable-shape-v1") || !shape.at("geometry_only").truth() || shape.at("coordinate_frame").text()!="HeliographicStonyhurst")
            throw std::invalid_argument("CME type and exported shape format/frame disagree");
        const auto& p=shape.at(expectCone?"reference_cone":"reference_gcs");
        m->height=p.at("height").real(); m->alpha=p.at("half_angle").real()*pi/180;
        m->latitude=p.at("lat").real(); m->longitude=p.at("lon").real();
        if(!expectCone) {m->kappa=p.at("kappa").real();m->tilt=p.at("tilt").real();}
        m->solarRadius=shape.at("solar_radius_cm").real();
        if(!(m->solarRadius>0)) throw std::invalid_argument("Export solar radius must be positive");
        m->fitEpoch=epoch(shape.at("epoch_utc").text());
        m->prepareReference();
        m->rotation=matrix(shape.at("native_to_hgs")); const auto expected=orientation(m->latitude,m->longitude,m->tilt);
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) if(!close(m->rotation[i][j],expected[i][j],1e-11)) throw std::invalid_argument("Exported coordinate rotation disagrees with its reference");
        double scale=m->height;
        if(expectCone) {
            if(shape.at("model").text()!="cone" || shape.at("reference_geometry_policy").text()!="swpc_pycat_bernoulli_lemniscate_v1") throw std::invalid_argument("Unsupported cone reference policy");
            scale=m->height*std::max(1.,std::tan(m->alpha)/(2*std::sqrt(2.)));
            if(!close(shape.at("deformation_scale_rsun").real(),scale)) throw std::invalid_argument("Cone deformation scale disagrees with reference geometry");
        }
        m->map.prepare(shape.at("deformation"),scale);
        if(expectCone && !fieldPath.empty()) throw std::invalid_argument("Cone sources are non-magnetic; do not supply a magnetic file");
        if(!expectCone) {
            if(fieldPath.empty()) throw std::invalid_argument("Deformed GCS needs both shape.json and magnetic_field.json");
            const auto field=CMEJson::read(fieldPath); const auto format=field.at("format").text();
            const bool variable=format=="gcs-deformable-magnetic-field-v2";
            if((format!="gcs-deformable-magnetic-field-v1" && !variable) || field.at("model").text()!=(variable?"front_weighted_reference_with_piola_deformation":"constant_turn_reference_with_piola_deformation") || field.at("coordinate_frame").text()!="HeliographicStonyhurst" || field.at("field_unit").text()!="gauss" || field.at("position_unit").text()!="Rsun" || field.at("flux_unit").text()!="Mx")
                throw std::invalid_argument("Unsupported exported magnetic format/model/units");
            if(!same(p,field.at("reference_gcs")) || !same(shape.at("deformation"),field.at("deformation")) || !same(shape.at("native_to_hgs"),field.at("native_to_hgs")) || !close(m->solarRadius,field.at("solar_radius_cm").real(),1e-12) || std::abs(secondsBetweenEpochs(m->fitEpoch,epoch(field.at("epoch_utc").text())))>1e-6)
                throw std::invalid_argument("Shape and magnetic files do not describe the same member and epoch");
            const auto& in=field.at("inputs"); m->poloidalFlux=in.at("poloidal_flux_mx").real();m->toroidalFlux=in.at("toroidal_flux_mx").real();
            m->helicity=in.at("helicity_sign").integer();m->polarity=in.at("axial_polarity").integer();
            m->windingRatio=in.has("front_winding_ratio")?in.at("front_winding_ratio").real():1;
            if(!(m->poloidalFlux>=0 && m->toroidalFlux>0 && (m->helicity==1||m->helicity==-1) && (m->polarity==1||m->polarity==-1) && m->windingRatio>=1 && m->windingRatio<=1000) || (!variable && m->windingRatio!=1)) throw std::invalid_argument("Invalid exported magnetic fluxes, signs, or winding ratio");
            m->twist=2*pi*(m->poloidalFlux/m->toroidalFlux)/m->length;
            if(!std::isfinite(m->twist) || !close(m->length,field.at("reference_length_rsun").real(),2e-9) || !close(m->twist,field.at("reference_twist_rsun_inverse").real(),2e-9)) throw std::invalid_argument("Exported magnetic normalization disagrees with its reference tube");
            if(!close(field.at("reference_turns").real(),m->poloidalFlux/m->toroidalFlux,2e-9))
                throw std::invalid_argument("Exported magnetic turns disagree with the fluxes");
            if(variable || field.has("winding_profile")) {
                const auto& profile=field.at("winding_profile");
                const auto& interval=profile.at("front_interval_rsun").items();
                const double delta=.1*std::min(m->joinLength,m->frontLength/2);
                const double leg=m->twist/(1+(m->windingRatio-1)*m->frontLength/m->length);
                if(interval.size()!=2 || !close(interval[0].real(),m->joinLength,2e-9) || !close(interval[1].real(),m->joinLength+m->frontLength,2e-9) ||
                    !close(profile.at("front_to_legs_ratio").real(),m->windingRatio,2e-9) ||
                    !close(profile.at("transition_half_width_rsun").real(),delta,2e-9) ||
                    !close(profile.at("weight_integral_rsun").real(),m->length+(m->windingRatio-1)*m->frontLength,2e-9) ||
                    !close(profile.at("leg_winding_rate_rsun_inverse").real(),leg,2e-9) ||
                    !close(profile.at("front_winding_rate_rsun_inverse").real(),leg*m->windingRatio,2e-9) ||
                    !close(profile.at("radial_profile_twist_rsun_inverse").real(),m->twist,2e-9) ||
                    !close(profile.at("phase_origin_arclength_rsun").real(),m->length/2,2e-9))
                    throw std::invalid_argument("Exported winding profile disagrees with the reference and magnetic inputs");
            }
            m->magnetic=true; m->magneticFile=fieldPath;
        }
        m->measureLeadingHeight(); return m;
    }
    static std::shared_ptr<Model> analyticCone(double height,double halfAngle,double lat,double lon,CalendarEpoch time) {
        auto m=std::make_shared<Model>();m->cone=true;m->height=height;m->alpha=halfAngle*pi/180;m->latitude=lat;m->longitude=lon;m->fitEpoch=time;
        m->prepareReference();m->rotation=orientation(lat,lon,0);
        m->map.scale=height*std::max(1.,std::tan(m->alpha)/(2*std::sqrt(2.)));
        m->map.lower={-m->map.scale,-m->map.scale,0};m->map.upper={m->map.scale,m->map.scale,m->map.scale};
        m->measureLeadingHeight();return m;
    }
};
} // namespace ImportedCME
