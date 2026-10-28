#pragma once

#include "MHD_IO.H"
#include "MHD_Thomson.H"

namespace HelioCubed
{
// Views of synchronized owned states, ordered coarse to fine. An AMR metric
// is the complete cell Jacobian; a uniform-grid metric is its radial factor.
// Sample only owned cells: no stale ghosts, panel interpolation, or duplicate
// coarse/fine contributions enter this diagnostic.
struct JmapGrid
{
    const MBLevelBoxData<double,NUMCOMPS,HOST>* state;
    const MBLevelBoxData<double,1,HOST>* metric;
    bool completeJacobian;
};

inline SightVector jmapObserver(const std::string& filename,double time)
{
    const auto& trajectory=probeTrajectory(filename);
    const double epoch=getPhysTime(time);
    if(epoch<trajectory.time.front() || epoch>trajectory.time.back())
        throw std::runtime_error("J-map observer trajectory does not cover simulation time: "+filename);
    const double radius=interpolate(trajectory.time,trajectory.radius,epoch,false)*c_AU;
    const double latitude=interpolate(trajectory.time,trajectory.latitude,epoch,false)*c_PI/180;
    // Unwrap before interpolating, including a 359 -> 0 degree crossing.
    auto longitude=trajectory.longitude;
    for(size_t i=1;i<longitude.size();++i)
        longitude[i]=longitude[i-1]+std::remainder(longitude[i]-longitude[i-1],360.);
    const double phi=(interpolate(trajectory.time,longitude,epoch,false)+181.02)*c_PI/180;
    if(!(radius>c_SR) || !std::isfinite(latitude) || std::abs(latitude)>c_PI/2)
        throw std::runtime_error("Invalid J-map observer trajectory position");
    return {radius*cos(latitude)*cos(phi),radius*cos(latitude)*sin(phi),radius*sin(latitude)};
}

inline void writeJmaps(const std::vector<JmapGrid>& grids,int iteration,double time,bool force=false,
    const std::string& suffix="")
{
    if(ParseInputs::get_jmaps().empty()) return;
    if(grids.empty()) MayDay<void>::Abort("J-map diagnostic needs a covering base grid");
    PR_TIME("HelioCubed::Jmap");
    static std::map<std::string,double> lastTimes;
    const auto shell=cubedSphereShellGeometry();
    const int bins=ParseInputs::get_Jmap_elongation_bins();
    const double minimum=ParseInputs::get_Jmap_min_elongation(),maximum=ParseInputs::get_Jmap_max_elongation();
    const double step=ParseInputs::get_Jmap_los_step()*c_AU,cadence=ParseInputs::get_Jmap_cadence();
    if(2*shell.r1/step>1000000)
        MayDay<void>::Abort("Jmap_los_step is too small (more than one million samples per ray)");
    for(const auto& config : ParseInputs::get_jmaps())
    {
        const std::string filename=config.filename+suffix;
        std::ostringstream description;
        description<<std::setprecision(17)<<"# observer="<<config.observer
            <<" reference="<<config.reference<<" PA_deg="<<config.positionAngle<<" elongation="<<minimum<<","<<maximum<<","<<bins
            <<" los_step_AU="<<ParseInputs::get_Jmap_los_step()<<" shell_cm="<<shell.r0<<","<<shell.r1;
        int error=0;
        auto last=lastTimes.find(filename);
        if(last==lastTimes.end())
        {
            double previous=-1;
            if(procID()==0)
            {
                std::ifstream file(filename);std::string line;
                if(file.good())
                {
                    bool compatible=false;
                    int rows=0;
                    while(std::getline(file,line))
                    {
                        if(line==description.str()) compatible=true;
                        if(line.empty() || line[0]=='#' || line[0]=='t') continue;
                        std::replace(line.begin(),line.end(),',',' ');
                        std::istringstream row(line);
                        double t,epoch,iter,elongation,b,pb,proxy;
                        if(!(row>>t>>epoch>>iter>>elongation>>b>>pb>>proxy)
                            || !std::isfinite(t+epoch+iter+elongation+b+pb+proxy)
                            || b<0 || pb<0 || proxy<0 || pb>b*(1+1e-12)) {error=1;break;}
                        if(previous>=0 && t!=previous)
                        {
                            if(t<previous || rows!=bins) {error=1;break;}
                            rows=0;
                        }
                        if(rows>=bins || abs(elongation-(minimum+(maximum-minimum)*rows/(bins-1)))>1e-9)
                        {error=1;break;}
                        previous=t;++rows;
                    }
                    if(!compatible || rows!=bins) error=1;
                }
            }
#ifdef PR_MPI
            MPI_Bcast(&error,1,MPI_INT,0,MPI_COMM_WORLD);
            MPI_Bcast(&previous,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
            if(error) MayDay<void>::Abort(("Incompatible or malformed existing J-map output: "+filename).c_str());
            last=lastTimes.emplace(filename,previous).first;
        }
        if(time<=last->second || (!force && last->second>=0
            && floor(time/cadence)==floor(last->second/cadence))) continue;
        SightVector observer,reference;
        try {
            observer=jmapObserver(config.observer,time);
            if(!config.reference.empty()) reference=jmapObserver(config.reference,time);
        }
        catch(const std::exception& e) {MayDay<void>::Abort(e.what());}
        std::vector<double> local(3*bins,0),brightness(3*bins,0),length(bins,0);
        std::vector<std::map<std::pair<int,int>,double>> angular(grids.size());
        int badDensity=0;
        for(int bin=0;bin<bins;++bin)
        {
            const double elongation=minimum+(maximum-minimum)*bin/(bins-1);
            SightVector direction;
            try {direction=sightDirection(observer,elongation,config.positionAngle,
                config.reference.empty() ? nullptr : &reference);}
            catch(const std::exception& e) {MayDay<void>::Abort(e.what());}
            for(const auto& segment : sightSegments(observer,direction,shell.r0,shell.r1))
            {
                const int count=std::max(1,int(ceil((segment[1]-segment[0])/step)));
                const double ds=(segment[1]-segment[0])/count;
                length[bin]+=segment[1]-segment[0];
                for(int sample=0;sample<count;++sample)
                {
                    const double distance=segment[0]+(sample+0.5)*ds;
                    Array<double,DIM> position,xi;
                    for(int d=0;d<3;++d) position[d]=observer[d]+distance*direction[d];
                    const double x=abs(position[0]),y=abs(position[1]),z=abs(position[2]);
                    const int block=z>=x && z>=y ? (position[2]>=0 ? 1 : 0)
                        : x>=y ? (position[0]>=0 ? 3 : 5) : (position[1]>=0 ? 4 : 2);
                    f_cubedSphereInverseMap(xi,position,block,shell);
                    for(int level=int(grids.size())-1;level>=0;--level)
                    {
                        const auto& grid=grids[level];const auto& layout=grid.state->layout();
                        const Point sizes=layout.domain().getBlock(block).sizes();
                        Point cell;
                        for(int d=0;d<DIM;++d) cell[d]=std::clamp(int(floor(xi[d]*sizes[d])),0,sizes[d]-1);
                        const auto patch=cell/layout.getBlock(block).boxSize();
                        const auto global=layout.partition().find(patch,block);
                        if(global==layout.numBoxes()) continue;
                        const MBIndex index(layout.partitionPtr(),global);
                        if(layout.procID(index)==Proto::procID())
                        {
                            double jacobian=(*grid.metric)[index](cell,0);
                            if(!grid.completeJacobian)
                            {
                                const auto key=std::make_pair(cell[1],cell[2]);
                                auto entry=angular[level].find(key);
                                if(entry==angular[level].end())
                                {
                                    Point p(0,cell[1],cell[2]);
                                    const auto det=CubedSphereShell::detA<double,HOST>(Box(p,p),1./sizes[1],0);
                                    entry=angular[level].emplace(key,det(p,0)).first;
                                }
                                jacobian*=entry->second;
                            }
                            const double ne=(*grid.state)[index](cell,0)/(jacobian*c_MP);
                            if(!(ne>=0) || !std::isfinite(ne)) {badDensity=1;break;}
                            const double radius=sqrt(position[0]*position[0]+position[1]*position[1]+position[2]*position[2]);
                            const double cosine=-(position[0]*direction[0]+position[1]*direction[1]+position[2]*direction[2])/radius;
                            const auto kernel=thomsonKernel(radius,cosine);
                            for(int c=0;c<2;++c) local[3*bin+c]+=ne*kernel[c]*ds;
                            // Original MS-FLUKSS printJmap diagnostic: LOS mean
                            // of ne*(r/AU)^2 over available modeled samples.
                            local[3*bin+2]+=ne*(radius/c_AU)*(radius/c_AU)*ds;
                        }
                        break; // Finest covering cell, even when another rank owns it.
                    }
                }
            }
        }
#ifdef PR_MPI
        MPI_Allreduce(MPI_IN_PLACE,&badDensity,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        MPI_Reduce(local.data(),brightness.data(),3*bins,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
#else
        brightness=local;
#endif
        if(badDensity) MayDay<void>::Abort("Nonfinite or negative electron density in J-map diagnostic");
        if(procID()==0)
        {
            const bool first=last->second<0;
            std::ofstream file(filename,std::ios::app);
            if(first) file<<"# HelioCubed synthetic J-map v1; full 3D, finest owned cell averages\n"
                <<description.str()<<"\n# B and pB are relative to disk-center solar radiance; limb darkening u=0.63; ne=rho/mp.\n"
                <<"# Only the modeled shell contributes; no exterior density or instrument response is assumed.\n"
                <<"# mean_ne_r2_cm3 is the MS-FLUKSS density proxy: LOS mean of ne*(r/AU)^2.\n"
                <<"time_s,decimal_year,iteration,elongation_deg,B_Bsun,pB_Bsun,mean_ne_r2_cm3\n";
            file<<std::setprecision(17);
            for(int bin=0;bin<bins;++bin)
                file<<time<<','<<getPhysTime(time)<<','<<iteration<<','<<minimum+(maximum-minimum)*bin/(bins-1)
                    <<','<<brightness[3*bin]<<','<<brightness[3*bin+1]<<','
                    <<(length[bin]>0 ? brightness[3*bin+2]/length[bin] : 0)<<'\n';
            file.close();error=!file;
        }
#ifdef PR_MPI
        MPI_Bcast(&error,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
        if(error) MayDay<void>::Abort(("Could not write J-map: "+filename).c_str());
        last->second=time;
        pout(0)<<"J-map "<<filename<<" written at iteration "<<iteration<<std::endl;
    }
}
}
