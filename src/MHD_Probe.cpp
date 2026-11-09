#include "MHD_Error.H"
#include "Proto.H"
#include "MHD_Mapping.H"
#include "MHD_Input_Parsing.H"
#include "MHD_Probe.H"
#include "MHD_Constants.H"
#include <iomanip>
#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <time.h>
extern Parsefrominputs inputs;

using namespace std;
/// @brief MHD_Probe namespace
namespace MHD_Probe {

    //======================================================================

    // Returns interpolated value at x from parallel arrays ( xData, yData )
    //   Assumes that xData has at least two elements, is sorted and is strictly monotonic increasing
    //   boolean argument extrapolate determines behaviour beyond ends of array (if needed)
    double interpolate( vector<double> &xData, vector<double> &yData, double x, bool extrapolate )
    {
        int size = xData.size();

        int i = 0;                                                                  // find left end of interval for interpolation
        if ( x >= xData[size - 2] )                                                 // special case: beyond right end
        {
            i = size - 2;
        }
        else
        {
            while ( x > xData[i+1] ) i++;
        }
        double xL = xData[i], yL = yData[i], xR = xData[i+1], yR = yData[i+1];      // points on either side (unless beyond ends)
        if ( !extrapolate )                                                         // if beyond ends of array and not extrapolating
        {
            if ( x < xL ) yR = yL;
            if ( x > xR ) yL = yR;
        }

        double dydx = ( yR - yL ) / ( xR - xL );                                    // gradient

        return yL + dydx * ( x - xL );                                              // linear interpolation
    }

    time_t mktimeUTC(struct tm* time_tm)
    {
        // *** enter in UTC mode
        char* oldTZ = getenv("TZ");
        putenv("TZ=UTC");
        tzset();
        // ***

        time_t ret = mktime ( time_tm );

        // *** Restore previous TZ
        if(oldTZ == NULL)
        {
            putenv("TZ=");
        }
        else
        {
            char buff[255];
            sprintf(buff,"TZ=%s",oldTZ);
            putenv(buff);
        }
        tzset();
        // ***

        return ret;
    }

    time_t getPosixTime(double a_time)
    {
        int year     = (int)(floor(a_time));
        int daysYear = (year % 4 == 0 ? 366 : 365);
            
        tm time_tm;
        bzero(&time_tm,sizeof(tm));
        
        time_tm.tm_year = year - 1900;
        time_tm.tm_sec  = (int)((a_time - year)*daysYear*24.0*3600.0);
        time_tm.tm_mday = 1;
        time_tm.tm_isdst = 0;
        
        time_t posix_time = mktimeUTC(&time_tm);
        
        return posix_time;  
    }

    double getPhysTime(double a_time)
    {      
        // double   timeRef      = eos_AU/m_lismV; // sec    
        time_t startBC      = getPosixTime(inputs.BC_start_time);
        double   timeSec      = (a_time);//*timeRef; // time in sec
        time_t timeSec_t    = (time_t)(floor(timeSec));
        
        time_t curTime      = startBC + timeSec_t;
        
        tm curTime_tm;
        //gmtime_r(&curTime,&curTime_tm);
        localtime_r(&curTime,&curTime_tm);
        
        
        int daysYear = (curTime_tm.tm_year % 4 == 0 ? 366 : 365);
        
        double PhysTime = curTime_tm.tm_year + 1900.0 + 
                        (double)(curTime_tm.tm_yday)/(double)(daysYear) +
                        (double)(curTime_tm.tm_hour)/(double)(daysYear*24.0) +
                        (double)(curTime_tm.tm_min )/(double)(daysYear*24.0*60.0) +
                        (double)(curTime_tm.tm_sec+timeSec-timeSec_t)/(double)(daysYear*24.0*3600.0);
                                
        return PhysTime;                 
        
    }

    void Probe(MHDLevelDataState& state,
                const double a_time,
                bool give_space)

    {
        int pid = procID();
        ofstream outputFile;
    	outputFile.open(inputs.Probe_data_file,std::ios::app);
		if(pid==0 && give_space) outputFile << endl;
        double a_dx = state.m_dx;
        double a_dy = state.m_dy;
        double a_dz = state.m_dz;
        double a_gamma = state.m_gamma;
        std::string probe_file = inputs.Probe_trajectory_file;
        int number_of_lines = 0;
        int totvars = 5; // Year DOY Radius Latitude Longitude
        std::string line;
        std::ifstream myfile(probe_file);

        MHD::require(myfile.is_open(), "Error: Unable to open trajectory file");

        while (std::getline(myfile, line))
            ++number_of_lines;
        
        int row, col;
        double *my_array;
        my_array = new double[number_of_lines*totvars];

        ifstream pFile (probe_file);   
            row = 0;
            while(!pFile.eof())
            {
                getline(pFile, line);
                stringstream ss(line);
                col = 0;
                
                while(ss >>  my_array[row*totvars +col])
                {
                    
                    col++;
                }	
                row++;
            } 
        pFile.close();

        vector<double> time_traj_file, r_traj_file, lat_traj_file, lon_traj_file;
        int year, day; 
        double rad_AU, latitude, longitude, relative_time;
        bool leap_year;
        for (int i=0; i<number_of_lines; i++){
            year      = my_array[i*totvars+0];
            day       = my_array[i*totvars+1];
            rad_AU    = my_array[i*totvars+2];
            latitude  = my_array[i*totvars+3];
            longitude = my_array[i*totvars+4];
            leap_year = false;
            if ((year % 4) == 0){
                leap_year = true;
                if ((year % 100) == 0 && (year % 400) != 0) leap_year = false;
            }
            if (leap_year){
                // need to check with Tae if the values are at start of day or end of day. Will use (day-1) then. 
                relative_time = year + (day-1)/366.0;
            } else {
                relative_time = year + (day-1)/365.0;
            }

            time_traj_file.push_back(relative_time);
            r_traj_file.push_back(rad_AU);
            lat_traj_file.push_back(latitude);
            lon_traj_file.push_back(longitude);
        }
        delete[] my_array;

        double physical_time = getPhysTime(a_time);
        double r_now = interpolate(time_traj_file, r_traj_file, physical_time, false);
        double latitude_now = interpolate(time_traj_file, lat_traj_file, physical_time, false);
        double longitude_now = interpolate(time_traj_file, lon_traj_file, physical_time, false);
        r_now *= c_AU;
        latitude_now = (90.0-latitude_now); //To co-latitude
        longitude_now = longitude_now + 181.02; // HGI to Carrington
        if (longitude_now >= 360.0) longitude_now -= 360;
        // longitude_now = longitude_now - 1.792951949248379e+02; // HGI to Carrington
        if (longitude_now < 0.0) longitude_now += 360;

        const double inertial_longitude = longitude_now;
        if (inputs.corotating_frame == 1)
        {
            // The trajectory is inertial; sample it on the rotating grid.
            longitude_now = fmod(longitude_now -
                    inputs.solar_angular_velocity()*a_time*180.0/c_PI, 360.0);
            if (longitude_now < 0.0) longitude_now += 360.0;
        }

        double x_n = sin(latitude_now*c_PI/180)*cos(longitude_now*c_PI/180);
        double y_n = sin(latitude_now*c_PI/180)*sin(longitude_now*c_PI/180);
        double z_n = cos(latitude_now*c_PI/180);

        double H3_to_RTN_00 = x_n;
        double H3_to_RTN_01 = y_n;
        double H3_to_RTN_02 = z_n;
        double H3_to_RTN_10 = -y_n;
        double H3_to_RTN_11 = x_n;
        double H3_to_RTN_12 = 0.0;
        double H3_to_RTN_20 = -x_n*z_n;
        double H3_to_RTN_21 = -y_n*z_n;
        double H3_to_RTN_22 = x_n*x_n + y_n*y_n;


        double R_t = (inputs.r_out*c_AU - inputs.r_in*c_AU)/(exp(inputs.C_rad) - 1.0);
        double eta0 = (1.0/inputs.C_rad)*log(1.0+(r_now - inputs.r_in*c_AU)/R_t);
        double eta1 = (latitude_now*c_PI/180.0)/c_PI;
        double eta2 = (longitude_now*c_PI/180.0)/2.0/c_PI;

        double pt0 = eta0/a_dx;
        double pt1 = eta1/a_dy;
        double pt2 = eta2/a_dz;

        int pt0_nearest, pt1_nearest, pt2_nearest;
        int pt0_neigbor, pt1_neigbor, pt2_neigbor;

        pt0_nearest = floor(pt0);
        pt1_nearest = floor(pt1);
        pt2_nearest = floor(pt2);

        if (round(pt0) == floor(pt0)){
            pt0_neigbor = pt0_nearest - 1;
        } else {
            pt0_neigbor = pt0_nearest + 1;
        }   

        if (round(pt1) == floor(pt1)){
            pt1_neigbor = pt1_nearest - 1;
        } else {
            pt1_neigbor = pt1_nearest + 1;
        }  

        if (round(pt2) == floor(pt2)){
            pt2_neigbor = pt2_nearest - 1;
        } else {
            pt2_neigbor = pt2_nearest + 1;
        }   

        double probed_values[NUMCOMPS];
        double probed_values_primitive[NUMCOMPS];
        Point index_cc(pt0_nearest,pt1_nearest,pt2_nearest);
        Point index_n0(pt0_neigbor,pt1_nearest,pt2_nearest);
        Point index_n1(pt0_nearest,pt1_neigbor,pt2_nearest);
        Point index_n2(pt0_nearest,pt1_nearest,pt2_neigbor);
        for (auto dit : state.m_U){
            Box dbx0 = state.m_U[dit].box();
            BoxData<double,NUMCOMPS> U_dim(dbx0);
            state.m_U[dit].copyTo(U_dim);
            MHDOp::NonDimToDimcalc(U_dim);
            BoxData<double,DIM> x_sph_cc(dbx0);
            MHD_Mapping::get_sph_coords_cc(x_sph_cc,dbx0,a_dx,a_dy,a_dz);

            
            if (dbx0.containsPoint(index_cc)){
                for(int i=0; i<NUMCOMPS; i++){
                    probed_values[i] = U_dim(index_cc, i);
                    if (dbx0.containsPoint(index_n0)) probed_values[i] += (U_dim(index_n0, i)-U_dim(index_cc, i))*(r_now          - x_sph_cc(index_cc,0))/(x_sph_cc(index_n0,0) - x_sph_cc(index_cc,0));
                    if (dbx0.containsPoint(index_n1)) probed_values[i] += (U_dim(index_n1, i)-U_dim(index_cc, i))*(latitude_now*c_PI/180   - x_sph_cc(index_cc,1))/(x_sph_cc(index_n1,1) - x_sph_cc(index_cc,1));
                    if (dbx0.containsPoint(index_n2)) probed_values[i] += (U_dim(index_n2, i)-U_dim(index_cc, i))*(longitude_now*c_PI/180  - x_sph_cc(index_cc,2))/(x_sph_cc(index_n2,2) - x_sph_cc(index_cc,2));
                }

                
                double v2 = 0.0;
                double B2 = 0.0;
                double gamma = a_gamma;
                probed_values_primitive[0] = probed_values[0];

                for (int i = 1; i <= DIM; i++)
                {
                    double v, B;
                    v = probed_values[i] / probed_values[0];
                    B = probed_values[DIM+1+i];
                    probed_values_primitive[i] = v;
                    probed_values_primitive[DIM+1+i] = B;
                    v2 += v*v;
                    B2 += B*B;
                }

                probed_values_primitive[DIM+1] = (probed_values[DIM+1] - .5 * probed_values[0] * v2  - B2/8.0/c_PI) * (gamma - 1.0);

                double rho = probed_values_primitive[0]/c_MP; // /cm^3;
                double Vx = probed_values_primitive[1]/1e5; // km/s;
                double Vy = probed_values_primitive[2]/1e5; // km/s;
                double Vz = probed_values_primitive[3]/1e5; // km/s;
                double p = probed_values_primitive[4]*1e12; // picodyne/cm*2
                double Bx = probed_values_primitive[5]*1e6; // microGauss
                double By = probed_values_primitive[6]*1e6; // microGauss
                double Bz = probed_values_primitive[7]*1e6; // microGauss

                if (inputs.corotating_frame == 1)
                {
                    // Return inertial velocities for comparison with observations.
                    const double rotation_speed = inputs.solar_angular_velocity()*
                            r_now*sin(latitude_now*c_PI/180.0)/1e5;
                    Vx -= rotation_speed*sin(longitude_now*c_PI/180.0);
                    Vy += rotation_speed*cos(longitude_now*c_PI/180.0);
                }

                double VR, VT, VN, BR, BT, BN;

                VR = H3_to_RTN_00*Vx + H3_to_RTN_01*Vy + H3_to_RTN_02*Vz;
                VT = H3_to_RTN_10*Vx + H3_to_RTN_11*Vy + H3_to_RTN_12*Vz;
                VN = H3_to_RTN_20*Vx + H3_to_RTN_21*Vy + H3_to_RTN_22*Vz;

                BR = H3_to_RTN_00*Bx + H3_to_RTN_01*By + H3_to_RTN_02*Bz;
                BT = H3_to_RTN_10*Bx + H3_to_RTN_11*By + H3_to_RTN_12*Bz;
                BN = H3_to_RTN_20*Bx + H3_to_RTN_21*By + H3_to_RTN_22*Bz;

                // if (probed_values[0] != 0) {
                    outputFile << setw(16) << setprecision(12) << physical_time
                    << setw(11) << setprecision(4) << r_now/c_AU		
                    << setw(11) << setprecision(4) << 90-latitude_now
                    << setw(11) << setprecision(4) << inertial_longitude
                    << setw(11) << setprecision(4) << rho
                    << setw(11) << setprecision(4) << VR
                    << setw(11) << setprecision(4) << VT
                    << setw(11) << setprecision(4) << VN
                    << setw(11) << setprecision(4) << p
                    << setw(11) << setprecision(4) << BR
                    << setw(11) << setprecision(4) << BT
                    << setw(11) << setprecision(4) << BN
                    << endl;
                // }
            }
        }

        outputFile.close();
       
    }
}
