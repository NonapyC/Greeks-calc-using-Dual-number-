// -------------------------------------------------------------------------------------
// Numerical caluclation using automatic differenciation
//
// This code is for comparing the numerical caluclation of Greeks by the result of central differenciation
// and the result of automatic differenciation.
// -------------------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <cmath>
#include "header/NS_dual.hpp"
#include "header/nonapy2.hpp"
#include "LibMK4/integ.hpp"
using namespace LibMK4;
using namespace std;
using namespace LibNS2;

constexpr double T_manki = 0.25;

// -------------------------------------------------------------------------------------
//　Definition of the required function
// -------------------------------------------------------------------------------------
double gaussfunc(double x){ //Gauss function
    return exp( - x * x / 2.0 );
}

double erf(double x){ //Error function
    return ( 1.0 / sqrt( 2.0 * M_PI ) ) * IntegR( gaussfunc, -1000. , x );
}

dual erf(dual x){
    dual ref;
    ref.value = erf(x.value);
    ref.diff = 1.0 / sqrt( 2.0 * M_PI ) * gaussfunc(x.value) * x.diff;
    return ref;
}

double BS(double S_0, double K, double r_kinri, double sigma){ //analytical solution
    double d_1 = ( log( S_0 / K ) + ( r_kinri + sigma * sigma / 2.0 ) * T_manki ) / ( sigma * sqrt( T_manki ) );
    double d_2 = ( log( S_0 / K ) + ( r_kinri - sigma * sigma / 2.0 ) * T_manki ) / ( sigma * sqrt( T_manki ) );
    return S_0 * erf(d_1) - K * exp( - r_kinri * T_manki ) * erf(d_2);
}

dual BS(dual S_0, dual K, dual r_kinri, dual sigma){
    dual d_1 = ( log( S_0 / K ) + ( r_kinri + sigma * sigma / 2.0 ) * T_manki ) / ( sigma * sqrt( T_manki ) );
    dual d_2 = ( log( S_0 / K ) + ( r_kinri - sigma * sigma / 2.0 ) * T_manki ) / ( sigma * sqrt( T_manki ) );
    return S_0 * erf(d_1) - K * exp( - r_kinri * T_manki ) * erf(d_2);
}


// -------------------------------------------------------------------------------------
//　Definition of analytical solution, central diff and automatic diff.
// -------------------------------------------------------------------------------------
double Delta(double S_0, double K, double r_kinri, double sigma, double h){//central diff
    return ( BS(S_0+h*0.5, K, r_kinri, sigma) - BS(S_0-0.5*h, K, r_kinri, sigma) ) / h;
}

double Vega(double S_0, double K, double r_kinri, double sigma, double h){//central diff
    return ( BS(S_0, K, r_kinri, sigma+h*0.5) - BS(S_0, K, r_kinri, sigma-0.5*h) ) / h;
}

double Rho(double S_0, double K, double r_kinri, double sigma, double h){//central diff
    return ( BS(S_0, K, r_kinri+h*0.5, sigma) - BS(S_0, K, r_kinri-0.5*h, sigma) ) / h;
}

double Delta_anal(double S_0, double K, double r_kinri, double sigma){ //analytical
    double d_1 = ( log( S_0 / K ) + ( r_kinri + sigma * sigma / 2.0 ) * T_manki ) / ( sigma * sqrt( T_manki ) );
    double d_2 = ( log( S_0 / K ) + ( r_kinri - sigma * sigma / 2.0 ) * T_manki ) / ( sigma * sqrt( T_manki ) );
    return erf(d_1);
}

double Vega_anal(double S_0, double K, double r_kinri, double sigma){ //analytical
    double d_1 = ( log( S_0 / K ) + ( r_kinri + sigma * sigma / 2.0 ) * T_manki ) / ( sigma * sqrt( T_manki ) );
    double d_2 = ( log( S_0 / K ) + ( r_kinri - sigma * sigma / 2.0 ) * T_manki ) / ( sigma * sqrt( T_manki ) );
    return S_0 * sqrt(T_manki) * gaussfunc(d_1) / sqrt(2.0*M_PI);
}

double rho_anal(double S_0, double K, double r_kinri, double sigma){ //analytical
    double d_2 = ( log( S_0 / K ) + ( r_kinri - sigma * sigma / 2.0 ) * T_manki ) / ( sigma * sqrt( T_manki ) );
    return (T_manki) * K * exp( - r_kinri * T_manki ) * erf(d_2);
}

// -------------------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------------------
int main(){
    ofstream fs;
    fs.open("test.txt");
    for (int i = 1; i < 500; i++) {
        dual x(80.0,0.0);
        dual K(100.0,0.0);
        dual r(0.01,0.0);
        dual sigma(0.3,0.0);

        dual x_d(80.0);
        dual delta = BS(x_d, K, r, sigma);
        dual sigma_d(0.3);
        dual vega = BS(x, K, r, sigma_d);
        dual r_d(0.01);
        dual rho = BS(x, K, r_d, sigma);

        double h = i * 0.00001;
        double RMS_sabun = fabs( Delta(80.0, 100.0, 0.01, 0.3, h) - Delta_anal(80.0, 100.0, 0.01, 0.3) );
        double RMS_dual = fabs( delta.diff - Delta_anal(80.0, 100.0, 0.01, 0.3) );
        double RMS_sabun2 = fabs( Vega(80.0, 100.0, 0.01, 0.3, h) - Vega_anal(80.0, 100.0, 0.01, 0.3) );
        double RMS_dual2 = fabs( vega.diff - Vega_anal(80.0, 100.0, 0.01, 0.3) );
        double RMS_sabun3 = fabs( Rho(80.0, 100.0, 0.01, 0.3, h) - rho_anal(80.0, 100.0, 0.01, 0.3) );
        double RMS_dual3 = fabs( rho.diff - rho_anal(80.0, 100.0, 0.01, 0.3) );

        fs << h << " " << RMS_sabun << " " << RMS_dual << " " << RMS_sabun2 << " " << RMS_dual2 << " " << RMS_sabun3 << " " << RMS_dual3  << endl;
    }
    fs.close();

    FILE *pp;
    pp = popen("python3","w");
    fprintf(pp, "import numpy as np \n");
    fprintf(pp, "import matplotlib.pyplot as plt \n");
    fprintf(pp, "plt.style.use('ggplot') \n");
    fprintf(pp, "data = np.loadtxt('test.txt')\n");
    fprintf(pp, "x = data[:,0]\n");
    fprintf(pp, "y = data[:,1]\n");
    fprintf(pp, "y2 = data[:,2]\n");
    fprintf(pp, "y3 = data[:,3]\n");
    fprintf(pp, "y4 = data[:,4]\n");
    fprintf(pp, "y5 = data[:,5]\n");
    fprintf(pp, "y6 = data[:,6]\n");
    // fprintf(pp, "plt.plot(x,y,label='by central diff') \n");
    // fprintf(pp, "plt.plot(x,y2,label='by automatic diff') \n");
    // fprintf(pp, "plt.plot(x,y3,label='by central diff') \n");
    // fprintf(pp, "plt.plot(x,y4,label='by automatic diff') \n");
    fprintf(pp, "plt.plot(x,y5,label='by central diff') \n");
    fprintf(pp, "plt.plot(x,y6,label='by automatic diff') \n");
    fprintf(pp, "plt.xlabel('h:difference') \n");
    fprintf(pp, "plt.ylabel('RMS') \n");
    fprintf(pp, "plt.title('rho numerical caluclation') \n");
    fprintf(pp, "plt.minorticks_on() \n");
    fprintf(pp, "plt.gca().ticklabel_format(style='sci', scilimits=(0,0), axis='y') \n");
    fprintf(pp, "plt.legend() \n");
    fprintf(pp, "plt.show() \n");
    pclose(pp);

    return 0;
}
