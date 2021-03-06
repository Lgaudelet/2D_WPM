#include <cmath>
#include <complex>
#include <algorithm>
#include <iostream>

#include <system.hh>
#include <kissfft.hh>

#ifndef PI
#define PI 3.14159265
#endif


/* typedef */
typedef std::complex<double> cpx;
typedef enum {PLANE, GAUSS} wave_t;
typedef cpx (*fresnel_t)(const cpx&, const cpx&, const cpx&, const cpx&);

/* Space and space frequency */
void init_Z(double* Z, int nz, double dz);
void init_X(double* X, int nx, double dx);
void init_KX(cpx* KX, int nx, double dkx);
void init_FX(double* FX, int nx, double dfx);
void meshgrid(double* XY, double* YX, double* X, int nx, double* Y, int ny);

/* System */
void init_N(syst_t system, cpx* N, int nz, int nx, double lambda, double dx,
	double dz, double ax);

/* Input parameters */
void init_input(int input, double* theta_deg, double* f_sig, double lambda);

/* Wave */
void init_E(wave_t wave, cpx* E, double* X, double* Z, int nz, int nx,
	double lambda, double A, double theta_deg);
void init_plane(cpx* E, double* X, double* Z,  int nz, int nx, double lambda,
	double A, double theta_deg);
void init_gauss(cpx* E, double* X, double* Z,  int nz, int nx, double lambda,
	double A, double theta_deg);

/* fresnel */
cpx fresnelTE(const cpx& N, const cpx& next_N, const cpx& KZ, const cpx& next_KZ);
cpx fresnelTM(const cpx& N, const cpx& next_N, const cpx& KZ, const cpx& next_KZ);


/* Wave Propagation Method */
void wpm( cpx* E, cpx* N, cpx* KX, double* X, double k0, double dz,
	int nz, int nx, int fresnel);

