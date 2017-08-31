#include <cmath>
#include <algorithm>
#include <complex>

#include <iostream>

using namespace std;

void init_homog_system(complex<double>* N, int nz, int nx);

void init_boundary_system(complex<double>* N, int nz, int nx);

void init_waveguide_system(complex<double>* N, int nz, int nx, double dx,
	double lambda);

void init_lens_system(complex<double>* N, int nz, int nx, double dz, double dx,
	double ax, double lambda);

void init_GRIN_system(complex<double>* N, int nz, int nx, double dz, double dx,
	double ax, double lambda);

void init_slit_system(complex<double>* N, int nz, int nx, double dz, double dx,
	double lambda);
