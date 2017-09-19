#include <wpm.hh>
#include <fstream>

/* Space and space frequency */

void init_Z(double* Z, int nz, double dz) {
	for(int i=0; i<nz; i++) {
		Z[i] = (i+1)*dz;
	}
}

void init_X(double* X, int nx, double dx) {
	for(int i=0; i<nx; i++) {
		X[i] = ( i-std::floor(nx/2) )*dx;
	}
}

void init_KX(cpx* KX, int nx, double dkx) {
	// the ordering is specific to the fft implementation used
	
//	KX[0] = 0;
//	int index=1;
//	for(int i=-nx/2; i<nx/2; i++) {
//		if(i) KX[index++] = cpx(i*dkx,0);
//	}

	int index = 0;
	for(int i=0; i<(nx/2); i++) {
		KX[index++] = cpx(i*dkx,0);
	}
	for(int i=-nx/2; i<0; i++) {
		KX[index++] = cpx(i*dkx,0);
	}

}

void init_FX(double* FX, int nx, double dfx) {
	// the ordering is specific to the fft implementation used
	// for kissfft: 0 -n:-1 1:n-1	
	FX[0] = 0;

	int index=1;
	for(int i=-nx/2; i<nx/2-1; i++) {
		FX[index++] = i*dfx;
	}

}

void meshgrid(double* XY, double* YX, double* X, int nx, double* Y, int ny) {
	//XY = (double*)malloc( nx*ny*sizeof(double));
	//YX = (double*)malloc( ny*nx*sizeof(double));

	for(int i=0; i<ny; i++) {
		for(int j=0; j<nx; j++) {
			XY[i*nx+j] = X[j];
		}
	}

	for(int i=0; i<ny; i++) {
		for(int j=0; j<nx; j++) {
			YX[i*nx+j] = Y[i];
		}
	}
}


/* System */
void init_N(syst_t system, cpx* N, int nz, int nx, double lambda, double dx,
	double dz, double ax) {
	switch(system) {
		case HOMOGENEOUS:	// ==== homogeneous system ====
			init_homog_system(N, nz, nx);				break;
		case BOUNDARY:	// ==== single boundary ====
			init_boundary_system( N, nz, nx);			break;
		case WAVEGUIDE:	// ==== Waveguide ===
			init_waveguide_system(N, nz, nx, dx, lambda);		break;
		case LENS:	// ==== Lens ====
			init_lens_system(N, nz, nx, dz, dx, ax, lambda);	break;
		case GRIN_LENS:	// ==== Equivalent GRIN lens ====
			init_GRIN_system(N, nz, nx, dz, dx, ax, lambda);	break;
		case SLIT:	// ==== slit ====
			init_slit_system(N, nz, nx, dz, dx, lambda);		break;
		default:
			std::cout << "ERROR - unknown system" << std::endl;
			exit(-1);
	}
}


/* Input parameters */

void init_input(int input, double* theta_deg, double* f_sig, double lambda) {
	switch(input) {
		case 0: *theta_deg = *f_sig = 0; break;
		case 1: *theta_deg = 5;  *f_sig = 1/lambda*std::sin((*theta_deg)*PI/180)/1e6;  break;
		case 2: *f_sig = 0.1;    *theta_deg = std::asin( (*f_sig)*1e6*lambda )*180/PI; break;
		case 3: *theta_deg = 30; *f_sig = 1/lambda*std::sin((*theta_deg)*PI/180)/1e6;  break;
		case 4: *f_sig = 0.400;  *theta_deg = std::asin( (*f_sig)*1e6*lambda )*180/PI; break;
		case 5: *theta_deg = 45; *f_sig = 1/lambda*std::sin((*theta_deg)*PI/180)/1e6;  break;
		case 6: *f_sig = 0.55;   *theta_deg = std::asin( (*f_sig)*1e6*lambda )*180/PI; break;
	}
}

/* Wave */

void init_E(wave_t wave, cpx* E, 
	double* X, double* Z, int nz, int nx, double lambda, double A, double theta_deg) {

	switch(wave) {
		case PLANE:
			init_plane(E, X, Z, nz, nx, lambda, A, theta_deg); break;
		case GAUSS:
			init_gauss(E, X, Z, nz, nx, lambda, A, theta_deg); break;
		default:
			std::cout << "ERROR - unknown wave type" << std::endl;
			exit(-1);
	}
}

void init_plane(cpx* E, double* X, double* Z,  int nz, int nx, double lambda,
	double A, double theta_deg) {

	double theta_rad = theta_deg * PI/180;
	double k0 = 2*PI/lambda;
	double kx = k0*std::sin(theta_rad);
	double kz = k0*std::cos(theta_rad);
	cpx i(0,1);

	for(int j=0; j<nx; j++) {
		E[j] = A * std::exp( i*(kx*X[j]+(kz*Z[j])) );
	}

	for(int j=nx; j<(nx*nz); j++) {
		E[j] = cpx(0,0);
	}
}

void init_gauss(cpx* E, double* X, double* Z,  int nz, int nx, double lambda,
	double A, double theta_deg) {

	double theta_rad = theta_deg * PI/180;
	double k0 = 2*PI/lambda;
	double kx = k0*std::sin(theta_rad);
	double kz = k0*std::cos(theta_rad);
	double W0 = 3*lambda;
	cpx i(0,1);

	for(int j=0; j<nx; j++) {
		E[j] = A * std::exp(-(X[j]*X[j])/(W0*W0)) *
			std::exp( i*(kx*X[j]+(kz*Z[j])) );
	}

	for(int j=nx; j<(nx*nz); j++) {
		E[j] = cpx(0,0);
	}
}


cpx fresnelTE( const cpx& N, const cpx& next_N, const cpx& KZ, const cpx& next_KZ) {
	return 2.0*KZ / (KZ+next_KZ);
}

cpx fresnelTM( const cpx& N, const cpx& next_N, const cpx& KZ, const cpx& next_KZ) {
	return 2.0*N*next_N*KZ / (next_N*next_N*KZ + N*N*next_KZ);
}

/* Wave Propagation Mathod */
void wpm( cpx* E, cpx* N, cpx* KX, double* X, double k0, double dz, int nz,
	int nx, int fresnel) {

	kissfft<double> fft(nx,false);
	cpx* S = static_cast<cpx*>(malloc(nx*sizeof(cpx)));
	cpx KZ, next_KZ, F;
	double nk0, kk0, next_nk0;

	cpx li(0,1);	cpx one(1,0);

	fresnel_t fresnel_p;
	switch(fresnel) {
		case 0:	fresnel_p = &fresnelTE;	break;
		case 1:	fresnel_p = &fresnelTM;	break;
		default: 
			std::cout << "Error - Unknown Fresnel Coefficient " << fresnel << std::endl;
			exit(-1);
	}

	for(int i=0; i<nz-1; i++) {

		// ==== Spectrum ====
		fft.transform(&E[i*nx], S);

		for(int kj=0; kj<nx; kj++) {
			for(int j=0; j<nx; j++) {

				// ==== Propagation vector ====
				nk0 = N[i*nx+j].real()*k0;
				kk0 = N[i*nx+j].imag()*k0;
				KZ = sqrt(nk0*nk0 - KX[kj]*KX[kj]);

				next_nk0 = N[(i+1)*nx+j].real()*k0;
				next_KZ = sqrt(next_nk0*next_nk0 - KX[kj]*KX[kj]);

				// exclude evanescent waves, i.e. only real KZ
				if( KZ.imag()<1e-6 && KZ.real()!=0 )  {	

					F = (*fresnel_p)(N[i*nx+j],N[(i+1)*nx+j],KZ,next_KZ);
					double sign = (kj%2)? -1:1;
					E[(i+1)*nx+j] += F * sign  * S[kj]/static_cast<double>(nx) *
						exp(-kk0*dz) * exp(li*(X[j]*KX[kj] + KZ*dz));
				}
			}
		}
	}
	free(S);
}

