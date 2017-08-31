#include <system.hh>


using namespace std;

void init_homog_system(complex<double>* N, int nz, int nx){

	complex<double> n(1.6,0);
	
	for(int i=0; i<nz; i++) {
		for(int j=0; j<nx; j++) {
			N[i*nx + j] = n;
		}
	}

}

void init_boundary_system(complex<double>* N, int nz, int nx) {

	// unrolling would be more effective than the test but it's only
	// initialization so let's give priority to concision

	complex<double> n1(1,0);
	complex<double> n2(1.15,0);

	for(int i=0; i<nz; i++) {
		for(int j=0; j<nx; j++) {
			N[i*nx + j] = i<(nz/2)? n1:n2;
		}
	}

}

void init_waveguide_system(complex<double>* N, int nz, int nx, double dx, double lambda) {

	complex<double> n_core(1.8,0);
	complex<double> n_clad(1,0);
	double D = 5*lambda / 2;

	int j0 = nx/2;

	for(int i=0; i<nz; i++) {
		for(int j=0; j<nx; j++) {
			N[i*nx+j] = (D >= abs((j - j0)*dx))? n_core:n_clad;
		}
	}

}

void init_lens_system(complex<double>* N, int nz, int nx, double dz, double dx, double ax, double lambda) {

	// lens diameter
	double D = ax;

	// radius left and right
	double Rl = 7*lambda; // radius left
	double Rr = -7*lambda; // radius right
	// bi-convex:		l 10	r -5
	// bi-convex:		l -5	r 10
	// bi-concave:		l -10	r 5
	// bi-concave:		l 5	r -10
	// concave-convex:	l 5	r 10
	// concave-convex:	l -5	r -10

	// Thickness left and right and dense
	double Tl = Rl - Rl * sqrt( 1 - pow(D/Rl,2)/4 );
	double Tr = Rr - Rr * sqrt( 1 - pow(D/Rr,2)/4 );
	double Td = 1*lambda;

	// position of the lens
	int i0 = nz/2-1;
	int j0 = nx/2-1;

	complex<double> n_lens(1.8,0);
	complex<double> n_medium(1,0);

	int Tl_inum = fabs(Tl)/dz;
	int Tr_inum = fabs(Tr)/dz;
	int Td_inum = fabs(Td/2)/dz;
	int T_inum = Tl_inum + Tr_inum + 2*Td_inum;

	int Rl_inum = abs(Rl)/dz;
	int Rr_inum = abs(Rr)/dz;

	int left_center = i0 - Tl_inum + Rl_inum - Td_inum;
	int right_center = i0 + Tr_inum - Rr_inum + Td_inum;

	complex<double> this_n;
	int this_i;
	
	for(int i=0; i<nz; i++) {
		for(int j=0; j<nx; j++) {
			
			this_n = n_medium;
			this_i = i;

			// left hemisphere of lens
			if ((max(i0 - Tl_inum - Td_inum, 1) <= i) && (i <= i0 )) {
				if(Rl>0) {
					this_n = ( pow(Rl,2) >= ( pow((j-j0)*dx,2) + pow((left_center-i)*dz,2)) )? n_lens:n_medium;
					this_i = i;
				}
				else {
					this_n = ( pow(Rl,2) >= ( pow((j-j0)*dx,2) + pow((left_center-i)*dz,2)) )? n_medium:n_lens;
					this_i = i0 - ( i - ( i0 - Tl_inum - Td_inum ) );
				}
			}

			// right hemisphere of lens
			else if ((i0 <= i) && (i <= min(i0 + Tr_inum + Td_inum, nz))) {
				if (Rr<0) {
					this_n = ( pow(Rr,2) >= ( pow((j-j0)*dx,2) + pow((i-right_center)*dz,2)) )? n_lens:n_medium;
					this_i = i;
				}
				else {
					this_n = ( pow(Rr,2) >= ( pow((j-j0)*dx,2) + pow((i-right_center)*dz,2 )) )? n_medium:n_lens;
					this_i = i0 + ( Tr_inum + Td_inum - ( i - i0 ) );
				}
			}

			N[ this_i*nx + j] = this_n;
		
		}
	}

}

void init_GRIN_system(complex<double>* N, int nz, int nx, double dz, double dx, double ax, double lambda) {

	// lens diameter
	double D = ax;

	// radius left and right
	double Rl = 7*lambda; // radius left
	double Rr = -7*lambda; // radius right

	// Thickness left and right and dense
	double Tl = Rl - Rl * sqrt( 1 - pow(D/Rl,2)/4 );
	double Tr = Rr - Rr * sqrt( 1 - pow(D/Rr,2)/4 );
	double Td = 1*lambda;

	// position of the lens
	int i0 = nz/2;
	int j0 = nx/2;

	complex<double> n_lens(1.8,0);
	complex<double> n_medium(1,0);

	int Tl_inum = fabs(Tl)/dz;
	int Tr_inum = fabs(Tr)/dz;
	int Td_inum = fabs(Td/2)/dz;
	int T_inum = Tl_inum + Tr_inum + 2*Td_inum;

	int Rl_inum = abs(Rl)/dz;
	int Rr_inum = abs(Rr)/dz;

	int left_center = i0 - Tl_inum + Rl_inum - Td_inum;
	int right_center = i0 + Tr_inum - Rr_inum + Td_inum;

	double theta_max_left = asin( ax/(2*fabs(Rl)) );
	double l_adjust = fabs(Rl) * cos(theta_max_left); 
	double theta_max_right = asin( ax/(2*fabs(Rr)) );
	double r_adjust = fabs(Rr) * cos(theta_max_right);

	double d_rod = 0.5*lambda;
	double d_rod_inum = floor(fabs(d_rod)/dz);

	double dl_tot, dr_tot, dl, dr, d_rel;
	complex<double> n_rod;

	for(int i=0; i<nz; i++) {
		for(int j=0; j<nx; j++) {
			//cout<<"i:"<<i<<"\tj:"<<j<<endl;				
			if(( abs((i0-i))*dz ) <= ( d_rod/2 )) {

				// total lens thickness in infinite aperture
				dl_tot = abs(Rl) * sqrt( 1 - ((j-j0)*dx / pow(abs(Rl),2)) );
				dr_tot = abs(Rr) * sqrt( 1 - ((j-j0)*dx / pow(abs(Rr),2)) );

				// adjusted lens thickness in limited aperture ax
				dl = max(static_cast<double>(0),dl_tot - l_adjust);
				dr = max(static_cast<double>(0),dr_tot - r_adjust);

				if(Rl<0) dl = Tl - dl; // concave
				if(Rr>0) dr = Tr - dr; // concave
								 
				// refractive index of GRIN rod
				// obtained from optical path length
				// n_rod(x) * d_rod = n_lens * d_lens(x)
				d_rel = ( dl + dr + Td ) / d_rod;
				n_rod = d_rel * n_lens;

				N[ i*nx + j ] = n_rod/d_rod_inum;
			}
			else {
				N[ i*nx + j ] = n_medium;
			}

		}
	}

}

void init_slit_system(complex<double>* N, int nz, int nx, double dz, double dx, double lambda) {

	int j0 = nx/2;
	int i0 = 2e-6/dz;

	complex<double> n_medium(1,0);
	complex<double> n_slit(1.4,1);

	double T = 0.25*lambda;
	double D = 4*lambda;

	int T_inum = abs(T)/dz;

	for(int i=0; i<nz; i++) {
		for(int j=0; j<nx; j++) {
			N[i*nx+j] = (T/2 >= abs((i - i0)*dz)) && (D/2 <= abs((j - j0)*dx))? n_medium:n_slit;

		}
	}

}


















