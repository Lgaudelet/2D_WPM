/*******************************************************
 *
 *	Wave Propagation Method
 *
 *
 *	author: Lucas Gaudelet
 *
 */


// Miscellaneous input/output and standard stuff
//#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <string>

// cuda libraries and utility functions
#include <chCommandLine.h>
#include <cuda_runtime.h>
#include <chTimer.hpp>

// Math libraries
#include <cmath>
#include <algorithm>
#include <complex>

// custom functions
#include <wpm.hh>
#include <kernel.h>

/*********************************
 *	constants
 *********************************/
 
const static int DEFAULT_X_SIZE = 	128;
const static int DEFAULT_Z_FACTOR = 	3;
const static int DEFAULT_INPUT =	0;
const static int DEFAULT_WAVE =		1;
const static int DEFAULT_SYSTEM =	3;
const static int DEFAULT_KERNEL =	0;
const static int DEFAULT_FRESNEL =	0;
const static int DEFAULT_BLOCK_SIZE = 	32;

using namespace std;



/*********************************
 *	utility functions prototypes
 *********************************/

int nextpow2(double x);
void convert(cpx* src, cufftDoubleComplex* dest, int n);
template<typename T> void check(T* array, string s);
template<typename T> void print_array(T* array, int nrow, int ncol);
template<typename T> void store_array(T* array, int nrow, int ncol,	string filename);
void store_array2(cufftDoubleComplex* array, int nrow, int ncol, string filename);
void store_results(string filename, int kernel, int blockSize, int nz, int nx,
	double kernelTime, double cpuTime=-1, double err=-1);
double compare(cufftDoubleComplex* gpu, cpx* cpu, int nz, int nx);
void printHelp(char* argv);



/**********************************
 *	main
 *********************************/
 
int main(int argc, char* argv[]) {

	bool showHelp = (chCommandLineGetBool("h", argc, argv))?
		true:chCommandLineGetBool("help", argc, argv);
	if (showHelp) {
		printHelp(argv[0]);
	exit(0);
	}

	bool verbose = (chCommandLineGetBool("v", argc, argv))?
		true:chCommandLineGetBool("verbose", argc, argv);

	if(!verbose)	cout << endl << "Init..." << endl;
	
	// wavelength and sampling
	double lambda = 1.3e-6;
	double f_samp = 16/lambda;

	/* grid parameters */
	int x_size = -1;
	chCommandLineGet<int>(&x_size, "x", argc, argv);
	chCommandLineGet<int>(&x_size, "x-size", argc, argv);
	x_size = (x_size!=-1)? x_size:DEFAULT_X_SIZE;

	double az, ax, scale, factor;
	switch(x_size) {
		case 128: {	// nx = 128
			scale = 8;	ax = scale*lambda;	break;
		}
		case 256: {	// nx = 256
			scale = 16;	ax = scale*lambda;	break;
		}
		case 512: {	// nx = 512
			scale = 32;	ax = scale*lambda;	break;
		}
		case 1024: {	// nx = 512
			scale = 64;	ax = scale*lambda;	break;
		}
		default: {
			cout << "Error - invalid x size" << endl;
			exit(-1);
		}
	}

	int z_factor = -1;
	chCommandLineGet<int>(&z_factor, "z", argc, argv);
	chCommandLineGet<int>(&z_factor, "z-size", argc, argv);
	z_factor = (z_factor!=-1)? z_factor:DEFAULT_Z_FACTOR;
	az = z_factor*scale*lambda;

	/* space Z (split step) */
	if(!verbose)	cout << "\tSpace Z...\t\t" << flush;
	double dz = lambda/8;
	int nz = static_cast<int>(az/dz);

	double* Z = static_cast<double*>(malloc(nz*sizeof(double)));
	check(Z, "Z");	init_Z(Z, nz, dz);
	if(!verbose)	cout << "done - nz=" << nz << endl;
	
	/* space X */
	if(!verbose)	cout << "\tSpace X...\t\t" << flush;
	double dx = 1/f_samp;
	int nx = nextpow2( ax/dx );

	double* X = static_cast<double*>(malloc(nx*sizeof(double)));
	check(X, "X");	init_X(X, nx, dx);
	if(!verbose)	cout << "done - nx=" << nx << endl;

	/* spatial frequency KX */
	if(!verbose)	cout << "\tSpatial frequency...\t" << flush;
	double dfx = 1/ax;
	double fx_max = dfx*(nx/2-1);
	double dtheta = asin(dfx*lambda)*180/PI;
	double dkx = 2*PI*dfx;
	if(!verbose)	cout << "done" << endl;

	cpx* KX = static_cast<cpx*>(malloc(nx*sizeof(cpx)));
	//double* FX = static_cast<double*>(malloc(nx*sizeof(double)));
	init_KX(KX, nx, dkx);	//init_FX(FX, nx, dfx);

	/* twodimensional grid */
	if(!verbose)	cout << "\tGrid...\t\t\t" << flush;
	double* XZ = static_cast<double*>(malloc(nz*nx*sizeof(double)));
	double* ZX = static_cast<double*>(malloc(nz*nx*sizeof(double)));
	check(XZ, "XZ"); check(ZX, "ZX");	meshgrid(XZ, ZX, X, nx, Z, nz);
	if(!verbose)	cout << "done" << endl;

	/* system */
	if(!verbose)	cout << "\tSystem...\t\t" << flush;
	int system = -1;
	chCommandLineGet<int>(&system, "s", argc, argv);
	chCommandLineGet<int>(&system, "system", argc, argv);
	system = (system!=-1)? system:DEFAULT_SYSTEM;
	
	cpx* N = static_cast<cpx*>(malloc(nz*nx*sizeof(cpx)));
	check(N, "N");	init_N(system, N, nz, nx, lambda, dx, dz, ax);
	if(!verbose)	cout << "done" << endl;

	/* test input parameters */
	if(!verbose)	cout << "\tInput parameters...\t" << flush;
	int input = -1;
	chCommandLineGet<int>(&input, "i", argc, argv);
	chCommandLineGet<int>(&input, "input", argc, argv);
	input = (input!=-1)? input:DEFAULT_INPUT;

	double theta_deg, f_sig;
	init_input(input, &theta_deg, &f_sig, lambda);
	if(!verbose)	cout << "done" << endl;

	/* wave */
	if(!verbose)	cout << "\tWave...\t\t\t" << flush;
	int wave = -1;
	chCommandLineGet<int>(&wave, "w", argc, argv);
	chCommandLineGet<int>(&wave, "wave", argc, argv);
	wave = (wave!=-1)? wave:DEFAULT_WAVE;	
	
	double A = 1;
	double k0 = 2*PI/lambda;
	cpx* E = static_cast<cpx*>(malloc(nz*nx*sizeof(cpx)));
	check(E, "E");	init_E(wave, E, XZ, ZX, nz, nx, lambda, A, theta_deg);
	if(!verbose)	cout << "done" << endl << endl;
	

	/* gpu wpm */
	ChTimer H2DTimer, kernelTimer, D2HTimer;

	if(!verbose) {
		cout << "Applying gpu WPM..." << endl;
		cout << "\tMemory allocation...\t" << flush;
	}
	
	//	host memory
	cufftDoubleComplex* h_E = static_cast<cufftDoubleComplex*>(
		malloc(nz*nx*sizeof(cufftDoubleComplex)));
	cufftDoubleComplex* h_N = static_cast<cufftDoubleComplex*>(
		malloc(nz*nx*sizeof(cufftDoubleComplex)));
	cufftDoubleComplex* h_KX = static_cast<cufftDoubleComplex*>(
		malloc(nz*nx*sizeof(cufftDoubleComplex)));
	
	//	device memory
	cufftDoubleComplex *d_E, *d_N, *d_KX; 
	double* d_XZ;

	cudaError_t err_E = cudaMalloc(&d_E,  nz*nx*sizeof(cufftDoubleComplex));
	cudaError_t err_N = cudaMalloc(&d_N,  nz*nx*sizeof(cufftDoubleComplex));
	cudaError_t err_KX = cudaMalloc(&d_KX, nx*sizeof(cufftDoubleComplex));
	cudaError_t err_XZ = cudaMalloc(&d_XZ, nz*nx*sizeof(double));
	
	if( d_E==NULL || d_N==NULL || d_KX==NULL || d_XZ==NULL ) {
		cout << "Error - GPU memory alloc" << endl;
		if(err_E != cudaSuccess) {
			cout << "\t\tError " << err_E << " on E alloc : " 
				<< cudaGetErrorString(err_E)  << endl;
		}
		if(err_N != cudaSuccess) {
			cout << "\t\tError " << err_N << " on N alloc : "
				<< cudaGetErrorString(err_N)  << endl;
		}
		if(err_KX != cudaSuccess) {
			cout << "\t\tError " << err_KX << " on KX alloc : "
				<< cudaGetErrorString(err_KX)  << endl;
		}
		if(err_XZ != cudaSuccess) {
			cout << "\t\tError " << err_XZ << " on XZ alloc : "
				<< cudaGetErrorString(err_XZ)  << endl;
		}
		exit(-1);
	}
	else if( d_E==NULL || d_N==NULL || d_KX==NULL || d_XZ==NULL ) {
		cout << "Error - cpu memory alloc" << endl;
		exit(-1);
	}
	else { if(!verbose) cout << "done" << endl; }

	if(!verbose)	cout << "\tCopy data H2D...\t" << flush;
	H2DTimer.start();

	//	convert type for host memory
	convert(E, h_E, nz*nx);
	convert(N, h_N, nz*nx);
	convert(KX, h_KX, nx);

	//	copy H2D
	cudaMemcpy(d_E, h_E, static_cast<size_t>(
		nz*nx*sizeof(cufftDoubleComplex)), cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, h_N, static_cast<size_t>(
		nz*nx*sizeof(cufftDoubleComplex)), cudaMemcpyHostToDevice);
	cudaMemcpy(d_KX, h_KX, static_cast<size_t>(
		nx*sizeof(cufftDoubleComplex)), cudaMemcpyHostToDevice);
	cudaMemcpy(d_XZ, XZ, static_cast<size_t>(
		nz*nx*sizeof(double)), cudaMemcpyHostToDevice);

	H2DTimer.stop();
	if(!verbose) {
		cout << "done - " << 1e3 * H2DTimer.getTime() << "ms" << endl;
		cout << "\tKernel call...\t\t" << flush;
	}

	//	kernel call
	int kernel = -1;
	chCommandLineGet<int>(&kernel, "k", argc, argv);
	chCommandLineGet<int>(&kernel, "kernel", argc, argv);
	kernel = (kernel!=-1)? kernel:DEFAULT_KERNEL;

	int fresnel = -1;
	chCommandLineGet<int>(&fresnel, "f", argc, argv);
	chCommandLineGet<int>(&fresnel, "fresnel-coef", argc, argv);
	fresnel = (fresnel!=-1)? fresnel:DEFAULT_FRESNEL;

	int blockSize = -1;
	chCommandLineGet<int>(&blockSize, "b", argc, argv);
	chCommandLineGet<int>(&blockSize, "block-size", argc, argv);
	blockSize = (blockSize!=-1)? blockSize:DEFAULT_BLOCK_SIZE;
	if( blockSize > 1024 ) {
		cout << "Error - Block size is too large" << endl;
		exit(-1);
	}
	
	switch(kernel) {
		case 0: {
			kernelTimer.start();
			naiveKernelWrapper(d_E, d_N, d_KX, d_XZ, 
				k0, dz, nz, nx, blockSize, fresnel);
			kernelTimer.stop();
			break;
		}
		case 1: {
			kernelTimer.start();
			shMemKernelWrapper(d_E, d_N, d_KX, d_XZ, 
				k0, dz, nz, nx, blockSize, fresnel);
			kernelTimer.stop();
			break;
		}
	}

	//	check errors
	cudaDeviceSynchronize();
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		cout << "\033[31m***ERROR*** " << cudaError << " - " 
			<< cudaGetErrorString(cudaError) << "***\033[0m" << endl;
		return -1;
	}
	else { if(!verbose) cout << "done - " 
		<< 1e3 * kernelTimer.getTime() << "ms" << endl; }

	if(!verbose)	cout << "\tCopy data D2H...\t" << flush;

	//	copy D2H
	D2HTimer.start();
	cudaMemcpy(h_E, d_E, static_cast<size_t>(
		nz*nx*sizeof(cufftDoubleComplex)), cudaMemcpyDeviceToHost);
	D2HTimer.stop();

	if(!verbose) cout << "done - " << 1e3 * D2HTimer.getTime() << "ms" << endl << endl;

	store_array2(h_E, nz, nx, "E_gpu.csv");

	/* cpu wpm */
	ChTimer cpuTimer;
	string filename;
	bool compare_cpu = (chCommandLineGetBool("c", argc, argv))?
		true:chCommandLineGetBool("compare-cpu", argc, argv);
	bool store = (chCommandLineGetBool("r", argc, argv))?
		true:chCommandLineGetBool("store-results", argc, argv);

	if(compare_cpu) {
		if(!verbose)	cout << "Applying cpu WPM ...\t\t" << flush;
		cpuTimer.start();
		wpm(E, N, KX, XZ, k0, dz, nz, nx, fresnel);
		cpuTimer.stop();
		if(!verbose)	cout << "done - "  << 1e3 * cpuTimer.getTime() << "ms" << endl;
		store_array(E, nz, nx, "E_cpu.csv");

		double err = compare(h_E, E, nz, nx);
		if(!verbose)    cout << "\tMax Error: " << err << endl << endl;

		if(store) {
			chCommandLineGet<string>(&filename, "r", argc, argv);
			chCommandLineGet<string>(&filename, "store-results", argc, argv);
			store_results(filename, kernel, blockSize, nz, nx, 
				1e3*kernelTimer.getTime(), 1e3*cpuTimer.getTime(), err); 
		}
	}
	else if(store) {
		chCommandLineGet<string>(&filename, "r", argc, argv);
		chCommandLineGet<string>(&filename, "store-results", argc, argv);
		store_results(filename, kernel, blockSize, nz, nx, 1e3*kernelTimer.getTime()); 
	}

	/* free */
	free(Z);	free(X);
	free(KX);	//free(FX);
	free(XZ);	free(ZX);
	free(N);	free(E);

	free(h_E);	free(h_N);
	free(h_KX);

	cudaFree(d_E);	cudaFree(d_N);
	cudaFree(d_KX);	cudaFree(d_XZ);

	return 0;
}



/**********************************
 *	utility functions
 *********************************/

int nextpow2(double x) {
	int result = 1;
	while (result < x) result <<= 1;
	return result;
}

template<typename T>
void print_array(T* array, int nrow, int ncol) {
	for(int i=0; i<nrow; i++) {
		for(int j=0; j<ncol; j++) {
			cout << array[i*ncol+j] << "\t";
		}
		cout << endl;
	}
	cout << endl;
}

template<typename T>
void store_array(T* array, int nrow, int ncol, string filename) {
	ofstream file;
	file.open(filename.c_str());
	for(int i=0; i<nrow; i++) {
		for(int j=0; j<ncol; j++) {
			file << array[i*ncol+j] << "\t";
		}
		file << endl;
	}
	file.close();
}

template<typename T>
void check(T* array, string s) {
	if(array==NULL) {
		cout << "Error - Memory allocation " << s << endl;
	}
}

void convert(cpx* src, cufftDoubleComplex* dest, int n) {
	for(int i=0; i<n; i++) {
		dest[i].x = src[i].real();
		dest[i].y = src[i].imag();
	}
}

void store_array2(cufftDoubleComplex* array, int nrow, int ncol, string filename) {
	ofstream file;
	file.open(filename.c_str());
	for(int i=0; i<nrow; i++) {
		for(int j=0; j<ncol; j++) {
			file << "(" << array[i*ncol+j].x << "," 
				<< array[i*ncol+j].y << ")\t";
		}
		file << endl;
	}
	file.close();
}

void store_results(string filename, int kernel, int blockSize, int nz, int nx,
	double kernelTime, double cpuTime, double err) {

	ofstream file;

	struct stat buffer;
	// if the file does not exist write column names
	if( stat(filename.c_str(), &buffer) ) {
		file.open(filename.c_str());
		file << "kernel_id;block_size;nz;nx;kernel_time;cpu_time;err" << endl;
	}
	else{	file.open(filename.c_str(), ios::app);	}
	
	file << kernel << ";" << blockSize << ";" << nz << ";" << nx << 
		";" << kernelTime << ";" << cpuTime << ";" << err << endl;
	file.close();

}

double compare(cufftDoubleComplex* gpu, cpx* cpu, int nz, int nx) {
	double max_err = 0, err;
	cpx tmp;

	for(int i=0; i<nz; i++) {
		for(int j=0; j<nx; j++) {
			tmp = cpx(gpu[i*nx+j].x, gpu[i*nx+j].y);
//			err = (cpu[i*nx+j]!=cpx(0,0))? 
//				abs( (tmp-cpu[i*nx+j]) / cpu[i*nx+j] ):abs(tmp-cpu[i*nx+j]);
			err = abs( (tmp-cpu[i*nx+j]) / cpu[i*nx+j] );
			max_err = (max_err<err)? err:max_err;
		}
	}

	return max_err;
}

void printHelp(char* argv) {
	cout << "Help:" << endl
		<< "  Usage: " << endl
		<< "  " << argv << " [options][-p <problem-size> ]" << endl << endl

		<< "  -x|--x-size" << endl
		<< "      discrete size of the aperture along x axis, must be a power of 2 and less than 1024" << endl
		<< "  -z|--z-factor" << endl
		<< "      this value will linearly modify the discrete size of z aperture in function of the x aperture" << endl
		<< "  -k|--kernel" << endl
		<< "      kernel to be used" << endl
		<< "  -b|--block-size" << endl
		<< "      size of thread blocks" << endl
		<< "  -f|--fresnel-coef" << endl
		<< "      fresnel coefficient to be used" << endl
		<< " (default)0: TE" << endl
		<< "          1: TM" << endl
		<< "  -c|--compare-cpu" << endl
		<< "      compare results with cpu" << endl
		<< "  -v|--verbose" << endl
		<< "      disables text display" << endl
		<< "  -s|--system" << endl
		<< "      specifies which medium the light goes through :" << endl
		<< "          0: homogeneous system" << endl
		<< "          1: single boundary" << endl
		<< "          2: waveguide" << endl
		<< " (default)3: lens" << endl
		<< "          4: GRIN lens" << endl
		<< "          5: slit" << endl
		<< "  -i|--input" << endl
		<< "      seven different sets of input parameters (from 0 to 6)" << endl
		<< "  -w|--wave" << endl
		<< "      specifies the nature of the incident wave" << endl
		<< "          0: plane wave" << endl
		<< " (default)1: gaussian wave" << endl
		<< endl;
}

