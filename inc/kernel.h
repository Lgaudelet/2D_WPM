#include <cufft.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>


/* kernels and wrappers */

// naive kernel
__global__ void naiveKernelTE(cufftDoubleComplex* E, cufftDoubleComplex* N,
	cufftDoubleComplex* KX, cufftDoubleComplex* S, 
	double* X, double k0, double dz, int nx, int i);

__global__ void naiveKernelTM(cufftDoubleComplex* E, cufftDoubleComplex* N,
	cufftDoubleComplex* KX, cufftDoubleComplex* S, 
	double* X, double k0, double dz, int nx, int i);
	
void naiveKernelWrapper(cufftDoubleComplex* E, cufftDoubleComplex* N,
	cufftDoubleComplex* KX,	double* X, double k0, double dz, int nz, int nx,
	int blockSize, int fresnel);
	
// shared memory kernel
__global__ void shMemKernelTE(cufftDoubleComplex* E, cufftDoubleComplex* N,
	cufftDoubleComplex* KX, cufftDoubleComplex* S, 
	double* X, double k0, double dz, int nx, int i);

//__global__ void shMemKernelTM(cufftDoubleComplex* E, cufftDoubleComplex* N,
//	cufftDoubleComplex* KX, cufftDoubleComplex* S, 
//	double* X, double k0, double dz, int nx, int i);
	
void shMemKernelWrapper(cufftDoubleComplex* E, cufftDoubleComplex* N,
	cufftDoubleComplex* KX,	double* X, double k0, double dz, int nz, int nx,
	int blockSize, int fresnel);



/* complex math functions */
__host__ __device__ static __inline__
	cuDoubleComplex cuCsub(double  x, cuDoubleComplex y);
__host__ __device__ static __inline__
	cuDoubleComplex cuCmul(cuDoubleComplex x, double y);
__host__ __device__ static __inline__
	cuDoubleComplex cuCdiv(cuDoubleComplex x, double y);
__host__ __device__ static __inline__
	cuDoubleComplex cuCexp(cuDoubleComplex x);
__host__ __device__ static __inline__
	cuDoubleComplex cuCsqrt(cuDoubleComplex x);

/* overload */
__host__ __device__ static __inline__ 
	cuDoubleComplex operator-(double x, cuDoubleComplex y);
__host__ __device__ static __inline__
	cuDoubleComplex operator*(cuDoubleComplex x, double y);
__host__ __device__ static __inline__
	cuDoubleComplex operator/(cuDoubleComplex x, double y);
__host__ __device__ static __inline__
	cuDoubleComplex operator+(cuDoubleComplex x, cuDoubleComplex y);
__host__ __device__ static __inline__
	cuDoubleComplex operator*(cuDoubleComplex x, cuDoubleComplex y);
__host__ __device__ static __inline__
	cuDoubleComplex operator/(cuDoubleComplex x, cuDoubleComplex y);


