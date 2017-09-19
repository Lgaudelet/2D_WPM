#include <kernel.h>
#include <iostream>
#include <stdio.h>


/*
 * Naive Kernel
 */

__global__ void naiveKernelTE(cufftDoubleComplex* E, cufftDoubleComplex* N,
	cufftDoubleComplex* KX, cufftDoubleComplex* S, double* X, 
	double k0, double dz, int nx, int i) {
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j<nx) {

		// load constants and variables
		cufftDoubleComplex li = make_cuDoubleComplex(0,1);
		cufftDoubleComplex result = make_cuDoubleComplex(0,0);
		
		cufftDoubleComplex n = N[i*nx+j];
		cufftDoubleComplex next_n = N[(i+1)*nx+j];

		// preliminary computing
		double nk0 = n.x * k0;
		double kk0 = n.y * k0;
		double next_nk0 = next_n.x * k0;

		cufftDoubleComplex KZ, next_KZ, F, tmp;

		// loop over frequencies
		for(int kj=0; kj<nx; kj++) {

			// compute KZ and next_KZ
			tmp = KX[kj];	KZ = cuCsqrt(nk0*nk0-tmp*tmp);
			tmp = KX[kj];	next_KZ = cuCsqrt(next_nk0*next_nk0-tmp*tmp);

			if( KZ.y < 1e-6 && KZ.x!=0 ) {	// exclude evanescent waves

				// Fresnel coefficient
				F = KZ*2/(KZ+next_KZ);
				
				// compute E
				int sign = (kj%2)? -1:1;
				result = result + 
						( F * (S[kj]/nx) * (sign*std::exp(-kk0*dz)) * 
						cuCexp(li*(KX[kj]*X[j]+KZ*dz)) );
			}
		}
		E[(i+1)*nx+j] = result;
	}
}

__global__ void naiveKernelTM(cufftDoubleComplex* E, cufftDoubleComplex* N,
	cufftDoubleComplex* KX, cufftDoubleComplex* S, double* X, 
	double k0, double dz, int nx, int i) {
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(j<nx) {

		// load constants and variables
		cufftDoubleComplex li = make_cuDoubleComplex(0,1);
		cufftDoubleComplex result = make_cuDoubleComplex(0,0);
		
		cufftDoubleComplex n = N[i*nx+j];
		cufftDoubleComplex next_n = N[(i+1)*nx+j];

		// preliminary computing
		double nk0 = n.x * k0;
		double kk0 = n.y * k0;
		double next_nk0 = next_n.x * k0;

		cufftDoubleComplex KZ, next_KZ, F, tmp;

		// loop over frequencies
		for(int kj=0; kj<nx; kj++) {

			// compute KZ and next_KZ
			tmp = KX[kj];	KZ = cuCsqrt(nk0*nk0-tmp*tmp);
			tmp = KX[kj];	next_KZ = cuCsqrt(next_nk0*next_nk0-tmp*tmp);

			if( KZ.y < 1e-6 && KZ.x!=0 ) {	// exclude evanescent waves

				// Fresnel coefficient
				F = n*next_n*KZ*2 / (next_n*next_n*KZ+n*n*next_KZ);
				
				// compute E
				int sign = (kj%2)? -1:1;
				result = result + 
						( F * (S[kj]/nx) * (sign*std::exp(-kk0*dz)) * 
						cuCexp(li*(KX[kj]*X[j]+KZ*dz)) );
			}
		}
		E[(i+1)*nx+j] = result;
	}
}

void naiveKernelWrapper(cufftDoubleComplex* E, cufftDoubleComplex* N, 
	cufftDoubleComplex* KX, double* X, double k0, double dz, int nz, int nx,
	int blockSize, int fresnel) {
	
	cufftHandle plan;
	cufftPlan1d(&plan, nx, CUFFT_Z2Z, 1);

	cufftDoubleComplex* S;
	cudaMalloc(&S, nx*sizeof(cufftDoubleComplex));

	dim3 block_dim(blockSize, 1, 1);
	dim3 grid_dim(ceil(static_cast<double>(nx)/static_cast<double>(blockSize)), 1, 1);

//	std::cout << grid_dim.x << "*" << blockSize << "\t" << std::flush;

	switch(fresnel) {
	case 0:
		for(int i=0; i<nz-1; i++) {
			// fftt
			cufftExecZ2Z(plan, E+i*nx, S, CUFFT_FORWARD);
			// compute E
			naiveKernelTE<<<grid_dim, block_dim>>>(E, N, KX, S, X, k0, dz, nx, i);
		}
		break;
	
	case 1:
		for(int i=0; i<nz-1; i++) {
			// fftt
			cufftExecZ2Z(plan, E+i*nx, S, CUFFT_FORWARD);
			// compute E
			naiveKernelTM<<<grid_dim, block_dim>>>(E, N, KX, S, X, k0, dz, nx, i);
		}
		break;
		
	default:
		std::cout << "Error - Unknown Fresnel Coefficient" << std::endl;
		exit(-1);
	}
}



/*
 * Shared Memory Kernel
 */

__global__ void shMemKernelTE(cufftDoubleComplex* E, cufftDoubleComplex* N,
	cufftDoubleComplex* KX, cufftDoubleComplex* S, double* X, 
	double k0, double dz, int nx, int i) {
	
	// initialite constants and variables
	extern __shared__ cufftDoubleComplex shMem[];
	cufftDoubleComplex* kx = shMem;
	cufftDoubleComplex* s = &shMem[blockDim.x];

	int nb = nx/blockDim.x;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	cufftDoubleComplex li, n, next_n;
	cufftDoubleComplex tmp, result;
	cufftDoubleComplex KZ, next_KZ, F;
	double nk0, kk0, next_nk0;

	// preliminary computing
	if(j<nx) {
		li = make_cuDoubleComplex(0,1);
		result = make_cuDoubleComplex(0,0);
		
		n = N[i*nx+j];
		next_n = N[(i+1)*nx+j];
		
		nk0 = n.x * k0;
		kk0 = n.y * k0;
		next_nk0 = next_n.x * k0;
	}

	for(int l=0; l<nb; l++) {
		// collaboratively load kx
		if(l*blockDim.x + threadIdx.x < nx ) {
			kx[threadIdx.x] = KX[l*blockDim.x + threadIdx.x];
			s[threadIdx.x] = S[l*blockDim.x + threadIdx.x];
		}
		__syncthreads();

		if(j<nx) {
			for(int ll=0; ll<blockDim.x; ll++) {

				// compute KZ and next_KZ
				tmp = kx[ll];	KZ = cuCsqrt(nk0*nk0-tmp*tmp);
				tmp = kx[ll];	next_KZ = cuCsqrt(next_nk0*next_nk0-tmp*tmp);

				if( KZ.y < 1e-6 && KZ.x!=0 ) {	// exclude evanescent waves

					// Fresnel coefficient
					F = KZ*2/(KZ+next_KZ);

					// compute E
					int sign = ( (l*blockDim.x+ll)%2 )? -1:1;
					result = result + 
						( F * (s[ll]/nx) * (sign*std::exp(-kk0*dz)) * 
						cuCexp(li*(kx[ll]*X[j]+KZ*dz)) );
				}	
			}
		}
		__syncthreads();
	}
	E[(i+1)*nx+j] = result;
}

__global__ void shMemKernelTM(cufftDoubleComplex* E, cufftDoubleComplex* N,
	cufftDoubleComplex* KX, cufftDoubleComplex* S, double* X, 
	double k0, double dz, int nx, int i) {
	
	// initialite constants and variables
	extern __shared__ cufftDoubleComplex shMem[];
	cufftDoubleComplex* kx = shMem;
	cufftDoubleComplex* s = &shMem[blockDim.x];

	int nb = nx/blockDim.x;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	cufftDoubleComplex li;
	cufftDoubleComplex tmp, result, n, next_n;
	cufftDoubleComplex KZ, next_KZ, F;
	double nk0, kk0, next_nk0;

	// preliminary computing
	if(j<nx) {
		li = make_cuDoubleComplex(0,1);
		result = make_cuDoubleComplex(0,0);
		
		n = N[i*nx+j];
		next_n = N[(i+1)*nx+j];
		
		nk0 = n.x * k0;
		kk0 = n.y * k0;
		next_nk0 = next_n.x * k0;
	}

	for(int l=0; l<nb; l++) {
		// collaboratively load kx
		if(l*blockDim.x + threadIdx.x < nx ) {
			kx[threadIdx.x] = KX[l*blockDim.x + threadIdx.x];
			s[threadIdx.x] = S[l*blockDim.x + threadIdx.x];
		}
		__syncthreads();

		if(j<nx) {
			for(int ll=0; ll<blockDim.x; ll++) {

				// compute KZ and next_KZ
				tmp = kx[ll];	KZ = cuCsqrt(nk0*nk0-tmp*tmp);
				tmp = kx[ll];	next_KZ = cuCsqrt(next_nk0*next_nk0-tmp*tmp);

				if( KZ.y < 1e-6 && KZ.x!=0 ) {	// exclude evanescent waves

					// Fresnel coefficient
					F = n*next_n*KZ*2 / (next_n*next_n*KZ+n*n*next_KZ);

					// compute E
					int sign = ( (l*blockDim.x+ll)%2 )? -1:1;
					result = result + 
						( F * (s[ll]/nx) * (sign*std::exp(-kk0*dz)) * 
						cuCexp(li*(kx[ll]*X[j]+KZ*dz)) );
				}	
			}
		}
		__syncthreads();
	}
	E[(i+1)*nx+j] = result;
}

void shMemKernelWrapper(cufftDoubleComplex* E, cufftDoubleComplex* N, 
	cufftDoubleComplex* KX, double* X, double k0, double dz, int nz, int nx,
	int blockSize, int fresnel) {
	
	cufftHandle plan;
	cufftPlan1d(&plan, nx, CUFFT_Z2Z, 1);

	cufftDoubleComplex* S;
	cudaMalloc(&S, nx*sizeof(cufftDoubleComplex));

	dim3 block_dim(blockSize, 1, 1);
	dim3 grid_dim(ceil(static_cast<double>(nx)/static_cast<double>(blockSize)), 1, 1);
	int sharedMemorySize = 2 * blockSize * sizeof(cufftDoubleComplex); 

//	std::cout << grid_dim.x << "*" << blockSize << " "
//		<< sharedMemorySize << "\t" << std::flush;

	switch(fresnel) {
	case 0:
		for(int i=0; i<nz-1; i++) {
			// fftt
			cufftExecZ2Z(plan, E+i*nx, S, CUFFT_FORWARD);
			// compute E
			shMemKernelTE<<<grid_dim, block_dim, sharedMemorySize>>>(
				E, N, KX, S, X, k0, dz, nx, i);
		}
		break;
	
	case 1:
		for(int i=0; i<nz-1; i++) {
			// fftt
			cufftExecZ2Z(plan, E+i*nx, S, CUFFT_FORWARD);
			// compute E
			shMemKernelTM<<<grid_dim, block_dim, sharedMemorySize>>>(
				E, N, KX, S, X, k0, dz, nx, i);
		}
		break;
		
	default:
		std::cout << "Error - Unknown Fresnel Coefficient " << fresnel << std::endl;
		exit(-1);
	}
}


/* complex math functions */
__host__ __device__ static __inline__ cuDoubleComplex cuCsub(double  x, cuDoubleComplex y) {
	return make_cuDoubleComplex(x - cuCreal(y), -cuCimag(y));
}

__host__ __device__ static __inline__ cuDoubleComplex cuCmul(cuDoubleComplex x, double y) {
	return make_cuDoubleComplex(cuCreal(x) * y, cuCimag(x) * y);
}

__host__ __device__ static __inline__ cuDoubleComplex cuCdiv(cuDoubleComplex x, double y) {
	return make_cuDoubleComplex(cuCreal(x) / y, cuCimag(x) / y);
}

__host__ __device__ static __inline__ cuDoubleComplex cuCexp(cuDoubleComplex x) {
	double factor = std::exp(cuCreal(x));
	return make_cuDoubleComplex(factor * std::cos(cuCimag(x)), factor * std::sin(cuCimag(x)));
}

__host__ __device__ static __inline__ cuDoubleComplex cuCsqrt(cuDoubleComplex x) {
	if(x.x==0 && x.y==0) return make_cuDoubleComplex(0, 0);
	
	double radius = cuCabs(x);
	double cosA = x.x / radius;
	cuDoubleComplex out;
	out.x = std::sqrt(radius * (cosA + 1.0) / 2.0);
	out.y = std::sqrt(radius * (1.0 - cosA) / 2.0);
	// signbit should be false if x.y is negative
	if (signbit(x.y))
		out.y *= -1.0;

	return out;
}

/* overload */

__host__ __device__ static __inline__ cuDoubleComplex operator-(double x, cuDoubleComplex y) {
	return cuCsub(x,y);
}

__host__ __device__ static __inline__ cuDoubleComplex operator*(cuDoubleComplex x, double y) {
	return cuCmul(x,y);
}

__host__ __device__ static __inline__ cuDoubleComplex operator/(cuDoubleComplex x, double y) {
	return cuCdiv(x,y);
}

__host__ __device__ static __inline__ cuDoubleComplex operator+(cuDoubleComplex x, cuDoubleComplex y) {
	return cuCadd(x,y);
}

__host__ __device__ static __inline__ cuDoubleComplex operator*(cuDoubleComplex x, cuDoubleComplex y) {
	return cuCmul(x,y);
}

__host__ __device__ static __inline__ cuDoubleComplex operator/(cuDoubleComplex x, cuDoubleComplex y) {
	return cuCdiv(x,y);
}


