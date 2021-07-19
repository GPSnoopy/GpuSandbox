#include "squared_distance.hpp"

#include "cuda_utils.hpp"
#include "utils.hpp"
#include <device_launch_parameters.h>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;

namespace
{
	typedef void(*OptimisedKernel)(Real*, const Real*, int, int);

	void cuda_optimised_impl(Array& mSquaredDistances, const Array& mCoordinates, const int c, const int n, const int sharedRealsPerThread, const char* const name, const OptimisedKernel kernel)
	{
		Real* pSquaredDistances;
		Real* pCoordinates;

		CudaCheck(cudaMalloc(&pSquaredDistances, mSquaredDistances.size() * sizeof(Real)));
		CudaCheck(cudaMalloc(&pCoordinates, mCoordinates.size() * sizeof(Real)));

		CudaCheck(cudaMemcpy(pCoordinates, mCoordinates.data(), mCoordinates.size() * sizeof(Real), cudaMemcpyKind::cudaMemcpyHostToDevice));
		CudaCheck(cudaDeviceSynchronize());

		const auto begin = steady_clock::now();

		int blockSize = 128;
		int gridSize = div_up(n, blockSize);
		int sharedSize = sharedRealsPerThread * blockSize * sizeof(Real);

		kernel<<<dim3(gridSize, gridSize), dim3(blockSize), sharedSize>>> (pSquaredDistances, pCoordinates, c, n);

		CudaCheck(cudaDeviceSynchronize());

		print_performance(begin, steady_clock::now(), name, n, c, n);

		CudaCheck(cudaMemcpy(mSquaredDistances.data(), pSquaredDistances, mSquaredDistances.size() * sizeof(Real), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		CudaCheck(cudaFree(pCoordinates));
		CudaCheck(cudaFree(pSquaredDistances));
	}

	__global__ void cuda_kernel(
		Real* const mSquaredDistances,
		const Real* const mCoordinates,
		const int c, const int n)
	{
		const int j = blockIdx.x * blockDim.x + threadIdx.x;

		if (j < n)
		{
			for (int i = 0; i < n; ++i)
			{
				Real dist = 0;

				for (int k = 0; k != c; ++k)
				{
					const Real coord1 = mCoordinates[k * n + i];
					const Real coord2 = mCoordinates[k * n + j];
					const Real diff = coord1 - coord2;

					dist += diff * diff;
				}

				mSquaredDistances[i * n + j] = dist;
			}
		}
	}

	__global__ void cuda_kernel_shared_memory(
		Real* const mSquaredDistances,
		const Real* const mCoordinates,
		const int c, const int n)
	{
		// We've got shared memory of two vector of K dimensions for B points:
		//
		//      var coordI = __shared__ new Real[k*blockDim.x];
		//      var coordJ = __shared__ new Real[k*blockDim.x];
		//
		// We fill in these two vectors with the coordinates of the I points and J points.
		// Afterwards, the current block will compute the euclidean distances between all 
		// the I points and J points, producing a square matrix [B, B].
		//
		// This optimisation means that when producing the square matrix, the I and J points
		// coordinates are only read once.
		//
		// This optimisation works well if K is small enough. Otherwise the shared memory is
		// too small and not enough blocks get schedule per SM.

		extern __shared__  Real shared[];

		const int bI = blockIdx.y * blockDim.x;
		const int bJ = blockIdx.x * blockDim.x;

		const int coordI = 0;
		const int coordJ = c * blockDim.x;

		for (int k = 0; k != c; ++k)
		{
			if (bI + threadIdx.x < n)
			{
				shared[coordI + k * blockDim.x + threadIdx.x] = mCoordinates[k * n + bI + threadIdx.x];
			}

			if (bJ + threadIdx.x < n)
			{
				shared[coordJ + k * blockDim.x + threadIdx.x] = mCoordinates[k * n + bJ + threadIdx.x];
			}
		}

		__syncthreads();

		if (bJ + threadIdx.x < n)
		{
			for (int i = 0; i < blockDim.x && bI + i < n; ++i)
			{
				Real dist = 0;

				for (int k = 0; k != c; ++k)
				{
					const Real coord1 = shared[coordI + k * blockDim.x + i]; //mCoordinates[k * x + i];
					const Real coord2 = shared[coordJ + k * blockDim.x + threadIdx.x]; //mCoordinates[k * x + j];
					const Real diff = coord1 - coord2;

					dist += diff * diff;
				}

				mSquaredDistances[(bI + i) * n + (bJ + threadIdx.x)] = dist;
			}
		}
	}

	__global__ void cuda_kernel_float2(
		Real* const mSquaredDistances,
		const Real* const mCoordinates,
		const int c, const int n)
	{
		// Same as cuda_kernel_shared_memory, but one thread does two element in one by using float2 reads.

		extern __shared__  Real shared[];

		const auto coordinatesI = shared;
		const auto coordinatesJ = shared + c * blockDim.x;

		const int bI = blockIdx.y * blockDim.x;
		const int bJ = blockIdx.x * blockDim.x;

		for (int k = 0; k != c; ++k)
		{
			if (bI + threadIdx.x < n)
			{
				coordinatesI[k * blockDim.x + threadIdx.x] = mCoordinates[k * n + bI + threadIdx.x];
			}

			if (bJ + threadIdx.x < n)
			{
				coordinatesJ[k * blockDim.x + threadIdx.x] = mCoordinates[k * n + bJ + threadIdx.x];
			}
		}

		__syncthreads();

		const int line = threadIdx.x / (blockDim.x / 2);
		const int tid = threadIdx.x % (blockDim.x / 2);

		if (bJ + tid * 2 < n)
		{
			const auto coordinatesJ2 = (float2*)coordinatesJ;

			for (int i = line; i < blockDim.x && bI + i < n; i += 2)
			{
				Real dist0 = 0;
				Real dist1 = 0;

				for (int k = 0; k != c; ++k)
				{
					const Real coord1 = coordinatesI[k * blockDim.x + i];
					const Real2 coord2 = coordinatesJ2[(k * blockDim.x / 2) + tid];
					const Real2 diff = { coord1 - coord2.x, coord1 - coord2.y };

					dist0 += diff.x * diff.x;
					dist1 += diff.y * diff.y;
				}

				mSquaredDistances[(bI + i) * n + (bJ + 2 * tid + 0)] = dist0;
				mSquaredDistances[(bI + i) * n + (bJ + 2 * tid + 1)] = dist1;
			}
		}
	}

	template <int c>
	__global__ void cuda_kernel_constants(
		Real* const mSquaredDistances,
		const Real* const mCoordinates,
		const int, const int n)
	{
		// Same as cuda_kernel_float2, but with loop unrolling and index calculations taken outside the inner loop.

		extern __shared__  Real shared[];

		const auto coordinatesI = shared;
		const auto coordinatesJ = shared + c * blockDim.x;

		const int bI = blockIdx.y * blockDim.x;
		const int bJ = blockIdx.x * blockDim.x;

		for (int k = 0; k != c; ++k)
		{
			if (bI + threadIdx.x < n)
			{
				coordinatesI[k * blockDim.x + threadIdx.x] = mCoordinates[k * n + bI + threadIdx.x];
			}

			if (bJ + threadIdx.x < n)
			{
				coordinatesJ[k * blockDim.x + threadIdx.x] = mCoordinates[k * n + bJ + threadIdx.x];
			}
		}

		__syncthreads();

		const int line = threadIdx.x / (blockDim.x / 2);
		const int tid = threadIdx.x % (blockDim.x / 2);

		if (bJ + tid * 2 < n)
		{
			const auto coordinatesJ2 = (float2*)coordinatesJ;

			for (int i = line; i < blockDim.x && bI + i < n; i += 2)
			{
				Real dist0 = 0;
				Real dist1 = 0;

				for (int k = 0; k != c; ++k)
				{
					const Real coord1 = coordinatesI[k * blockDim.x + i];
					const Real2 coord2 = coordinatesJ2[(k * blockDim.x / 2) + tid];
					const Real2 diff = { coord1 - coord2.x, coord1 - coord2.y };

					dist0 += diff.x * diff.x;
					dist1 += diff.y * diff.y;
				}

				mSquaredDistances[(bI + i) * n + (bJ + 2 * tid + 0)] = dist0;
				mSquaredDistances[(bI + i) * n + (bJ + 2 * tid + 1)] = dist1;
			}
		}
	}

	template
	<unsigned c, unsigned tileDimY, unsigned halfBlockMask = 0x007f>
	__global__ 
	//__launch_bounds__(256, 8)
	void cuda_kernel_local_memory(
		Real* const mSquaredDistances,
		const Real* const __restrict__ mCoordinates,
		const unsigned /*c*/, const unsigned n)
	{
		__shared__  Real coordinatesI[c* tileDimY];
		Real2 coordinatesJ[c];

		const unsigned bI = blockIdx.y * tileDimY;
		const unsigned bJ = blockIdx.x * blockDim.x;
		const unsigned tid = threadIdx.x & halfBlockMask;
		const unsigned imax = min(tileDimY, n - bI);
		const bool copyI = threadIdx.x < imax, copyJ = bJ + tid * 2 < n;
		const Real*  __restrict__ mcI = &mCoordinates[bI + threadIdx.x];
		const Real2* __restrict__ mcJ = &((const float2*)&mCoordinates[bJ])[tid];

		for (unsigned k = 0; k != c; ++k)
		{
			if (copyI)
			{
				coordinatesI[k * tileDimY + threadIdx.x] = *mcI;
				mcI += n;
				//coordinatesI[k * tileDimY + threadIdx.x] = mCoordinates[k * n + bI + threadIdx.x];
				//coordinatesI[k * tileDimY + threadIdx.x] = __ldg(&mCoordinates[k * n + bI + threadIdx.x]);
			}
			if (copyJ)
			{
				coordinatesJ[k] = *mcJ;
				mcJ += n >> 1;
				//coordinatesJ[k] = ((const float2*)&mCoordinates[k * n + bJ])[tid];
				//coordinatesJ[k] = __ldg(&((const float2*)&mCoordinates[k * n + bJ])[tid]);
			}
		}

		__syncthreads();

		unsigned line = threadIdx.x & (blockDim.x>>1) ? 1 : 0;
		Real2* msd = (Real2*)&mSquaredDistances[(bI + line) * n + (bJ + 2 * tid)];

		if (copyJ)
		{
			for (unsigned i = line; i < imax; i += 2)
			{
				Real2 dist = { 0, 0 };
				int io = i;

				for (unsigned k = 0; k != c; ++k)
				{
					Real coord1 = coordinatesI[io];
					Real2 coord2 = coordinatesJ[k];
					Real2 diff = { coord1 - coord2.x, coord1 - coord2.y };

					dist.x += diff.x * diff.x;
					dist.y += diff.y * diff.y;

					io += tileDimY;
				}

				*msd = dist;
				msd += n;
			}
		}
	}
}

void SquaredDistance::initialise(Array& mSquaredDistances, Array& mCoordinates, const int c, const int n)
{
	default_random_engine generator(42);
	uniform_real_distribution<Real> rand(0, 1);

	for (int i = 0; i != n; ++i)
		for (int j = 0; j != n; ++j)
			mSquaredDistances[i * n + j] = 0;

	for (int i = 0; i != c * n; ++i)
	{
		mCoordinates[i] = rand(generator);
	}
}

void SquaredDistance::native(Array& mSquaredDistances, const Array& mCoordinates, const int c, const int n)
{
	const auto begin = steady_clock::now();

	for (int i = 0; i != n; ++i)
	{
		for (int j = 0; j != n; ++j)
		{
			Real dist = 0;

			for (int k = 0; k != c; ++k)
			{
				const Real coord1 = mCoordinates[k * n + i];
				const Real coord2 = mCoordinates[k * n + j];
				const Real diff = coord1 - coord2;

				dist += diff * diff;
			}

			mSquaredDistances[i * n + j] = dist;
		}
	}

	print_performance(begin, steady_clock::now(), "SquaredDistance.Native", n, c, n);
}

void SquaredDistance::cuda(Array& mSquaredDistances, const Array& mCoordinates, const int c, const int n)
{
	Real* pSquaredDistances;
	Real* pCoordinates;

	CudaCheck(cudaMalloc(&pSquaredDistances, mSquaredDistances.size()*sizeof(Real)));
	CudaCheck(cudaMalloc(&pCoordinates, mCoordinates.size()*sizeof(Real)));

	CudaCheck(cudaMemcpy(pCoordinates, mCoordinates.data(), mCoordinates.size()*sizeof(Real), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CudaCheck(cudaDeviceSynchronize());

	const auto begin = steady_clock::now();

	cuda_kernel<<<div_up(n*n, 128), 128>>>(pSquaredDistances, pCoordinates, c, n);

	CudaCheck(cudaDeviceSynchronize());

	print_performance(begin, steady_clock::now(), "SquaredDistance.Cuda", n, c, n);

	CudaCheck(cudaMemcpy(mSquaredDistances.data(), pSquaredDistances, mSquaredDistances.size()*sizeof(Real), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	CudaCheck(cudaFree(pCoordinates));
	CudaCheck(cudaFree(pSquaredDistances));
}

void SquaredDistance::cuda_shared_memory(Array& mSquaredDistances, const Array& mCoordinates, const int c, const int n)
{
	cuda_optimised_impl(mSquaredDistances, mCoordinates, c, n, 2*c, "SquaredDistance.Cuda.Float2", cuda_kernel_shared_memory);
}

void SquaredDistance::cuda_float2(Array& mSquaredDistances, const Array& mCoordinates, const int c, const int n)
{
	cuda_optimised_impl(mSquaredDistances, mCoordinates, c, n, 2*c, "SquaredDistance.Cuda.SharedMemory", cuda_kernel_float2);
}

void SquaredDistance::cuda_constant(Array& mSquaredDistances, const Array& mCoordinates, const int c, const int n)
{
	cuda_optimised_impl(mSquaredDistances, mCoordinates, c, n, 2*c, "SquaredDistance.Cuda.Constants", cuda_kernel_constants<20>);
}

void SquaredDistance::cuda_local_memory(Array& mSquaredDistances, const Array& mCoordinates, const int c, const int n)
{
	CudaCheck(cudaThreadSetCacheConfig(cudaFuncCachePreferShared));

	Real* pSquaredDistances;
	Real* pCoordinates;
	const int blockSize = 256; // must be power of 2
	const unsigned halfBlockMask = (blockSize >> 1) - 1;

	CudaCheck(cudaMalloc(&pSquaredDistances, mSquaredDistances.size() * sizeof(Real)));
	CudaCheck(cudaMalloc(&pCoordinates, (mCoordinates.size()) * sizeof(Real))); // Add padding to remove need for bound checking

	CudaCheck(cudaMemcpy(pCoordinates, mCoordinates.data(), mCoordinates.size() * sizeof(Real), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CudaCheck(cudaDeviceSynchronize());

	const int tileDimY = 76;
	int gridSizeX = div_up(n, blockSize);
	int gridSizeY = div_up(n, tileDimY);
	int sharedSize = 0;
	
	const auto begin = steady_clock::now();

	cuda_kernel_local_memory<20, tileDimY, halfBlockMask><<<dim3(gridSizeX, gridSizeY), dim3(blockSize), sharedSize>>>(pSquaredDistances, pCoordinates, c, n);

	CudaCheck(cudaDeviceSynchronize());

	print_performance(begin, steady_clock::now(), "SquaredDistance.Cuda.LocalMemory", n, c, n);

	CudaCheck(cudaMemcpy(mSquaredDistances.data(), pSquaredDistances, mSquaredDistances.size() * sizeof(Real), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	CudaCheck(cudaFree(pCoordinates));
	CudaCheck(cudaFree(pSquaredDistances));
}