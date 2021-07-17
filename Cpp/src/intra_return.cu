#include "intra_return.hpp"
#include "cuda_utils.hpp"
#include "utils.hpp"
#include <device_launch_parameters.h>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;

namespace
{
	__host__ inline Real mask(Real value)
	{
		return value * 0 == 0 ? value : 0;
	}

	__device__ inline float mask_cuda(float value)
	{
		return isfinite(value) != 0 ? value : 0;
	}

	__global__ void cuda_kernel(
		Real* const mIntraReturn, 
		const Real* const vClose, 
		const Real* const vIsAlive, 
		const Real* const vIsValidDay,
		const int m, const int n)
	{
		const int i = blockIdx.y * blockDim.y + threadIdx.y;
		const int j = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < m && j < n)
		{
			const Real close = vClose[j];
			const Real isAlive = vIsAlive[j];
			const Real isValidDay = vIsValidDay[j];
			const Real current = mIntraReturn[i * n + j];

			mIntraReturn[i * n + j] = close > 0 ? mask_cuda((current - close) / close) * isAlive * isValidDay : 0;
		}
	}
}

void IntraReturn::initialise(Array& mIntraReturn, Array& vClose, Array& vIsAlive, Array& vIsValidDay, const int m, const int n)
{
	default_random_engine generator(42);
	uniform_real_distribution<Real> rand(0, 1);

	for (int i = 0; i != m * n; ++i)
	{
		mIntraReturn[i] = rand(generator);
	}

	for (int i = 0; i != n; ++i)
	{
		vClose[i] = rand(generator);
		vIsAlive[i] = static_cast<Real>(rand(generator) < 0.9 ? 1 : 0);
		vIsValidDay[i] = static_cast<Real>(rand(generator) < 0.9 ? 1 : 0);
	}
}

void IntraReturn::native(Array& mIntraReturn, const Array& vClose, const Array& vIsAlive, const Array& vIsValidDay, const int m, const int n)
{
	const auto begin = steady_clock::now();

	for (int i = 0; i != m; ++i)
	{
		for (int j = 0; j != n; ++j)
		{
			const Real current = mIntraReturn[i * n + j];
			const Real close = vClose[j];
			const Real isAlive = vIsAlive[j];
			const Real isValidDay = vIsValidDay[j];

			mIntraReturn[i * n + j] = close > 0 ? mask((current - close) / close) * isAlive * isValidDay : 0;
		}
	}

	print_performance(begin, steady_clock::now(), "IntraReturn.Native", 5, m, n);
}

void IntraReturn::cuda(Array& mIntraReturn, const Array& vClose, const Array& vIsAlive, const Array& vIsValidDay, const int m, const int n)
{
	Real* pIntraReturn;
	Real* pClose;
	Real* pIsAlive;
	Real* pIsValidDay;

	CudaCheck(cudaMalloc(&pIntraReturn, m*n*sizeof(Real)));
	CudaCheck(cudaMalloc(&pClose, n*sizeof(Real)));
	CudaCheck(cudaMalloc(&pIsAlive, n*sizeof(Real)));
	CudaCheck(cudaMalloc(&pIsValidDay, n*sizeof(Real)));

	CudaCheck(cudaMemcpy(pIntraReturn, mIntraReturn.data(), mIntraReturn.size()*sizeof(Real), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CudaCheck(cudaMemcpy(pClose, vClose.data(), vClose.size()*sizeof(Real), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CudaCheck(cudaMemcpy(pIsAlive, vIsAlive.data(), vIsAlive.size()*sizeof(Real), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CudaCheck(cudaMemcpy(pIsValidDay, vIsValidDay.data(), vIsValidDay.size()*sizeof(Real), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CudaCheck(cudaDeviceSynchronize());

	const auto begin = steady_clock::now();

	int gridSizeX = div_up(n, 32);
	int gridSizeY = div_up(m, 8);
	cuda_kernel<<<dim3(gridSizeX, gridSizeY), dim3(32, 8)>>>(pIntraReturn, pClose, pIsAlive, pIsValidDay, m, n);

	CudaCheck(cudaDeviceSynchronize());

	print_performance(begin, steady_clock::now(), "IntraReturn.Cuda", 5, m, n);

	CudaCheck(cudaMemcpy(mIntraReturn.data(), pIntraReturn, mIntraReturn.size()*sizeof(Real), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	CudaCheck(cudaFree(pIsValidDay));
	CudaCheck(cudaFree(pIsAlive));
	CudaCheck(cudaFree(pClose));
	CudaCheck(cudaFree(pIntraReturn));
}
