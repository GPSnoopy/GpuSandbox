#include "add_vector.hpp"
#include "cuda_utils.hpp"
#include "utils.hpp"
#include <device_launch_parameters.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

namespace
{
	__global__ void add_kernel(Real* const matrix, const Real* const vector, const int m, const int n)
	{
		const int i = blockIdx.y * blockDim.y + threadIdx.y;
		const int j = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < m && j < n)
		{
			matrix[i * n + j] += vector[j];
		}
	}
}

void AddVector::initialise(Array& matrix, Array& vector, const int m, const int n)
{
	int counter = 0;

	for (int i = 0; i != m; ++i)
		for (int j = 0; j != n; ++j)
			matrix[i * n + j] = static_cast<Real>(counter++);

	for (int j = 0; j != n; ++j)
		vector[j] = static_cast<Real>(j);
}

void AddVector::native(Array& matrix, const Array& vector, const int m, const int n)
{
	const auto begin = steady_clock::now();

	for (int i = 0; i != m; ++i)
		for (int j = 0; j != n; ++j)
			matrix[i * n + j] += vector[j];

	print_performance(begin, steady_clock::now(), "AddVector.Native", 3, m, n);
}

void AddVector::cuda(Array& matrix, const Array& vector, const int m, const int n)
{
	Real* pMatrix;
	Real* pVector;

	CudaCheck(cudaMalloc(&pMatrix, m*n*sizeof(Real)));
	CudaCheck(cudaMalloc(&pVector, n*sizeof(Real)));

	CudaCheck(cudaMemcpy(pMatrix, matrix.data(), matrix.size()*sizeof(Real), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CudaCheck(cudaMemcpy(pVector, vector.data(), vector.size()*sizeof(Real), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CudaCheck(cudaDeviceSynchronize());

	const auto begin = steady_clock::now();

	int gridSizeX = div_up(n, 32);
	int gridSizeY = div_up(m, 8);
	add_kernel<<<dim3(gridSizeX, gridSizeY), dim3(32, 8)>>>(pMatrix, pVector, m, n);

	CudaCheck(cudaDeviceSynchronize());

	print_performance(begin, steady_clock::now(), "AddVector.Cuda", 3, m, n);

	CudaCheck(cudaMemcpy(matrix.data(), pMatrix, matrix.size()*sizeof(Real), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	CudaCheck(cudaFree(pVector));
	CudaCheck(cudaFree(pMatrix));
}
