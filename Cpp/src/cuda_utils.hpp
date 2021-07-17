#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define CudaCheck(val) check_cuda_error ( (val), #val, __FILE__, __LINE__ )

[[noreturn]]
void throw_cuda_error(cudaError_t result, char const* func, const char* file, int line);

inline void check_cuda_error(const cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		throw_cuda_error(result, func, file, line);
	}
}

inline size_t div_up(const size_t value, const size_t divisor)
{
	return (value + divisor - 1) / divisor;
}
