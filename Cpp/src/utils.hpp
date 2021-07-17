#pragma once

#include "real.hpp"
#include <chrono>

inline int div_up(int a, int b)
{
	return (a + b - 1) / b;
}

inline void print_performance(std::chrono::time_point<std::chrono::steady_clock> begin, std::chrono::time_point<std::chrono::steady_clock> end, const char* prefix, int numOps, int m, int n)
{
	const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	const double totalMilliseconds = elapsed / (1000.0*1000.0);
	const double totalSeconds = totalMilliseconds / 1000.0;

	printf("%s: %f ms, %.1f GB/s\n",
		prefix,
		totalMilliseconds,
		(double)m * n * sizeof(Real) * numOps / (totalSeconds * 1024 * 1024 * 1024));
}
