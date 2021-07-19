
#include "add_vector.hpp"
#include "intra_return.hpp"
#include "squared_distance.hpp"
#include "cuda_utils.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;

namespace
{
	const int Loops = 10;

	void application();
	void initialize_cuda(int selected_device);
	void run_add_vector();
	void run_intra_return();
	void run_squared_distance();
	void run(int loops, const function<void()>& baseInitialise, const function<void()>& testInitialise, const function<void()>& compare, const function<void()>& baseRun, const vector<function<void()>>& testRuns);
	void assert_are_equal(const Array& left, const Array& right, int m, int n);
}

int main(int argc, char* argv[])
{
	try
	{
		application();
		return EXIT_SUCCESS;
	}

	catch (const std::exception& error)
	{
		std::cerr << "ERROR: " << typeid(error).name() << ": " << error.what() << std::endl;
	}

	catch (...)
	{
		std::cerr << "ERROR: unhandled exception" << std::endl;
	}

	return EXIT_FAILURE;
}

namespace
{

	void application()
	{
		initialize_cuda(0);

		run_add_vector();
		run_intra_return();
		run_squared_distance();
	}

	void initialize_cuda(const int selected_device)
	{
		int runtimeVersion;
		int driverVersion;

		CudaCheck(cudaDriverGetVersion(&driverVersion));
		CudaCheck(cudaRuntimeGetVersion(&runtimeVersion));

		cout << "CUDA Driver Version: " << driverVersion / 1000 << "." << driverVersion % 1000 << endl;
		cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." << runtimeVersion % 1000 << endl;
		cout << endl;
		cout << "Available CUDA devices: " << endl;

		int deviceCount = 0;
		CudaCheck(cudaGetDeviceCount(&deviceCount));

		for (int device = 0; device != deviceCount; ++device)
		{
			cudaDeviceProp prop{};
			CudaCheck(cudaGetDeviceProperties(&prop, device));

			cout << format("- Device {} [{:04x}:{:02x}:{:02x}.0] : {} {}GB{}\n",
				device, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, prop.name,
				prop.totalGlobalMem / (1024 * 1024 * 1024),
				selected_device == device ? " (Selected)" : "");
		}

		if (selected_device < 0 || selected_device >= deviceCount)
		{
			throw std::out_of_range(format("selected device {} does not exist", selected_device));
		}

		CudaCheck(cudaSetDevice(selected_device));

		cout << std::endl;
	}

	void run_add_vector()
	{
		const int m = 2 * 24 * 12;
		const int n = 2 * 25600 - 1;

		Array matrixM(m * n);
		Array matrixC(m * n);
		Array vector(n);

		run(Loops,
			[&]() { AddVector::initialise(matrixM, vector, m, n); },
			[&]() { AddVector::initialise(matrixC, vector, m, n); },
			[&]() { assert_are_equal(matrixM, matrixC, m, n); },
			[&]() { AddVector::native(matrixM, vector, m, n); },
			{ [&]() { AddVector::cuda(matrixC, vector, m, n); } });
	}

	void run_intra_return()
	{
		const int m = 2 * 24 * 12;
		const int n = 2 * 25600 - 1;

		Array matrixM(m * n);
		Array matrixC(m * n);
		Array vector1(n);
		Array vector2(n);
		Array vector3(n);

		run(Loops,
			[&]() { IntraReturn::initialise(matrixM, vector1, vector2, vector3, m, n); },
			[&]() { IntraReturn::initialise(matrixC, vector1, vector2, vector3, m, n); },
			[&]() { assert_are_equal(matrixM, matrixC, m, n); },
			[&]() { IntraReturn::native(matrixM, vector1, vector2, vector3, m, n); },
			{ [&]() { IntraReturn::cuda(matrixC, vector1, vector2, vector3, m, n); } });
	}

	void run_squared_distance()
	{
		const int c = 20;
		const int x = 2 * 10000;

		Array matrixM(x * x);
		Array matrixC(x * x);
		Array coordinates(c * x);

		run(Loops,
			[&]() { SquaredDistance::initialise(matrixM, coordinates, c, x); },
			[&]() { SquaredDistance::initialise(matrixC, coordinates, c, x); },
			[&]() { assert_are_equal(matrixM, matrixC, x, x); },
			[&]() { SquaredDistance::native(matrixM, coordinates, c, x); },
		{
			[&]() { SquaredDistance::cuda(matrixC, coordinates, c, x); },
			[&]() { SquaredDistance::cuda_shared_memory(matrixC, coordinates, c, x); },
			[&]() { SquaredDistance::cuda_float2(matrixC, coordinates, c, x); },
			[&]() { SquaredDistance::cuda_constant(matrixC, coordinates, c, x); },
			[&]() { SquaredDistance::cuda_local_memory(matrixC, coordinates, c, x); }
		});
	}

	void run(
		int loops, 
		const function<void()>& baseInitialise, 
		const function<void()>& testInitialise, 
		const function<void()>& compare, 
		const function<void()>& baseRun, 
		const vector<function<void()>>& testRuns)
	{
		//for (int i = 0; i != loops; ++i)
		{
			baseInitialise();
			baseRun();
		}

		printf("\n");

		for (auto& run : testRuns)
		{
			for (int i = 0; i != loops; ++i)
			{
				testInitialise();
				run();
				compare();
			}

			printf("\n");
		}
	}

	void assert_are_equal(const Array& left, const Array& right, const int m, const int n)
	{
		const Real e = sizeof(Real) == 4 ? static_cast<Real>(1e-5f) : static_cast<Real>(1e-12);

		for (int i = 0; i != m; ++i)
		{
			for (int j = 0; j != n; ++j)
			{
				const Real a = abs(left[i * n + j]);
				const Real b = abs(right[i * n + j]);
				const Real d = abs(left[i * n + j] - right[i * n + j]);

				if (d > e && d / std::min(a + b, numeric_limits<Real>::max()) > e)
				{			
					throw runtime_error(format("{} != {} [{}, {}]", left[i * n + j], right[i * n + j], i, j));
				}
			}
		}
	}
	
}
