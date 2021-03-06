﻿using System;
using System.Diagnostics;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

#if DOUBLE_PRECISION
    using Real = System.Double;
#else
    using Real = System.Single;
#endif

namespace GpuSandbox.Benchmarks
{
    internal static class MatrixMultiplication
    {
        public static void Initialise(Real[] left, Real[] right, int n)
        {
            var rand = new Random(1);

            for (int i = 0; i != n * n; ++i)
            {
                left[i] = (Real) rand.NextDouble();
                right[i] = (Real) rand.NextDouble();
            }
        }

        public static void Managed(Real[] result, Real[] left, Real[] right, int n)
        {
            var timer = Stopwatch.StartNew();

            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    Real value = 0;

                    for (int k = 0; k < n; ++k)
                    {
                        value += left[i + k * n] * right[k + j * n];
                    }

                    result[i + j * n] = value;
                }
            }

            PrintPerformance(timer, "MatrixMultiplication.Managed", n, n, n);
        }

        public static void Gpu(CudaAccelerator gpu, Real[] result, Real[] left, Real[] right, int n)
        {
            using (var cudaResult = gpu.Allocate1D(result))
            using (var cudaLeft = gpu.Allocate1D(left))
            using (var cudaRight = gpu.Allocate1D(right))
            {
                using var blas = new CuBlas(gpu, CuBlasAPIVersion.V11);

                var timer = Stopwatch.StartNew();
                
                blas.Gemm(
                    CuBlasOperation.NonTranspose,
                    CuBlasOperation.NonTranspose,
                    n, n, n,
                    1, cudaLeft.View, n,
                    cudaRight.View, n, 0,
                    cudaResult.View, n);

                gpu.Synchronize();

                PrintPerformance(timer, "MatrixMultiplication.Gpu.cuBLAS", n, n, n);

                cudaResult.CopyToCPU(result);
            }
        }

        private static void PrintPerformance(Stopwatch timer, string prefix, int numOps, int m, int n)
        {
            var elapsed = timer.Elapsed;

            Console.WriteLine("{0}: {1} ms, {2:F1} GB/s",
                prefix,
                elapsed.TotalMilliseconds,
                (Real) m * n * sizeof(Real) * numOps / elapsed.TotalSeconds / (1024 * 1024 * 1024));
        }
    }
}
