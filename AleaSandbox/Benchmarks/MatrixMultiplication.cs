using System;
using System.Diagnostics;
using Alea;

#if DOUBLE_PRECISION
    using Real = System.Double;
#else
    using Real = System.Single;
#endif

namespace AleaSandbox.Benchmarks
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

        public static void Cuda(Real[] result, Real[] left, Real[] right, int n)
        {
            var gpu = Gpu.Default;

            using (var cudaResult = gpu.AllocateDevice(result))
            using (var cudaLeft = gpu.AllocateDevice(left))
            using (var cudaRight = gpu.AllocateDevice(right))
            {
                var timer = Stopwatch.StartNew();

                Alea.cuBLAS.Blas.Get(gpu).Gemm(
                    Alea.cuBLAS.Operation.N,
                    Alea.cuBLAS.Operation.N,
                    n, n, n,
                    1, cudaLeft.Ptr, n,
                    cudaRight.Ptr, n, 0,
                    cudaResult.Ptr, n);

                gpu.Synchronize();

                PrintPerformance(timer, "MatrixMultiplication.cuBLAS", n, n, n);

                Gpu.Copy(cudaResult, result);
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
