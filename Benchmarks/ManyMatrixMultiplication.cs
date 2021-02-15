using System;
using System.Diagnostics;
using System.Linq;
using Alea;

#if DOUBLE_PRECISION
    using Real = System.Double;
#else
    using Real = System.Single;
#endif

namespace AleaSandbox.Benchmarks
{
    internal static class ManyMatrixMultiplication
    {
        public static void Initialise(
            Real[] cube,
            Real[] matrix,
            int m,
            int n)
        {
            var rand = new Random(1);

            for (int i = 0; i != m * n * n; ++i)
            {
                cube[i] = (Real)rand.NextDouble();
            }

            for (int i = 0; i != n * n; ++i)
            {
                matrix[i] = (Real)rand.NextDouble();
            }
        }

        public static void Managed(Real[] result, Real[] cube, Real[] matrix, int m, int n)
        {
            var timer = Stopwatch.StartNew();

            var offset = 0;

            for (int i = 0; i != m; ++i)
            {
                Multiply(result, cube, matrix, offset, n);

                offset += n * n;
            }

            PrintPerformance(timer, "ManyMatrixMultiplication.Managed", n * m, n, n);
        }

        private static void Multiply(Real[] result, Real[] left, Real[] right, int offset, int n)
        {
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    Real value = 0;

                    for (int k = 0; k < n; ++k)
                    {
                        value += left[offset + i + k * n] * right[k + j * n];
                    }

                    result[offset + i + j * n] = value;
                }
            }
        }

        public static unsafe void Alea(Gpu gpu, Real[] result, Real[] left, Real[] right, int m, int n)
        {
            using (var cudaResult = gpu.AllocateDevice(result))
            using (var cudaLeft = gpu.AllocateDevice(left))
            using (var cudaRight = gpu.AllocateDevice(right))
            {
                var alphas = new Real[] { 1 };
                var betas = new Real[] { 0 };
                var results = Enumerable.Range(0, m).Select(i => cudaResult.Ptr.Handle + i * n * n * sizeof(Real)).ToArray();
                var lefts = Enumerable.Range(0, m).Select(i => cudaLeft.Ptr.Handle + i * n * n * sizeof(Real)).ToArray();
                var rights = Enumerable.Range(0, m).Select(i => cudaRight.Ptr.Handle).ToArray();

                using (var cudaResults = gpu.AllocateDevice(results))
                using (var cudaLefts = gpu.AllocateDevice(lefts))
                using (var cudaRights = gpu.AllocateDevice(rights))
                {
                    fixed (Real* pAlphas = alphas)
                    fixed (Real* pBetas = betas)
                    {
                        var timer = Stopwatch.StartNew();

                        var blas = global::Alea.cuBLAS.Blas.Get(gpu);
                        var lAlphas = pAlphas;
                        var lBetas = pBetas;

                        gpu.EvalAction(() =>
                            global::Alea.cuBLAS.Interop.cublasSafeCall(
#if DOUBLE_PRECISION
                            global::Alea.cuBLAS.Interop.cublasDgemmBatched(
#else
                                global::Alea.cuBLAS.Interop.cublasSgemmBatched(
#endif
                                    blas.Handle,
                                    global::Alea.cuBLAS.Operation.N,
                                    global::Alea.cuBLAS.Operation.N,
                                    n,
                                    n,
                                    n,
                                    lAlphas,
                                    // ReSharper disable AccessToDisposedClosure
                                    cudaLefts.Ptr.Handle,
                                    n,
                                    cudaRights.Ptr.Handle,
                                    n,
                                    lBetas,
                                    cudaResults.Ptr.Handle,
                                    // ReSharper restore AccessToDisposedClosure
                                    n,
                                    m)));

                        gpu.Synchronize();

                        PrintPerformance(timer, "MatrixMultiplication.cuBLAS", m * n, n, n);

                        Gpu.Copy(cudaResult, result);
                    }
                }
            }
        }

        private static void PrintPerformance(Stopwatch timer, string prefix, int numOps, int m, int n)
        {
            var elapsed = timer.Elapsed;

            Console.WriteLine("{0}: {1} ms, {2:F1} GB/s",
                prefix,
                elapsed.TotalMilliseconds,
                (Real)m * n * sizeof(Real) * numOps / elapsed.TotalSeconds / (1024 * 1024 * 1024));
        }
    }
}
