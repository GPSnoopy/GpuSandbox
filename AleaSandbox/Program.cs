using System;
using Alea;
using AleaSandbox.Benchmarks;
using ILGPU.Runtime.Cuda;

#if DOUBLE_PRECISION
    using Real = System.Double;
#else
    using Real = System.Single;
#endif

namespace AleaSandbox
{
    internal sealed class Program
    {
        public static void Main()
        {
            try
            {
                Alea.Device.Default.Print();
                Console.WriteLine();

                using var context = new ILGPU.Context();
                var aleaGpu = Alea.Gpu.Default;
                var ilGpu = new CudaAccelerator(context);

                RunAddVector(aleaGpu, ilGpu);
                RunIntraReturn(aleaGpu, ilGpu);
                RunSquaredDistance(aleaGpu, ilGpu);
                RunMatrixMultiplication(aleaGpu, ilGpu);
                RunManyMatrixMultiplication();

                // ILGPU remarks:
                // - Allocate and copy extension methods?
                // - Allocate 64-bits / long size?
                // - Access to additional CUDA intrinsics.
            }

            catch (Exception exception)
            {
                var color = Console.ForegroundColor;

                Console.ForegroundColor = ConsoleColor.Red;
                Console.Error.WriteLine("ERROR: " + exception);
                Console.ForegroundColor = color;
            }
        }

        private static void RunAddVector(Gpu aleaGpu, CudaAccelerator ilGpu)
        {
            const int m = 24 * 12;
            const int n = 25600 - 1;

            var matrixM = new Real[m * n];
            var matrixC = new Real[m * n];
            var vector = new Real[n];

            Benchmark.Run(Loops,
                () => AddVector.Initialise(matrixM, vector, m, n),
                () => AddVector.Initialise(matrixC, vector, m, n),
                () => AssertAreEqual(matrixM, matrixC, m, n),
                () => AddVector.Managed(matrixM, vector, m, n),
                () => AddVector.Alea(aleaGpu, matrixC, vector, m, n),
                () => AddVector.IlGpu(ilGpu, matrixC, vector, m, n));
        }

        private static void RunIntraReturn(Gpu aleaGpu, CudaAccelerator ilGpu)
        {
            const int m = 24 * 12;
            const int n = 25600 - 1;

            var matrixM = new Real[m * n];
            var matrixC = new Real[m * n];
            var vector1 = new Real[n];
            var vector2 = new Real[n];
            var vector3 = new Real[n];

            Benchmark.Run(Loops,
                () => IntraReturn.Initialise(matrixM, vector1, vector2, vector3, m, n),
                () => IntraReturn.Initialise(matrixC, vector1, vector2, vector3, m, n),
                () => AssertAreEqual(matrixM, matrixC, m, n),
                () => IntraReturn.Managed(matrixM, vector1, vector2, vector3, m, n),
                () => IntraReturn.Alea(aleaGpu, matrixC, vector1, vector2, vector3, m, n),
                () => IntraReturn.IlGpu(ilGpu, matrixC, vector1, vector2, vector3, m, n));
        }

        private static void RunSquaredDistance(Gpu aleaGpu, CudaAccelerator ilGpu)
        {
            const int c = 20;
            const int x = 10000;

            var matrixM = new Real[x * x];
            var matrixC = new Real[x * x];
            var coordinates = new Real[c * x];

            Benchmark.Run(Loops,
                () => SquaredDistance.Initialise(coordinates, c, x),
                () => SquaredDistance.Initialise(coordinates, c, x),
                () => AssertAreEqual(matrixM, matrixC, x, x),
                () => SquaredDistance.Managed(matrixM, coordinates, c, x),
                () => SquaredDistance.Alea(aleaGpu, matrixC, coordinates, c, x),
                () => SquaredDistance.IlGpu(ilGpu, matrixC, coordinates, c, x),
                () => SquaredDistance.AleaSharedMemory(aleaGpu, matrixC, coordinates, c, x),
                () => SquaredDistance.AleaFloat2(aleaGpu, matrixC, coordinates, c, x),
                () => SquaredDistance.AleaConstants(aleaGpu, matrixC, coordinates, c, x),
                () => SquaredDistance.AleaLocalMemory(aleaGpu, matrixC, coordinates, c, x));
        }

        private static void RunMatrixMultiplication(Gpu aleaGpu, CudaAccelerator ilGpu)
        {
            const int n = 1500 - 1;

            var resultM = new Real[n * n];
            var resultC = new Real[n * n];
            var left = new Real[n * n];
            var right = new Real[n * n];

            Benchmark.Run(Loops,
                () => MatrixMultiplication.Initialise(left, right, n),
                () => MatrixMultiplication.Initialise(left, right, n),
                () => AssertAreEqual(resultM, resultC, n, n),
                () => MatrixMultiplication.Managed(resultM, left, right, n),
                () => MatrixMultiplication.Alea(aleaGpu, resultC, left, right, n),
                () => MatrixMultiplication.IlGpu(ilGpu, resultC, left, right, n));
        }

        private static void RunManyMatrixMultiplication()
        {
            const int m = 100;
            const int n = 250 - 1;

            var resultM = new Real[m * n * n];
            var resultC = new Real[m * n * n];
            var left = new Real[m * n * n];
            var right = new Real[n * n];

            Benchmark.Run(Loops,
                () => ManyMatrixMultiplication.Initialise(left, right, m, n),
                () => ManyMatrixMultiplication.Initialise(left, right, m, n),
                () => AssertAreEqual(resultM, resultC, m * n, n),
                () => ManyMatrixMultiplication.Managed(resultM, left, right, m, n),
                () => ManyMatrixMultiplication.Cuda(resultC, left, right, m, n));
        }

        private static void AssertAreEqual(Real[] left, Real[] right, int m, int n)
        {
            var e = typeof(Real) == typeof(float) ? 1e-5 : 1e-12;

            for (int i = 0; i != m; ++i)
            {
                for (int j = 0; j != n; ++j)
                {
                    var a = Math.Abs(left[i * n + j]);
                    var b = Math.Abs(right[i * n + j]);
                    var d = Math.Abs(left[i * n + j] - right[i * n + j]);

                    if (d > e && d / Math.Min(a + b, Real.MaxValue) > e)
                        throw new Exception(left[i * n + j] + " != " + right[i * n + j] + " [" + i + ", " + j + "]");
                }
            }
        }

        private const int Loops = 5;
    }
}
