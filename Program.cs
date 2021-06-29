//#define USE_ALEA

using System;
using System.Linq;
using Alea;
using GpuSandbox.Benchmarks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
#if DOUBLE_PRECISION
    using Real = System.Double;
#else
    using Real = System.Single;
#endif

namespace GpuSandbox
{
    internal sealed class Program
    {
        public static void Main()
        {
            try
            {
#if USE_ALEA
                Alea.Device.Default.Print();
                using var aleaGpu = Gpu.Default;
#else
                var aleaGpu = (Gpu) null;
#endif
                Console.WriteLine();

                using var context = ILGPU.Context.Create(b => b
                    .Optimize(OptimizationLevel.O2)
                    .OpenCL());

                foreach (var device in context)
                {
                    device.PrintInformation(Console.Out);
                    Console.WriteLine();
                }

                using var ilGpu = context.Devices.Any(d => d.AcceleratorType == AcceleratorType.Cuda) 
                    ? (Accelerator) context.CreateCudaAccelerator(0)
                    : context.CreateCLAccelerator(0);

                RunAddVector(aleaGpu, ilGpu);
                RunIntraReturn(aleaGpu, ilGpu);
                RunSquaredDistance(aleaGpu, ilGpu);

                if (ilGpu is CudaAccelerator ilGpuCuda)
                {
                    RunMatrixMultiplication(aleaGpu, ilGpuCuda);
                    RunManyMatrixMultiplication(aleaGpu, ilGpuCuda);
                }
            }

            catch (Exception exception)
            {
                var color = Console.ForegroundColor;

                Console.ForegroundColor = ConsoleColor.Red;
                Console.Error.WriteLine("ERROR: " + exception);
                Console.ForegroundColor = color;
            }
        }

        private static void RunAddVector(Gpu aleaGpu, Accelerator ilGpu)
        {
            const int m = 2 * 24 * 12;
            const int n = 2 * 25600 - 1;

            var matrixM = new Real[m * n];
            var matrixC = new Real[m * n];
            var vector = new Real[n];

            Benchmark.Run(Loops,
                () => AddVector.Initialise(matrixM, vector, m, n),
                () => AddVector.Initialise(matrixC, vector, m, n),
                () => AssertAreEqual(matrixM, matrixC, m, n),
                () => AddVector.Managed(matrixM, vector, m, n),
#if USE_ALEA
                () => AddVector.Alea(aleaGpu, matrixC, vector, m, n),
#endif
                () => AddVector.IlGpu(ilGpu, matrixC, vector, m, n));
        }

        private static void RunIntraReturn(Gpu aleaGpu, Accelerator ilGpu)
        {
            const int m = 2 * 24 * 12;
            const int n = 2 * 25600 - 1;

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
#if USE_ALEA
                () => IntraReturn.Alea(aleaGpu, matrixC, vector1, vector2, vector3, m, n),
#endif
                () => IntraReturn.IlGpu(ilGpu, matrixC, vector1, vector2, vector3, m, n));
        }

        private static void RunSquaredDistance(Gpu aleaGpu, Accelerator ilGpu)
        {
            const int c = 20;
            const int x = 2 * 10000;

            var matrixM = new Real[x * x];
            var matrixC = new Real[x * x];
            var coordinates = new Real[c * x];

            Benchmark.Run(Loops,
                () => SquaredDistance.Initialise(matrixM, coordinates, c, x),
                () => SquaredDistance.Initialise(matrixC, coordinates, c, x),
                () => AssertAreEqual(matrixM, matrixC, x, x),
                () => SquaredDistance.Managed(matrixM, coordinates, c, x),
#if USE_ALEA              
                () => SquaredDistance.Alea(aleaGpu, matrixC, coordinates, c, x),
#endif
                () => SquaredDistance.IlGpu(ilGpu, matrixC, coordinates, c, x),
#if USE_ALEA               
                () => SquaredDistance.AleaSharedMemory(aleaGpu, matrixC, coordinates, c, x),
#endif
                () => SquaredDistance.IlGpuSharedMemory(ilGpu, matrixC, coordinates, c, x),
#if USE_ALEA              
                () => SquaredDistance.AleaFloat2(aleaGpu, matrixC, coordinates, c, x),
#endif
                () => SquaredDistance.IlGpuFloat2(ilGpu, matrixC, coordinates, c, x),
#if USE_ALEA             
                () => SquaredDistance.AleaConstants(aleaGpu, matrixC, coordinates, c, x),
#endif
                () => SquaredDistance.IlGpuConstants(ilGpu, matrixC, coordinates, c, x),
#if USE_ALEA            
                () => SquaredDistance.AleaLocalMemory(aleaGpu, matrixC, coordinates, c, x),
#endif
                () => SquaredDistance.IlGpuLocalMemory(ilGpu, matrixC, coordinates, c, x));
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

        private static void RunManyMatrixMultiplication(Gpu aleaGpu, CudaAccelerator ilGpu)
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
                () => ManyMatrixMultiplication.Alea(aleaGpu, resultC, left, right, m, n));
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

        private const int Loops = 10;
    }
}
