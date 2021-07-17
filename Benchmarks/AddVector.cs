using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;

#if DOUBLE_PRECISION
    using Real = System.Double;
#else
    using Real = System.Single;
#endif

namespace GpuSandbox.Benchmarks
{
    internal static class AddVector
    {
        public static void Initialise(Real[] matrix, Real[] vector, int m, int n)
        {
            var counter = 0;

            for (int i = 0; i != m; ++i)
                for (int j = 0; j != n; ++j)
                    matrix[i * n + j] = counter++;

            for (int j = 0; j != n; ++j)
                vector[j] = j;
        }

        public static void Managed(Real[] matrix, Real[] vector, int m, int n)
        {
            var timer = Stopwatch.StartNew();

            for (int i = 0; i != m; ++i)
                for (int j = 0; j != n; ++j)
                    matrix[i * n + j] += vector[j];

            Util.PrintPerformance(timer, "AddVector.Managed", 3, m, n);
        }

        public static void Gpu(Accelerator gpu, Real[] matrix, Real[] vector, int m, int n)
        {
            using (var cudaMatrix = gpu.Allocate1D(matrix))
            using (var cudaVector = gpu.Allocate1D(vector))
            {
                var timer = Stopwatch.StartNew();

                var gridSizeX = Util.DivUp(n, 32);
                var gridSizeY = Util.DivUp(m, 8);
                var lp = ((gridSizeX, gridSizeY, 1), (32, 8));

                gpu.Launch(Kernel, gpu.DefaultStream, lp, cudaMatrix.View, cudaVector.View, m, n);

                gpu.Synchronize();
                Util.PrintPerformance(timer, "AddVector.Gpu", 3, m, n);

                cudaMatrix.CopyToCPU(matrix);
            }
        }

        private static void Kernel(ArrayView1D<Real, Stride1D.Dense> matrix, ArrayView1D<Real, Stride1D.Dense> vector, int m, int n)
        {
            var i = Grid.IdxY * Group.DimY + Group.IdxY;
            var j = Grid.IdxX * Group.DimX + Group.IdxX;

            if (i < m && j < n)
            {
                matrix[i * n + j] += vector[j];
            }
        }
    }
}
