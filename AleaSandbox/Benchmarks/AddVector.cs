using System.Diagnostics;
using Alea;
using Alea.CSharp;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

#if DOUBLE_PRECISION
    using Real = System.Double;
#else
    using Real = System.Single;
#endif

namespace AleaSandbox.Benchmarks
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

        public static void Alea(Gpu gpu, Real[] matrix, Real[] vector, int m, int n)
        {
            using (var cudaMatrix = gpu.AllocateDevice(matrix))
            using (var cudaVector = gpu.AllocateDevice(vector))
            {
                var timer = Stopwatch.StartNew();

                var gridSizeX = Util.DivUp(n, 32);
                var gridSizeY = Util.DivUp(m, 8);
                var lp = new LaunchParam(new dim3(gridSizeX, gridSizeY), new dim3(32, 8));

                gpu.Launch(AleaKernel, lp, cudaMatrix.Ptr, cudaVector.Ptr, m, n);

                gpu.Synchronize();
                Util.PrintPerformance(timer, "AddVector.Alea", 3, m, n);

                Gpu.Copy(cudaMatrix, matrix);
            }
        }

        public static void IlGpu(CudaAccelerator gpu, Real[] matrix, Real[] vector, int m, int n)
        {
            using (var cudaMatrix = gpu.Allocate(matrix))
            using (var cudaVector = gpu.Allocate(vector))
            {
                var timer = Stopwatch.StartNew();

                var gridSizeX = Util.DivUp(n, 32);
                var gridSizeY = Util.DivUp(m, 8);
                var lp = ((gridSizeX, gridSizeY, 1), (32, 8));

                var kernel = gpu.LoadStreamKernel<ArrayView<Real>, ArrayView<Real>, int, int>(IlGpuKernel);
                kernel(lp, cudaMatrix.View, cudaVector.View, m, n);

                gpu.Synchronize();
                Util.PrintPerformance(timer, "AddVector.IlGpu", 3, m, n);

                cudaMatrix.CopyTo(matrix, 0, 0, matrix.Length);
            }
        }

        private static void AleaKernel(deviceptr<Real> matrix, deviceptr<Real> vector, int m, int n)
        {
            var i = blockIdx.y * blockDim.y + threadIdx.y;
            var j = blockIdx.x * blockDim.x + threadIdx.x;

            if (i < m && j < n)
            {
                matrix[i * n + j] += vector[j];
            }
        }
        
        private static void IlGpuKernel(ArrayView<Real> matrix, ArrayView<Real> vector, int m, int n)
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
