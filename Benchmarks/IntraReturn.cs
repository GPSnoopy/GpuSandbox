using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Alea;
using Alea.CSharp;
using ILGPU;
using ILGPU.Runtime;

#if DOUBLE_PRECISION
    using Real = System.Double;
#else
    using Real = System.Single;
#endif

namespace GpuSandbox.Benchmarks
{
    internal static class IntraReturn
    {
        public static void Initialise(
            Real[] mIntraReturn,
            Real[] vClose,
            Real[] vIsAlive,
            Real[] vIsValidDay,
            int m,
            int n)
        {
            var rand = new Random(1);

            for (int i = 0; i != m * n; ++i)
            {
                mIntraReturn[i] = (Real) rand.NextDouble();
            }

            for (int i = 0; i != n; ++i)
            {
                vClose[i] = (Real) rand.NextDouble();
                vIsAlive[i] = rand.NextDouble() < 0.9 ? 1 : 0;
                vIsValidDay[i] = rand.NextDouble() < 0.9 ? 1 : 0;
            }
        }

        public static void Managed(
            Real[] mIntraReturn,
            Real[] vClose,
            Real[] vIsAlive,
            Real[] vIsValidDay,
            int m,
            int n)
        {
            var timer = Stopwatch.StartNew();

            for (int i = 0; i != m; ++i)
            {
                for (int j = 0; j != n; ++j)
                {
                    var current = mIntraReturn[i * n + j];
                    var close = vClose[j];
                    var isAlive = vIsAlive[j];
                    var isValidDay = vIsValidDay[j];

                    mIntraReturn[i * n + j] = close > 0 ? Mask((current - close) / close) * isAlive * isValidDay : 0;
                }
            }

            Util.PrintPerformance(timer, "IntraReturn.Managed", 5, m, n);
        }

        public static void Alea(
            Gpu gpu, 
            Real[] mIntraReturn,
            Real[] vClose,
            Real[] vIsAlive,
            Real[] vIsValidDay,
            int m,
            int n)
        {
            using (var cudaIntraReturn = gpu.AllocateDevice(mIntraReturn))
            using (var cudaClose = gpu.AllocateDevice(vClose))
            using (var cudaIsAlive = gpu.AllocateDevice(vIsAlive))
            using (var cudaIsValidDay = gpu.AllocateDevice(vIsValidDay))
            {
                var timer = Stopwatch.StartNew();

                var gridSizeX = Util.DivUp(n, 32);
                var gridSizeY = Util.DivUp(m, 8);
                var lp = new LaunchParam(new dim3(gridSizeX, gridSizeY), new dim3(32, 8));

                gpu.Launch(AleaKernel, lp, cudaIntraReturn.Ptr, cudaClose.Ptr, cudaIsAlive.Ptr, cudaIsValidDay.Ptr, m, n);

                gpu.Synchronize();
                Util.PrintPerformance(timer, "IntraReturn.Alea", 5, m, n);

                Gpu.Copy(cudaIntraReturn, mIntraReturn);
            }
        }

        public static void IlGpu(
            Accelerator gpu,
            Real[] mIntraReturn,
            Real[] vClose,
            Real[] vIsAlive,
            Real[] vIsValidDay,
            int m,
            int n)
        {
            using (var cudaIntraReturn = gpu.Allocate1D(mIntraReturn))
            using (var cudaClose = gpu.Allocate1D(vClose))
            using (var cudaIsAlive = gpu.Allocate1D(vIsAlive))
            using (var cudaIsValidDay = gpu.Allocate1D(vIsValidDay))
            {
                var timer = Stopwatch.StartNew();

                var gridSizeX = Util.DivUp(n, 32);
                var gridSizeY = Util.DivUp(m, 8);
                var lp = ((gridSizeX, gridSizeY, 1), (32, 8));

                gpu.Launch(IlGpuKernel, gpu.DefaultStream, lp, cudaIntraReturn.View, cudaClose.View, cudaIsAlive.View, cudaIsValidDay.View, m, n);

                gpu.Synchronize();
                Util.PrintPerformance(timer, "IntraReturn.IlGpu", 5, m, n);

                cudaIntraReturn.CopyToCPU(mIntraReturn);
            }
        }

        private static void AleaKernel(
            deviceptr<Real> mIntraReturn,
            deviceptr<Real> vClose,
            deviceptr<Real> vIsAlive,
            deviceptr<Real> vIsValidDay,
            int m,
            int n)
        {
            var i = blockIdx.y * blockDim.y + threadIdx.y;
            var j = blockIdx.x * blockDim.x + threadIdx.x;

            if (i < m && j < n)
            {
                var close = vClose[j];
                var isAlive = vIsAlive[j];
                var isValidDay = vIsValidDay[j];
                var current = mIntraReturn[i * n + j];

                mIntraReturn[i * n + j] = close > 0 ? MaskAlea((current - close) / close) * isAlive * isValidDay : 0;
            }
        }

        private static void IlGpuKernel(
            ArrayView1D<Real, Stride1D.Dense> mIntraReturn,
            ArrayView1D<Real, Stride1D.Dense> vClose,
            ArrayView1D<Real, Stride1D.Dense> vIsAlive,
            ArrayView1D<Real, Stride1D.Dense> vIsValidDay,
            int m,
            int n)
        {
            var i = Grid.IdxY * Group.DimY + Group.IdxY;
            var j = Grid.IdxX * Group.DimX + Group.IdxX;

            if (i < m && j < n)
            {
                var close = vClose[j];
                var isAlive = vIsAlive[j];
                var isValidDay = vIsValidDay[j];
                var current = mIntraReturn[i * n + j];

                mIntraReturn[i * n + j] = close > 0 ? MaskIlGpu((current - close) / close) * isAlive * isValidDay : 0;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Real Mask(Real value)
        {
            // ReSharper disable once CompareOfFloatsByEqualityOperator
            return value * 0 == 0 ? value : 0;
        }

        private static Real MaskAlea(Real value)
        {
            return LibDevice.__nv_isfinited(value) != 0 ? value : 0;
        }

        private static Real MaskIlGpu(Real value)
        {
            // TODO figure out access to CUDA intrinsics
            // ReSharper disable once CompareOfFloatsByEqualityOperator
            return value * 0 == 0 ? value : 0;
        }
    }
}
