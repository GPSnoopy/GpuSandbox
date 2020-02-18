using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
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
            CudaAccelerator gpu,
            Real[] mIntraReturn,
            Real[] vClose,
            Real[] vIsAlive,
            Real[] vIsValidDay,
            int m,
            int n)
        {
            using (var cudaIntraReturn = gpu.Allocate<Real>(mIntraReturn.Length))
            using (var cudaClose = gpu.Allocate<Real>(vClose.Length))
            using (var cudaIsAlive = gpu.Allocate<Real>(vIsAlive.Length))
            using (var cudaIsValidDay = gpu.Allocate<Real>(vIsValidDay.Length))
            {
                cudaIntraReturn.CopyFrom(mIntraReturn, 0, 0, mIntraReturn.Length);
                cudaClose.CopyFrom(vClose, 0, 0, vClose.Length);
                cudaIsAlive.CopyFrom(vIsAlive, 0, 0, vIsAlive.Length);
                cudaIsValidDay.CopyFrom(vIsValidDay, 0, 0, vIsValidDay.Length);

                var timer = Stopwatch.StartNew();

                var gridSizeX = Util.DivUp(n, 32);
                var gridSizeY = Util.DivUp(m, 8);
                var lp = ((gridSizeX, gridSizeY, 1), (32, 8));

                var kernel = gpu.LoadStreamKernel<ArrayView<Real>, ArrayView<Real>, ArrayView<Real>, ArrayView<Real>, int, int>(IlGpuKernel);
                kernel(lp, cudaIntraReturn.View, cudaClose.View, cudaIsAlive.View, cudaIsValidDay.View, m, n);

                gpu.Synchronize();
                Util.PrintPerformance(timer, "IntraReturn.IlGpu", 5, m, n);

                cudaIntraReturn.CopyTo(mIntraReturn, 0, 0, mIntraReturn.Length);
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
            ArrayView<Real> mIntraReturn,
            ArrayView<Real> vClose,
            ArrayView<Real> vIsAlive,
            ArrayView<Real> vIsValidDay,
            int m,
            int n)
        {
            var i = Grid.IndexY * Group.DimensionY + Group.IndexY;
            var j = Grid.IndexX * Group.DimensionX + Group.IndexX;

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
