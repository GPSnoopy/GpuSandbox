using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Alea;
using Alea.CSharp;

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

        public static void Cuda(
            Real[] mIntraReturn,
            Real[] vClose,
            Real[] vIsAlive,
            Real[] vIsValidDay,
            int m,
            int n)
        {
            var gpu = Gpu.Default;

            using (var cudaIntraReturn = gpu.AllocateDevice(mIntraReturn))
            using (var cudaClose = gpu.AllocateDevice(vClose))
            using (var cudaIsAlive = gpu.AllocateDevice(vIsAlive))
            using (var cudaIsValidDay = gpu.AllocateDevice(vIsValidDay))
            {
                var timer = Stopwatch.StartNew();

                var gridSizeX = Util.DivUp(n, 32);
                var gridSizeY = Util.DivUp(m, 8);
                var lp = new LaunchParam(new dim3(gridSizeX, gridSizeY), new dim3(32, 8));

                gpu.Launch(CudaKernel, lp, cudaIntraReturn.Ptr, cudaClose.Ptr, cudaIsAlive.Ptr, cudaIsValidDay.Ptr, m, n);

                gpu.Synchronize();
                Util.PrintPerformance(timer, "IntraReturn.Cuda", 5, m, n);

                Gpu.Copy(cudaIntraReturn, mIntraReturn);
            }
        }

        private static void CudaKernel(
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

                mIntraReturn[i * n + j] = close > 0 ? MaskCuda((current - close) / close) * isAlive * isValidDay : 0;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Real Mask(Real value)
        {
            // ReSharper disable once CompareOfFloatsByEqualityOperator
            return value * 0 == 0 ? value : 0;
        }

        private static Real MaskCuda(Real value)
        {
            return LibDevice.__nv_isfinited(value) != 0 ? value : 0;
        }
    }
}
