using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
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

        public static void Gpu(
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

                gpu.Launch(Kernel, gpu.DefaultStream, lp, cudaIntraReturn.View, cudaClose.View, cudaIsAlive.View, cudaIsValidDay.View, m, n);

                gpu.Synchronize();
                Util.PrintPerformance(timer, "IntraReturn.Gpu", 5, m, n);

                cudaIntraReturn.CopyToCPU(mIntraReturn);
            }
        }

        private static void Kernel(
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

                mIntraReturn[i * n + j] = close > 0 ? Mask((current - close) / close) * isAlive * isValidDay : 0;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Real Mask(Real value)
        {
            // ReSharper disable once CompareOfFloatsByEqualityOperator
            return value * 0 == 0 ? value : 0;
        }
    }
}
