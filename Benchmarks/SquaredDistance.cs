using System;
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;

#if DOUBLE_PRECISION
    using Real = System.Double;
    using Real2 = ILGPU.Util.Double2;
#else
    using Real = System.Single;
    using Real2 = ILGPU.Util.Float2;
#endif

namespace GpuSandbox.Benchmarks
{
    internal static class SquaredDistance
    {
        public static void Initialise(
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            var rand = new Random(1);

            for (int i = 0; i != n; ++i)
            {
                for (int j = 0; j != n; ++j)
                {
                    mSquaredDistances[i * n + j] = 0;
                }
            }

            for (int i = 0; i != c * n; ++i)
            {
                mCoordinates[i] = (Real) rand.NextDouble();
            }
        }

        public static void Managed(
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            var timer = Stopwatch.StartNew();

            for (int i = 0; i != n; ++i)
            {
                for (int j = 0; j != n; ++j)
                {
                    Real dist = 0;

                    for (int k = 0; k != c; ++k)
                    {
                        var coord1 = mCoordinates[k * n + i];
                        var coord2 = mCoordinates[k * n + j];
                        var diff = coord1 - coord2;

                        dist += diff * diff;
                    }

                    mSquaredDistances[i * n + j] = dist;
                }
            }

            Util.PrintPerformance(timer, "SquaredDistance.Managed", n, c, n);
        }

        public static void Gpu(
            Accelerator gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            using var cudaSquaredDistance = gpu.Allocate1D(mSquaredDistances);
            using var cudaCoordinates = gpu.Allocate1D(mCoordinates);
            var timer = Stopwatch.StartNew();

            const int blockSize = 128;

            var gridSize = Util.DivUp(n * n, blockSize);
            var lp = (gridSize, blockSize);

            gpu.Launch(Kernel, gpu.DefaultStream, lp, cudaSquaredDistance.View, cudaCoordinates.View, c, n);
            gpu.Synchronize();

            Util.PrintPerformance(timer, "SquaredDistance.Gpu", n, c, n);

            cudaSquaredDistance.CopyToCPU(mSquaredDistances);
        }

        public static void GpuSharedMemory(
            Accelerator gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            GpuOptimisedImpl(gpu, mSquaredDistances, mCoordinates, c, n, "SquaredDistance.Gpu.SharedMemory", KernelSharedMemory, i => i);
        }

        public static void GpuFloat2(
            Accelerator gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            if (n % 2 != 0) throw new ArgumentException("n must be a multiple of 2");

            GpuOptimisedImpl(gpu, mSquaredDistances, mCoordinates, c, n, "SquaredDistance.Gpu.Float2", KernelFloat2, i => i);
        }

        public static void GpuConstants(
            Accelerator gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            if (n % 2 != 0) throw new ArgumentException("n must be a multiple of 2");

            GpuOptimisedImpl(gpu, mSquaredDistances, mCoordinates, c, n, "SquaredDistance.Gpu.Constants", KernelConstants, i => new SpecializedValue<int>(i));
        }

        public static void GpuLocalMemory(
            Accelerator gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            if (n % 2 != 0) throw new ArgumentException("n must be a multiple of 2");

            GpuOptimisedImpl(gpu, mSquaredDistances, mCoordinates, c, n, "SquaredDistance.Gpu.LocalMemory", KernelLocalMemory);
        }

        private static void GpuOptimisedImpl<TInt>(
            Accelerator gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n,
            string name,
            Action<ArrayView2D<Real, Stride2D.DenseX>, ArrayView1D<Real, Stride1D.Dense>, TInt, int> kernelFunc,
            Func<int, TInt> numCoordGetter)
        where TInt : struct
        {
            using var cudaSquaredDistance = gpu.Allocate2DDenseX<Real>((n, n));
            using var cudaCoordinates = gpu.Allocate1D(mCoordinates);
            var timer = Stopwatch.StartNew();

            const int blockSize = 128;
            var gridSize = Util.DivUp(n, blockSize);
            var lp = ((gridSize, gridSize, 1), (blockSize, 1, 1), SharedMemoryConfig.RequestDynamic<Real>(2 * c * blockSize));

            gpu.Launch(kernelFunc, gpu.DefaultStream, lp, cudaSquaredDistance.View, cudaCoordinates.View, numCoordGetter(c), n);
            gpu.Synchronize();

            Util.PrintPerformance(timer, name, n, c, n);

            cudaSquaredDistance.View.As1DView(new Stride1D.Dense()).CopyToCPU(mSquaredDistances);
        }

        private static void GpuOptimisedImpl(
            Accelerator gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n,
            string name,
            Action<ArrayView2D<Real, Stride2D.DenseX>, ArrayView1D<Real, Stride1D.Dense>, SpecializedValue<int>, SpecializedValue<int>, int> kernelFunc)
        {
            using var cudaSquaredDistance = gpu.Allocate2DDenseX<Real>((n, n));
            using var cudaCoordinates = gpu.Allocate1D(mCoordinates);
            var timer = Stopwatch.StartNew();

            const int blockSize = 128;
            var gridSize = Util.DivUp(n, blockSize);
            var lp = ((gridSize, gridSize, 1), (blockSize, 1, 1));

            gpu.Launch(kernelFunc, gpu.DefaultStream, lp, cudaSquaredDistance.View, cudaCoordinates.View, SpecializedValue.New(blockSize), SpecializedValue.New(c), n);
            gpu.Synchronize();

            Util.PrintPerformance(timer, name, n, c, n);

            cudaSquaredDistance.AsArrayView<Real>(0, n * n).CopyToCPU(mSquaredDistances);
        }

        private static void Kernel(
            ArrayView1D<Real, Stride1D.Dense> mSquaredDistances,
            ArrayView1D<Real, Stride1D.Dense> mCoordinates,
            int c,
            int n)
        {
            var j = Grid.IdxX * Group.DimX + Group.IdxX;

            if (j < n)
            {
                for (int i = 0; i < n; ++i)
                {
                    Real dist = 0;

                    for (int k = 0; k != c; ++k)
                    {
                        var coord1 = mCoordinates[k * n + i];
                        var coord2 = mCoordinates[k * n + j];
                        var diff = coord1 - coord2;

                        dist += diff * diff;
                    }

                    mSquaredDistances[i * n + j] = dist;
                }
            }
        }

        private static void KernelSharedMemory(
            ArrayView2D<Real, Stride2D.DenseX> mSquaredDistances,
            ArrayView1D<Real, Stride1D.Dense> mCoordinates,
            int c,
            int n)
        {
            // We've got shared memory of two vector of K dimensions for B points:
            //
            //      var coordI = __shared__ new Real[k*blockDim.x];
            //      var coordJ = __shared__ new Real[k*blockDim.x];
            //
            // We fill in these two vectors with the coordinates of the I points and J points.
            // Afterwards, the current block will compute the euclidean distances between all 
            // the I points and J points, producing a square matrix [B, B].
            //
            // This optimisation means that when producing the square matrix, the I and J points
            // coordinates are only read once.
            //
            // This optimisation works well if K is small enough. Otherwise the shared memory is
            // too small and not enough blocks get schedule per SM.

            var shared = SharedMemory.GetDynamic<Real>();
            var coordinatesI = shared.SubView(0, c * Group.DimX);
            var coordinatesJ = shared.SubView(c * Group.DimX);

            var bI = Grid.IdxY * Group.DimX;
            var bJ = Grid.IdxX * Group.DimX;

            for (int k = 0; k != c; ++k)
            {
                if (bI + Group.IdxX < n)
                {
                    coordinatesI[k * Group.DimX + Group.IdxX] = mCoordinates[k * n + bI + Group.IdxX];
                }

                if (bJ + Group.IdxX < n)
                {
                    coordinatesJ[k * Group.DimX + Group.IdxX] = mCoordinates[k * n + bJ + Group.IdxX];
                }
            }
            
            Group.Barrier();

            if (bJ + Group.IdxX < n)
            {
                for (int i = 0; i < Group.DimX && bI + i < n; ++i)
                {
                    Real dist = 0;

                    for (int k = 0; k != c; ++k)
                    {
                        var coord1 = coordinatesI[k * Group.DimX + i]; //mCoordinates[k * x + i];
                        var coord2 = coordinatesJ[k * Group.DimX + Group.IdxX]; //mCoordinates[k * x + j];
                        var diff = coord1 - coord2;

                        dist += diff * diff;
                    }

                    mSquaredDistances[bJ + Group.IdxX, bI + i] = dist;
                }
            }
        }

        private static void KernelFloat2(
            ArrayView2D<Real, Stride2D.DenseX> mSquaredDistances,
            ArrayView1D<Real, Stride1D.Dense> mCoordinates,
            int c,
            int n)
        {
            // Same as KernelSharedMemory, but one thread does two element in one by using float2 reads.

            var shared = SharedMemory.GetDynamic<Real>();
            var coordinatesI = shared.SubView(0, c * Group.DimX);
            var coordinatesJ = shared.SubView(c * Group.DimX);

            var bI = Grid.IdxY * Group.DimX;
            var bJ = Grid.IdxX * Group.DimX;

            for (int k = 0; k != c; ++k)
            {
                if (bI + Group.IdxX < n)
                {
                    coordinatesI[k * Group.DimX + Group.IdxX] = mCoordinates[k * n + bI + Group.IdxX];
                }

                if (bJ + Group.IdxX < n)
                {
                    coordinatesJ[k * Group.DimX + Group.IdxX] = mCoordinates[k * n + bJ + Group.IdxX];
                }
            }

            Group.Barrier();

            var line = Group.IdxX / (Group.DimX / 2);
            var tid = Group.IdxX % (Group.DimX / 2);

            if (bJ + tid * 2 < n)
            {
                var coordinatesJ2 = coordinatesJ.Cast<Real2>();

                for (int i = line; i < Group.DimX && bI + i < n; i += 2)
                {
                    var dist = default(Real2);

                    for (int k = 0; k != c; ++k)
                    {
                        var coord1 = coordinatesI[k * Group.DimX + i];
                        var coord2 = coordinatesJ2[(k * Group.DimX / 2) + tid];
                        var diff = new Real2(coord1 - coord2.X, coord1 - coord2.Y);

                        dist += diff * diff;
                    }

                    mSquaredDistances[bJ + 2 * tid + 0, bI + i] = dist.X;
                    mSquaredDistances[bJ + 2 * tid + 1, bI + i] = dist.Y;
                }
            }
        }

        private static void KernelConstants(
            ArrayView2D<Real, Stride2D.DenseX> mSquaredDistances,
            ArrayView1D<Real, Stride1D.Dense> mCoordinates,
            SpecializedValue<int> c,
            int n)
        {
            // Same as CudaKernelOptimised2, but the number of coordinates is given as a meta-constant.
            // Also, we write the results as float2.

            var shared = SharedMemory.GetDynamic<Real>();
            var coordinatesI = shared.SubView(0, c * Group.DimX);
            var coordinatesJ = shared.SubView(c * Group.DimX);

            var bI = Grid.IdxY * Group.DimX;
            var bJ = Grid.IdxX * Group.DimX;

            for (int k = 0; k != c; ++k)
            {
                if (bI + Group.IdxX < n)
                {
                    coordinatesI[k * Group.DimX + Group.IdxX] = mCoordinates[k * n + bI + Group.IdxX];
                }

                if (bJ + Group.IdxX < n)
                {
                    coordinatesJ[k * Group.DimX + Group.IdxX] = mCoordinates[k * n + bJ + Group.IdxX];
                }
            }

            Group.Barrier();

            var line = Group.IdxX / (Group.DimX / 2);
            var tid = Group.IdxX % (Group.DimX / 2);

            if (bJ + tid * 2 < n)
            {
                var coordinatesJ2 = coordinatesJ.Cast<Real2>();

                for (int i = line; i < Group.DimX & bI + i < n; i += 2)
                {
                    var dist = default(Real2);

                    for (int k = 0; k != c; ++k)
                    {
                        var coord1 = coordinatesI[k * Group.DimX + i];
                        var coord2 = coordinatesJ2[(k * Group.DimX / 2) + tid];
                        var diff = new Real2(coord1 - coord2.X, coord1 - coord2.Y);

                        dist += diff * diff;
                    }

                    var dst = mSquaredDistances.Cast<Real, Real2>();
                    dst[bJ / 2 + tid, bI + i] = dist;
                }
            }
        }

        private static void KernelLocalMemory(
            ArrayView2D<Real, Stride2D.DenseX> mSquaredDistances,
            ArrayView1D<Real, Stride1D.Dense> mCoordinates,
            SpecializedValue<int> dimX,
            SpecializedValue<int> c,
            int n)
        {
            // Same as KernelConstants, but use both local and shared memory to increase the effective shared memory.

            var coordinatesI = SharedMemory.Allocate<Real>(c * dimX);
            var coordinatesJ = new Real2[c.Value];

            var bI = Grid.IdxY * dimX;
            var bJ = Grid.IdxX * dimX;
            var line = Group.IdxX / (dimX / 2);
            var tid = Group.IdxX % (dimX / 2);
            var isActive = bJ + tid * 2 < n;

            for (int k = 0; k != c.Value; ++k)
            {
                if (bI + Group.IdxX < n)
                {
                    coordinatesI[k * dimX + Group.IdxX] = mCoordinates[k * n + bI + Group.IdxX];
                }

                if (isActive)
                {
                    var mCoordinates2 = mCoordinates.Cast<Real, Real2>();
                    coordinatesJ[k] = mCoordinates2[(k * n + bJ) / 2 + tid];
                }
            }

            Group.Barrier();

            if (isActive)
            {
                for (int i = line; i < dimX && bI + i < n; i += 2)
                {
                    var dist = default(Real2);

                    for (int k = 0; k != c.Value; ++k)
                    {
                        var coord1 = coordinatesI[k * dimX + i];
                        var coord2 = coordinatesJ[k];
                        var diff = new Real2(coord1 - coord2.X, coord1 - coord2.Y);

                        dist += diff * diff;
                    }

                    var dst = mSquaredDistances.Cast<Real, Real2>();
                    dst[bJ / 2 + tid, bI + i] = dist;
                }
            }
        }
    }
}
