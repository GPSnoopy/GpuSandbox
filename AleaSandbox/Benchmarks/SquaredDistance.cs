using System;
using System.Diagnostics;
using Alea;
using Alea.CSharp;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

#if DOUBLE_PRECISION
    using Real = System.Double;
    using Real2 = Alea.double2;
#else
    using Real = System.Single;
    using Real2 = Alea.float2;
#endif

namespace AleaSandbox.Benchmarks
{
    internal static class SquaredDistance
    {
        public static void Initialise(
            Real[] mCoordinates,
            int c,
            int n)
        {
            var rand = new Random(1);

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

        public static void Alea(
            Gpu gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            using (var cudaSquaredDistance = gpu.AllocateDevice(mSquaredDistances))
            using (var cudaCoordinates = gpu.AllocateDevice(mCoordinates))
            {
                var timer = Stopwatch.StartNew();

                const int blockSize = 128;

                var gridSize = Util.DivUp(n * n, blockSize);
                var lp = new LaunchParam(gridSize, blockSize);

                gpu.Launch(AleaKernel, lp, cudaSquaredDistance.Ptr, cudaCoordinates.Ptr, c, n);

                gpu.Synchronize();
                Util.PrintPerformance(timer, "SquaredDistance.Alea", n, c, n);

                Gpu.Copy(cudaSquaredDistance, mSquaredDistances);
            }
        }

        public static void IlGpu(
            CudaAccelerator gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            using (var cudaSquaredDistance = gpu.Allocate<Real>(mSquaredDistances.Length))
            using (var cudaCoordinates = gpu.Allocate<Real>(mCoordinates.Length))
            {
                cudaSquaredDistance.CopyFrom(mSquaredDistances, 0, 0, mSquaredDistances.Length);
                cudaCoordinates.CopyFrom(mCoordinates, 0, 0, mCoordinates.Length);

                var timer = Stopwatch.StartNew();

                const int blockSize = 128;

                var gridSize = Util.DivUp(n * n, blockSize);
                var lp = (gridSize, blockSize);

                var kernel = gpu.LoadStreamKernel<ArrayView<Real>, ArrayView<Real>, int, int>(IlGpuKernel);
                kernel(lp, cudaSquaredDistance.View, cudaCoordinates.View, c, n);

                gpu.Synchronize();
                Util.PrintPerformance(timer, "SquaredDistance.IlGpu", n, c, n);

                cudaSquaredDistance.CopyTo(mSquaredDistances, 0, 0, mSquaredDistances.Length);
            }
        }

        public static void AleaSharedMemory(
            Gpu gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            AleaOptimisedImpl(gpu, mSquaredDistances, mCoordinates, c, n, "SquaredDistance.AleaSharedMemory", AleaKernelSharedMemory, i => i);
        }

        public static void IlGpuSharedMemory(
            CudaAccelerator gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            IlGpuOptimisedImpl(gpu, mSquaredDistances, mCoordinates, c, n, "SquaredDistance.IlGpuSharedMemory", IlGpuKernelSharedMemory, i => i);
        }

        public static void AleaFloat2(
            Gpu gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            if (n % 2 != 0) throw new ArgumentException("n must be a multiple of 2");

            AleaOptimisedImpl(gpu, mSquaredDistances, mCoordinates, c, n, "SquaredDistance.AleaFloat2", AleaKernelFloat2, i => i);
        }

        public static void AleaConstants(
            Gpu gpu, 
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            if (n % 2 != 0) throw new ArgumentException("n must be a multiple of 2");

            AleaOptimisedImpl(gpu, mSquaredDistances, mCoordinates, c, n, "SquaredDistance.AleaConstants", AleaKernelConstants, Gpu.Constant);
        }

        public static void AleaLocalMemory(
            Gpu gpu, 
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n)
        {
            if (n % 2 != 0) throw new ArgumentException("n must be a multiple of 2");

            AleaOptimisedImpl(gpu, mSquaredDistances, mCoordinates, c, n, "SquaredDistance.AleaLocalMemory", AleaKernelLocalMemory);
        }

        private static void AleaOptimisedImpl<TInt>(
            Gpu gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n,
            string name,
            Action<deviceptr<Real>, deviceptr<Real>, TInt, int, int> kernel,
            Func<int, TInt> numCoordGetter)
        {
            using (var cudaSquaredDistance = gpu.AllocateDevice<Real>(n, n))
            using (var cudaCoordinates = gpu.AllocateDevice(mCoordinates))
            {
                var timer = Stopwatch.StartNew();

                const int blockSize = 128;
                var gridSize = Util.DivUp(n, blockSize);
                var lp = new LaunchParam(new dim3(gridSize, gridSize, 1), new dim3(blockSize, 1, 1), 2 * c * blockSize * sizeof(Real));
                var pitch = cudaSquaredDistance.PitchInElements.ToInt32();

                gpu.Launch(kernel, lp, cudaSquaredDistance.Ptr, cudaCoordinates.Ptr, numCoordGetter(c), n, pitch);

                gpu.Synchronize();
                Util.PrintPerformance(timer, name, n, c, n);

                Gpu.Copy2D(cudaSquaredDistance, mSquaredDistances, n, n);
            }
        }

        private static void AleaOptimisedImpl(
            Gpu gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n,
            string name,
            Action<deviceptr<Real>, deviceptr<Real>, Constant<int>, Constant<int>, int, int> kernel)
        {
            using (var cudaSquaredDistance = gpu.AllocateDevice<Real>(n, n))
            using (var cudaCoordinates = gpu.AllocateDevice(mCoordinates))
            {
                var timer = Stopwatch.StartNew();

                const int blockSize = 256;
                var gridSize = Util.DivUp(n, blockSize);
                var lp = new LaunchParam(new dim3(gridSize, gridSize, 1), new dim3(blockSize, 1, 1));
                var pitch = cudaSquaredDistance.PitchInElements.ToInt32();

                gpu.Launch(kernel, lp, cudaSquaredDistance.Ptr, cudaCoordinates.Ptr, Gpu.Constant(blockSize), Gpu.Constant(c), n, pitch);

                gpu.Synchronize();
                Util.PrintPerformance(timer, name, n, c, n);

                Gpu.Copy2D(cudaSquaredDistance, mSquaredDistances, n, n);
            }
        }

        private static void IlGpuOptimisedImpl<TInt>(
            CudaAccelerator gpu,
            Real[] mSquaredDistances,
            Real[] mCoordinates,
            int c,
            int n,
            string name,
            Action<ArrayView2D<Real>, ArrayView<Real>, TInt, int> kernelFunc,
            Func<int, TInt> numCoordGetter)
        where TInt : struct
        {
            using (var cudaSquaredDistance = gpu.Allocate<Real>(n, n))
            using (var cudaCoordinates = gpu.Allocate<Real>(mCoordinates.Length))
            {
                cudaSquaredDistance.CopyFrom(mSquaredDistances, 0, (0, 0), mSquaredDistances.Length);
                cudaCoordinates.CopyFrom(mCoordinates, 0, 0, mCoordinates.Length);

                var timer = Stopwatch.StartNew();

                const int blockSize = 128;
                var gridSize = Util.DivUp(n, blockSize);
                var lp = ((gridSize, gridSize, 1), (blockSize, 1, 1), 2 * c * blockSize);
                var pitch = cudaSquaredDistance.Extent.X;

                var kernel = gpu.LoadStreamKernel(kernelFunc);
                kernel(lp, cudaSquaredDistance.View, cudaCoordinates.View, numCoordGetter(c), n);

                gpu.Synchronize();
                Util.PrintPerformance(timer, name, n, c, n);

                cudaSquaredDistance.CopyTo(mSquaredDistances, (0, 0), 0, (n, n));
            }
        }

        private static void AleaKernel(
            deviceptr<Real> mSquaredDistances,
            deviceptr<Real> mCoordinates,
            int c,
            int n)
        {
            var j = blockIdx.x * blockDim.x + threadIdx.x;

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

        private static void IlGpuKernel(
            ArrayView<Real> mSquaredDistances,
            ArrayView<Real> mCoordinates,
            int c,
            int n)
        {
            var j = Grid.IndexX * Group.DimensionX + Group.IndexX;

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

        private static void AleaKernelSharedMemory(
            deviceptr<Real> mSquaredDistances,
            deviceptr<Real> mCoordinates,
            int c,
            int n,
            int pitch)
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

            var shared = DeviceFunction.AddressOfArray(__shared__.ExternArray<Real>());
            var coordinatesI = shared.Ptr(0);
            var coordinatesJ = shared.Ptr(c * blockDim.x);

            var bI = blockIdx.y * blockDim.x;
            var bJ = blockIdx.x * blockDim.x;

            for (int k = 0; k != c; ++k)
            {
                if (bI + threadIdx.x < n)
                {
                    coordinatesI[k * blockDim.x + threadIdx.x] = mCoordinates[k * n + bI + threadIdx.x];
                }

                if (bJ + threadIdx.x < n)
                {
                    coordinatesJ[k * blockDim.x + threadIdx.x] = mCoordinates[k * n + bJ + threadIdx.x];
                }
            }

            DeviceFunction.SyncThreads();

            if (bJ + threadIdx.x < n)
            {
                for (int i = 0; i < blockDim.x && bI + i < n; ++i)
                {
                    Real dist = 0;

                    for (int k = 0; k != c; ++k)
                    {
                        var coord1 = coordinatesI[k * blockDim.x + i]; //mCoordinates[k * x + i];
                        var coord2 = coordinatesJ[k * blockDim.x + threadIdx.x]; //mCoordinates[k * x + j];
                        var diff = coord1 - coord2;

                        dist += diff * diff;
                    }

                    mSquaredDistances[(bI + i) * pitch + (bJ + threadIdx.x)] = dist;
                }
            }
        }

        private static void IlGpuKernelSharedMemory(
            ArrayView2D<Real> mSquaredDistances,
            ArrayView<Real> mCoordinates,
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

            var shared = SharedMemory.AllocateDynamic<Real>();
            var coordinatesI = shared.GetSubView(0, c * Group.DimensionX);
            var coordinatesJ = shared.GetSubView(c * Group.DimensionX);

            var bI = Group.IndexY * Group.DimensionX;
            var bJ = Group.IndexX * Group.DimensionX;

            for (int k = 0; k != c; ++k)
            {
                if (bI + Group.IndexX < n)
                {
                    coordinatesI[k * Group.DimensionX + Group.IndexX] = mCoordinates[k * n + bI + Group.IndexX];
                }

                if (bJ + Group.IndexX < n)
                {
                    coordinatesJ[k * Group.DimensionX + Group.IndexX] = mCoordinates[k * n + bJ + Group.IndexX];
                }
            }
            
            Group.Barrier();

            if (bJ + Group.IndexX < n)
            {
                for (int i = 0; i < Group.DimensionX && bI + i < n; ++i)
                {
                    Real dist = 0;

                    for (int k = 0; k != c; ++k)
                    {
                        var coord1 = coordinatesI[k * Group.DimensionX + i]; //mCoordinates[k * x + i];
                        var coord2 = coordinatesJ[k * Group.DimensionX + Group.IndexX]; //mCoordinates[k * x + j];
                        var diff = coord1 - coord2;

                        dist += diff * diff;
                    }

                    mSquaredDistances[bI + i, bJ + Group.IndexX] = dist;
                }
            }
        }

        private static void AleaKernelFloat2(
            deviceptr<Real> mSquaredDistances,
            deviceptr<Real> mCoordinates,
            int c,
            int n,
            int pitch)
        {
            // Same as KernelSharedMemory, but one thread does two element in one by using float2 reads.

            var shared = DeviceFunction.AddressOfArray(__shared__.ExternArray<Real>());
            var coordinatesI = shared.Ptr(0);
            var coordinatesJ = shared.Ptr(c * blockDim.x);

            var bI = blockIdx.y * blockDim.x;
            var bJ = blockIdx.x * blockDim.x;

            for (int k = 0; k != c; ++k)
            {
                if (bI + threadIdx.x < n)
                {
                    coordinatesI[k * blockDim.x + threadIdx.x] = mCoordinates[k * n + bI + threadIdx.x];
                }

                if (bJ + threadIdx.x < n)
                {
                    coordinatesJ[k * blockDim.x + threadIdx.x] = mCoordinates[k * n + bJ + threadIdx.x];
                }
            }

            DeviceFunction.SyncThreads();

            var line = threadIdx.x / (blockDim.x / 2);
            var tid = threadIdx.x % (blockDim.x / 2);

            if (bJ + tid * 2 < n)
            {
                var coordinatesJ2 = coordinatesJ.Reinterpret<Real2>();

                for (int i = line; i < blockDim.x && bI + i < n; i += 2)
                {
                    Real dist0 = 0;
                    Real dist1 = 0;

                    for (int k = 0; k != c; ++k)
                    {
                        var coord1 = coordinatesI[k * blockDim.x + i];
                        var coord2 = coordinatesJ2[(k * blockDim.x / 2) + tid];
                        var diff = new Real2(coord1 - coord2.x, coord1 - coord2.y);

                        dist0 += diff.x * diff.x;
                        dist1 += diff.y * diff.y;
                    }

                    mSquaredDistances[(bI + i) * pitch + (bJ + 2 * tid + 0)] = dist0;
                    mSquaredDistances[(bI + i) * pitch + (bJ + 2 * tid + 1)] = dist1;
                }
            }
        }

        private static void AleaKernelConstants(
            deviceptr<Real> mSquaredDistances,
            deviceptr<Real> mCoordinates,
            Constant<int> c,
            int n,
            int pitch)
        {
            // Same as CudaKernelOptimised2, but the number of coordinates is given as a meta-constant.
            // Also, we write the results as float2.

            var shared = DeviceFunction.AddressOfArray(__shared__.ExternArray<Real>());
            var coordinatesI = shared.Ptr(0);
            var coordinatesJ = shared.Ptr(c.Value * blockDim.x);

            var bI = blockIdx.y * blockDim.x;
            var bJ = blockIdx.x * blockDim.x;

            for (int k = 0; k != c.Value; ++k)
            {
                if (bI + threadIdx.x < n)
                {
                    coordinatesI[k * blockDim.x + threadIdx.x] = mCoordinates[k * n + bI + threadIdx.x];
                }

                if (bJ + threadIdx.x < n)
                {
                    coordinatesJ[k * blockDim.x + threadIdx.x] = mCoordinates[k * n + bJ + threadIdx.x];
                }
            }

            DeviceFunction.SyncThreads();

            var line = threadIdx.x / (blockDim.x / 2);
            var tid = threadIdx.x % (blockDim.x / 2);

            if (bJ + tid * 2 < n)
            {
                var coordinatesJ2 = coordinatesJ.Reinterpret<Real2>();

                for (int i = line; i < blockDim.x && bI + i < n; i += 2)
                {
                    var dist = default(Real2);

                    for (int k = 0; k != c.Value; ++k)
                    {
                        var coord1 = coordinatesI[k * blockDim.x + i];
                        var coord2 = coordinatesJ2[(k * blockDim.x / 2) + tid];
                        var diff = new Real2(coord1 - coord2.x, coord1 - coord2.y);

                        dist.x += diff.x * diff.x;
                        dist.y += diff.y * diff.y;
                    }

                    var dst = mSquaredDistances.Ptr((bI + i) * pitch + bJ).Reinterpret<Real2>();
                    dst[tid] = dist;
                }
            }
        }

        private static void AleaKernelLocalMemory(
            deviceptr<Real> mSquaredDistances,
            deviceptr<Real> mCoordinates,
            Constant<int> dimX,
            Constant<int> c,
            int n,
            int pitch)
        {
            // Same as KernelConstants, but use both local and shared memory to increase the effective shared memory.

            var coordinatesI = __shared__.Array<Real>(c.Value * dimX.Value);
            var coordinatesJ = __local__.Array<Real2>(c.Value);

            var bI = blockIdx.y * dimX.Value;
            var bJ = blockIdx.x * dimX.Value;
            var line = threadIdx.x / (dimX.Value / 2);
            var tid = threadIdx.x % (dimX.Value / 2);
            var isActive = bJ + tid * 2 < n;

            for (int k = 0; k != c.Value; ++k)
            {
                if (bI + threadIdx.x < n)
                {
                    coordinatesI[k * dimX.Value + threadIdx.x] = mCoordinates[k * n + bI + threadIdx.x];
                }

                if (isActive)
                {
                    var mCoordinates2 = mCoordinates.Reinterpret<Real2>();
                    coordinatesJ[k] = mCoordinates2[(k * n + bJ) / 2 + tid];
                }
            }

            DeviceFunction.SyncThreads();

            if (isActive)
            {
                for (int i = line; i < dimX.Value && bI + i < n; i += 2)
                {
                    var dist = default(Real2);

                    for (int k = 0; k != c.Value; ++k)
                    {
                        var coord1 = coordinatesI[k * dimX.Value + i];
                        var coord2 = coordinatesJ[k];
                        var diff = new Real2(coord1 - coord2.x, coord1 - coord2.y);

                        dist.x += diff.x * diff.x;
                        dist.y += diff.y * diff.y;
                    }

                    var dst = mSquaredDistances.Reinterpret<Real2>();
                    dst[((bI + i) * pitch + bJ) / 2 + tid] = dist;
                }
            }
        }
    }
}
