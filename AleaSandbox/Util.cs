using System;
using System.Diagnostics;

#if DOUBLE_PRECISION
    using Real = System.Double;
#else
    using Real = System.Single;
#endif

namespace AleaSandbox
{
    internal static class Util
    {
        public static int DivUp(int a, int b)
        {
            return (a + b - 1) / b;
        }

        public static void PrintPerformance(Stopwatch timer, string prefix, int numOps, int m, int n)
        {
            var elapsed = timer.Elapsed;

            Console.WriteLine("{0}: {1} ms, {2:F1} GB/s",
                prefix,
                elapsed.TotalMilliseconds,
                (double)m * n * sizeof(Real) * numOps / elapsed.TotalSeconds / (1024 * 1024 * 1024));
        }
    }
}
