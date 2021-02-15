using System;

namespace GpuSandbox
{
    internal static class Benchmark
    {
        public static void Run(int loops, Action baseInitialise, Action testInitialise, Action compare, Action baseRun, params Action[] testRuns)
        {
            //for (int i = 0; i != loops; ++i)
            {
                baseInitialise();
                baseRun();
            }

            Console.WriteLine();

            foreach (var run in testRuns)
            {
                for (int i = 0; i != loops; ++i)
                {
                    testInitialise();
                    run();
                    compare();
                }

                Console.WriteLine();
            }
        }
    }
}
