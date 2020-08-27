using MathNet.Numerics;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CSharpNN.Infrastructure
{
    internal static class Shuffler
    {
        public static Tuple<List<T>, List<T>> Shuffle<T>(IEnumerable<T> x, IEnumerable<T> y, int seed = 13)
        {
            var rng = new Random(seed);

            var tuples = x.Zip(y, (i, j) => new Tuple<T, T>(i, j)).SelectPermutation(rng).ToList();

            var l0 = tuples.Select(i => i.Item1).ToList();
            var l1 = tuples.Select(i => i.Item2).ToList();

            return new Tuple<List<T>, List<T>>(l0, l1);
        }
    }
}