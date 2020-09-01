using MathNet.Numerics;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CSharpNN.Infrastructure
{
    internal static class Shuffler
    {
        public static (List<T> x, List<T> y) Shuffle<T>(int? randomSeed, IEnumerable<T> x, IEnumerable<T> y)
        {
            var random = randomSeed.HasValue ? new Random(randomSeed.Value) : new Random();
            var tuples = x.Zip(y, (i, j) => (i, j)).SelectPermutation(random).ToList();

            var l0 = tuples.Select(item => item.i).ToList();
            var l1 = tuples.Select(item => item.j).ToList();

            return (l0, l1);
        }
    }
}