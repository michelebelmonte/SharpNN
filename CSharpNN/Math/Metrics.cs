using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;

namespace CSharpNN
{
    internal static class Metrics
    {
        internal static Matrix<double> GetConfusionMatrix(Matrix<double> y, Matrix<double> pred)
        {
            var numberOfLabels = y.RowCount;

            var result = Matrix<double>.Build.Dense(numberOfLabels, numberOfLabels, (i, j) => 0);

            var yMaxIndices = GetMaxIndices(y);
            var predMaxIndices = GetMaxIndices(pred);

            for (int i = 0; i < yMaxIndices.Count; i++)
            {
                result[yMaxIndices[i], predMaxIndices[i]] += 1;
            }

            return result;
        }

        internal static Vector<double> GetHistogramVector(Matrix<double> y)
        {
            var numberOfLabels = y.RowCount;

            var result = Vector<double>.Build.Dense(numberOfLabels, (i) => 0);

            var yMaxIndices = GetMaxIndices(y);

            for (int i = 0; i < yMaxIndices.Count; i++)
            {
                result[yMaxIndices[i]] += 1;
            }

            return result;
        }

        internal static double GetMse(Matrix<double> y, Matrix<double> pred)
        {
            var mse = (pred - y).PointwisePower(2).ColumnSums().Sum() / pred.ColumnCount / 2.0;

            return mse;
        }

        internal static double GetPrecision(Matrix<double> y, Matrix<double> pred)
        {
            var yMaxIndices = GetMaxIndices(y);
            var predMaxIndices = GetMaxIndices(pred);

            var count = 0.0;
            for (int i = 0; i < yMaxIndices.Count; i++)
            {
                if (yMaxIndices[i] == predMaxIndices[i])
                {
                    count++;
                }
            }

            var precision = count / yMaxIndices.Count;
            return precision;
        }

        private static List<int> GetMaxIndices(Matrix<double> y)
        {
            var maxIndices = new List<int>();
            for (int i = 0; i < y.ColumnCount; i++)
            {
                var column = y.Column(i);

                var index = column.MaximumIndex();

                maxIndices.Add(index);
            }

            return maxIndices;
        }
    }
}