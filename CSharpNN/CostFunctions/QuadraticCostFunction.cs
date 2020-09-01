using MathNet.Numerics.LinearAlgebra;

namespace CSharpNN.CostFunctions
{
    public class QuadraticCostFunction : ICostFunction
    {
        public Matrix<double> Get(Matrix<double> a, Matrix<double> y)
        {
            var mse = (a - y).PointwisePower(2) / a.ColumnCount / 2.0;

            return mse;
        }

        public Matrix<double> GetFirstDerivative(Matrix<double> a, Matrix<double> y)
        {
            return (a - y) / a.ColumnCount;
        }
    }
}