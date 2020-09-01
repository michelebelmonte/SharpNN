using MathNet.Numerics.LinearAlgebra;

namespace CSharpNN.CostFunctions
{
    public class EntropyCostFunction : ICostFunction
    {
        private const double epsilon = 0.0000000001;

        public Matrix<double> Get(Matrix<double> a, Matrix<double> y)
        {
            var a_epsilon = a + epsilon;
            var a_1_epsilon = 1.0 - a + epsilon;

            //-(ylna+(1−y)ln(1−a))
            var result = -y.PointwiseMultiply(a_epsilon.PointwiseLog())
                         - (1.0 - y).PointwiseMultiply(a_1_epsilon.PointwiseLog());

            return result / a.ColumnCount;
        }

        public Matrix<double> GetFirstDerivative(Matrix<double> a, Matrix<double> y)
        {
            var result = -y.PointwiseDivide(a + epsilon) - (1.0 - y).PointwiseDivide(1.0 - a + epsilon);

            return result;
        }
    }
}