using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Linq;

namespace CSharpNN
{
    internal class Layer
    {
        private readonly IFunction _f;
        private Matrix<double> _a;
        private Vector<double> _biases;
        private Matrix<double> _input;
        private Matrix<double> _weights;
        private Matrix<double> _z;

        public Layer(int input, int output, IFunction f)
        {
            _f = f;
            IContinuousDistribution distribution = new Normal();

            _weights = DenseMatrix.CreateRandom(output, input, distribution);

            _biases = DenseVector.CreateRandom(output, distribution);
        }

        public Matrix<double> Activation => _a;
        public IFunction ActivationFunction => _f;

        public Vector<double> Biases
        {
            get => _biases;
            set => _biases = value;
        }

        public Matrix<double> Input => _input;

        public Matrix<double> Weights
        {
            get => _weights;
            set => _weights = value;
        }

        public Matrix<double> Z => _z;

        public Matrix<double> Forward(Matrix<double> input)
        {
            _input = input;
            var m = _weights * input;

            var biasMatrix = DenseMatrix.Build.DenseOfColumnVectors(Enumerable.Repeat(_biases, m.ColumnCount));

            _z = m + biasMatrix;

            _a = _z.Map(_f.Get);

            return _a;
        }
    }
}