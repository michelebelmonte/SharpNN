namespace CSharpNN
{
    internal class Relu : IFunction
    {
        private double _alpha;
        private double _beta;

        public Relu() : this(1, 0.0001)
        {
        }

        public Relu(double alpha, double beta)
        {
            _alpha = alpha;
            _beta = beta;
        }

        public double Get(double x)
        {
            if (x >= 0) return x * _alpha;

            return x * _beta;
        }

        public double GetFirstDerivative(double x)
        {
            if (x >= 0) return _alpha;

            return _beta;
        }
    }
}