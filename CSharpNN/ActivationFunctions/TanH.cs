using System;

namespace CSharpNN
{
    internal class TanH : IFunction
    {
        public double Get(double x)
        {
            return (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
        }

        public double GetFirstDerivative(double x)
        {
            var tanh = Get(x);
            return 1 - tanh * tanh;
        }
    }
}