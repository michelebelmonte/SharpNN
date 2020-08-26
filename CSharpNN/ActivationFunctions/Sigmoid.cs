using System;

namespace CSharpNN
{
    internal class Sigmoid : IFunction
    {
        public double Get(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public double GetFirstDerivative(double x)
        {
            return Get(x) * (1 - Get(x));
        }
    }
}