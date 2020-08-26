using MathNet.Numerics.LinearAlgebra;
using System;

namespace CSharpNN.CostFunctions
{
    internal interface ICostFunction
    {
        Matrix<double> Get(Matrix<double> a, Matrix<double> y);

        Matrix<double> GetFirstDerivative(Matrix<double> a, Matrix<double> y);
    }
}