namespace CSharpNN
{
    internal interface IFunction
    {
        double Get(double value);

        double GetFirstDerivative(double value);
    }
}