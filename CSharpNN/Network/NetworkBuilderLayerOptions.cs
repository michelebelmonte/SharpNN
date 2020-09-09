namespace CSharpNN
{
    internal class LayerOptions
    {
        public LayerOptions(int outputNodes, IFunction activationFunction)
        {
            OutputNodes = outputNodes;
            ActivationFunction = activationFunction;
        }

        public IFunction ActivationFunction { get; }
        public int OutputNodes { get; }
    }
}