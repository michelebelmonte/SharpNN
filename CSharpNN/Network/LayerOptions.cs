namespace CSharpNN
{
    internal class LayerOptions
    {
        public LayerOptions(int nodes, IFunction activationFunction)
        {
            Nodes = nodes;
            ActivationFunction = activationFunction;
        }

        public IFunction ActivationFunction { get; }
        public int Nodes { get; }
    }
}