using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace CSharpNN
{
    internal class Network
    {
        private List<Layer> _layers = new List<Layer>();

        public Network(int input, int output, IFunction lastFunction, int[] hiddenLayers, IFunction[] activationFunctions)
        {
            var currentInput = input;

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                var currentOutput = hiddenLayers[i];
                var f = activationFunctions[i];

                var layer = new Layer(currentInput, currentOutput, f);

                _layers.Add(layer);

                currentInput = currentOutput;
            }

            var lastLayer = new Layer(currentInput, output, lastFunction);

            _layers.Add(lastLayer);
        }

        public IReadOnlyCollection<Layer> Layers => _layers;

        public Matrix<double> Forward(Matrix<double> matrix)
        {
            var currentMatrix = matrix;

            foreach (var layer in _layers)
            {
                currentMatrix = layer.Forward(currentMatrix);
            }

            return currentMatrix;
        }
    }
}