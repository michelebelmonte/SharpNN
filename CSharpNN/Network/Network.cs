using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;

namespace CSharpNN
{
    internal class Network
    {
        private readonly List<Layer> _layers = new List<Layer>();

        public Network(int input, LayerOptions outputLayerOptions, LayerOptions[] hiddenLayerOptions)
        {
            var currentInput = input;

            for (int i = 0; i < hiddenLayerOptions.Length; i++)
            {
                var hiddenLayerOption = hiddenLayerOptions[i];

                var layer = new Layer(currentInput, hiddenLayerOption.Nodes, hiddenLayerOption.ActivationFunction);

                _layers.Add(layer);

                currentInput = hiddenLayerOption.Nodes;
            }

            var lastLayer = new Layer(currentInput, outputLayerOptions.Nodes, outputLayerOptions.ActivationFunction);

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