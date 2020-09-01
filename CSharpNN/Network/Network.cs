using System;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;

namespace CSharpNN
{
    internal class Network
    {
        private readonly List<Layer> _layers = new List<Layer>();

        private Network(List<Layer> layers)
        {
            _layers.AddRange(layers);
        }

        public IReadOnlyCollection<Layer> Layers => _layers;

        public static Network Build(int input, LayerOptions outputLayerOptions, LayerOptions[] hiddenLayerOptions, int? randomSeed)
        {
            var currentInput = input;

            List<Layer> layers = new List<Layer>();
            for (int i = 0; i < hiddenLayerOptions.Length; i++)
            {
                var hiddenLayerOption = hiddenLayerOptions[i];

                var layer = new Layer(currentInput, hiddenLayerOption.Nodes, hiddenLayerOption.ActivationFunction, randomSeed);

                layers.Add(layer);

                currentInput = hiddenLayerOption.Nodes;
            }

            var lastLayer = new Layer(currentInput, outputLayerOptions.Nodes, outputLayerOptions.ActivationFunction, randomSeed);

            layers.Add(lastLayer);

            return new Network(layers);
        }

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