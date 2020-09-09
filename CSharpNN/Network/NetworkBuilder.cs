using System.Collections.Generic;

namespace CSharpNN.Network
{
    internal class NetworkBuilder
    {
        public static Network Build(int inputNodes, LayerOptions outputLayerOptions, LayerOptions[] hiddenLayerOptions, int? randomSeed)
        {
            var currentInput = inputNodes;

            var layers = new List<Layer>();

            foreach (var hiddenLayerOption in hiddenLayerOptions)
            {
                var layer = new Layer(currentInput, hiddenLayerOption.OutputNodes, hiddenLayerOption.ActivationFunction, randomSeed);

                layers.Add(layer);

                currentInput = hiddenLayerOption.OutputNodes;
            }

            var lastLayer = new Layer(currentInput, outputLayerOptions.OutputNodes, outputLayerOptions.ActivationFunction, randomSeed);

            layers.Add(lastLayer);

            return new Network(layers);
        }
    }
}