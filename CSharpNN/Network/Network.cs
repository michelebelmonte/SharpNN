using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace CSharpNN.Network
{
    internal class Network
    {
        private readonly List<Layer> _layers = new List<Layer>();

        public Network(IEnumerable<Layer> layers)
        {
            _layers.AddRange(layers);
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