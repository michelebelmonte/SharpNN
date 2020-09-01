using CSharpNN.CostFunctions;
using CSharpNN.Infrastructure;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CSharpNN.Trainers
{
    internal class SgdTrainer
    {
        private readonly ICostFunction _costFunction;
        private readonly int _epochs = 30;
        private readonly ILogger _logger;
        private readonly int _minibatch = 10;
        private readonly int? _randomSeed;
        private readonly double _η;

        public SgdTrainer(int epochs, int minibatch, double η, ICostFunction costFunction, ILogger logger, int? randomSeed)
        {
            _epochs = epochs;
            _minibatch = minibatch;
            _η = η;
            _costFunction = costFunction;
            _logger = logger;
            _randomSeed = randomSeed;
        }

        public void Train(Network network, List<double[]> x, List<double[]> y, double trainDataRatio)
        {
            var toBeTaken = (int)(x.Count * trainDataRatio);

            var trainX = x.Take(toBeTaken).ToList();
            var devX = DenseMatrix.OfColumnArrays(x.Skip(toBeTaken));

            var trainY = y.Take(toBeTaken).ToList();
            var devY = DenseMatrix.OfColumnArrays(y.Skip(toBeTaken));

            for (int i = 0; i < _epochs; i++)
            {
                _logger.Log($"Epoch {i:D4} started.");
                var randomTrainData = Shuffler.Shuffle(_randomSeed, trainX, trainY);

                var trainXBatches = SplitList(randomTrainData.Item1, _minibatch).ToList();
                var trainYBatches = SplitList(randomTrainData.Item2, _minibatch).ToList();

                for (int j = 0; j < trainXBatches.Count; j++)
                {
                    var xBatch = trainXBatches[j];
                    var yBatch = trainYBatches[j];

                    var xBatchMatrix = DenseMatrix.OfColumnArrays(xBatch.Select(k => k));
                    var yBatchMatrix = DenseMatrix.OfColumnArrays(yBatch.Select(k => k));

                    var inBatchPred = network.Forward(xBatchMatrix);

                    if (j % 500 == 0 || j + 1 == trainXBatches.Count)
                    {
                        var inBatchmse = _costFunction.Get(yBatchMatrix, inBatchPred).RowSums().Sum();
                        var inBatchprecision = Metrics.GetPrecision(yBatchMatrix, inBatchPred);
                        _logger.Log($"                   Batch: {j + 1:D4}/{trainXBatches.Count()}, #Samples: {xBatch.Count} Mse: {inBatchmse:F3}, Precision: {inBatchprecision:P0}");
                    }

                    Backward(network, yBatchMatrix);
                }

                var pred = network.Forward(devX);

                var mse = _costFunction.Get(devY, pred).RowSums().Sum();
                var precision = Metrics.GetPrecision(devY, pred);

                _logger.Log($"Epoch {i:D4} finished: Mse: {mse:F3}, Precision: {precision:P0}");
                _logger.Log();
            }
        }

        private static List<Layer> GetBackwardLayers(Network network)
        {
            var backwardLayers = network.Layers.ToList();

            backwardLayers.Reverse();
            return backwardLayers;
        }

        private static IEnumerable<List<T>> SplitList<T>(List<T> locations, int nSize = 30)
        {
            for (var i = 0; i < locations.Count; i += nSize)
            {
                yield return locations.GetRange(i, Math.Min(nSize, locations.Count - i));
            }
        }

        private void Backward(Network network, DenseMatrix y)
        {
            var backwardLayers = GetBackwardLayers(network);

            var gradient = _costFunction.GetFirstDerivative(backwardLayers.First().Activation, y);

            foreach (var layer in backwardLayers)
            {
                var sp = layer.Z.Map(layer.ActivationFunction.GetFirstDerivative);

                var δ = gradient.PointwiseMultiply(sp);

                var δ_biases = δ.RowSums();
                var δ_weights = (δ * layer.Input.Transpose());

                gradient = layer.Weights.Transpose() * δ;

                layer.Biases -= _η * δ_biases;
                layer.Weights -= _η * δ_weights;
            }
        }
    }
}