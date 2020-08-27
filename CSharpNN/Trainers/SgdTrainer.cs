using System;
using System.Collections.Generic;
using System.Linq;
using CSharpNN.CostFunctions;
using CSharpNN.Infrastructure;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;

namespace CSharpNN.Trainers
{
    internal class SgdTrainer
    {
        private readonly ICostFunction _costFunction;
        private readonly ILogger _logger;
        private readonly double _η;
        private int epochs = 30;

        private int minibatch = 10;

        public SgdTrainer(int epochs, int minibatch, double η, ICostFunction costFunction, ILogger logger)
        {
            this.epochs = epochs;
            this.minibatch = minibatch;
            _η = η;
            _costFunction = costFunction;
            _logger = logger;
        }

        //public static void Shuffle<T>(IList<T> list)
        //{
        //    var provider = new RNGCryptoServiceProvider();
        //    var n = list.Count;
        //    while (n > 1)
        //    {
        //        var box = new byte[1];
        //        do provider.GetBytes(box);
        //        while (!(box[0] < n * (Byte.MaxValue / n)));
        //        var k = (box[0] % n);
        //        n--;
        //        var value = list[k];
        //        list[k] = list[n];
        //        list[n] = value;
        //    }
        //}

        public void Train(Network network, List<double[]> x, List<double[]> y, double trainDataRatio)
        {
            var toBeTaken = (int)(x.Count * trainDataRatio);

            var trainX = x.Take(toBeTaken).ToList();
            var devX = DenseMatrix.OfColumnArrays(x.Skip(toBeTaken));

            var trainY = y.Take(toBeTaken).ToList();
            var devY = DenseMatrix.OfColumnArrays(y.Skip(toBeTaken));

            for (int i = 0; i < epochs; i++)
            {
                _logger.Log($"Epoch {i:D4} started.");
                var randomTrainData = Shuffler.Shuffle(trainX, trainY);

                var trainXBatches = SplitList(randomTrainData.Item1, minibatch).ToList();
                var trainYBatches = SplitList(randomTrainData.Item2, minibatch).ToList();

                for (int j = 0; j < trainXBatches.Count(); j++)
                {
                    var xBatch = trainXBatches[j];
                    var yBatch = trainYBatches[j];

                    var xBatchMatrix = DenseMatrix.OfColumnArrays(xBatch.Select(k => k));
                    var yBatchMatrix = DenseMatrix.OfColumnArrays(yBatch.Select(k => k));

                    var inBatchPred = network.Forward(xBatchMatrix);

                    if (j % 500 == 0)
                    {
                        var inBatchmse = Metrics.GetMse(yBatchMatrix, inBatchPred);
                        var inBatchprecision = Metrics.GetPrecision(yBatchMatrix, inBatchPred);
                        _logger.Log($"                   Batch: {j + 1:D4}/{trainXBatches.Count()}, Mse: {inBatchmse:F3}, Precision: {inBatchprecision:P0}");
                    }

                    Backward(network, yBatchMatrix);
                }

                var pred = network.Forward(devX);

                var mse = Metrics.GetMse(devY, pred);
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

                var δ_biases = δ.RowSums() / δ.ColumnCount;
                var δ_weights = (δ * layer.Input.Transpose()) / δ.ColumnCount;

                gradient = layer.Weights.Transpose() * δ;

                layer.Biases -= _η * δ_biases;
                layer.Weights -= _η * δ_weights;
            }
        }
    }
}