using System;
using System.Collections.Generic;
using System.Linq;
using CSharpNN.CostFunctions;
using CSharpNN.Infrastructure;
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

        public static List<T> Shuffle<T>(IEnumerable<T> list, int seed = 13)
        {
            var rng = new Random(seed);

            return list.OrderBy(x => rng.Next()).ToList();
        }

        public void Train(Network network, List<Tuple<double[], double[]>> data, double trainDataRatio)
        {
            var randomData = Shuffle(data);

            var toBeTaken = (int)(randomData.Count * trainDataRatio);

            var trainData = randomData.Take(toBeTaken);
            var devData = randomData.Skip(toBeTaken);

            var xDev = DenseMatrix.OfColumnArrays(devData.Select(j => j.Item1).ToArray());
            var yDev = DenseMatrix.OfColumnArrays(devData.Select(j => j.Item2).ToArray());

            var histogramVector = Metrics.GetHistogramVector(DenseMatrix.OfColumnArrays(trainData.Select(k => k.Item2).ToArray()));

            _logger.Log("Train data histogram");
            _logger.Log(Environment.NewLine + StringCharts.Histogram(histogramVector.AsArray()));

            histogramVector = Metrics.GetHistogramVector(yDev);

            _logger.Log("Dev data histogram");
            _logger.Log(Environment.NewLine + StringCharts.Histogram(histogramVector.AsArray()));

            for (int i = 0; i < epochs; i++)
            {
                _logger.Log($"Epoch {i:D4} started.");
                var randomTrainData = Shuffle(trainData);

                var batches = SplitList(randomTrainData, minibatch).ToList();

                for (int j = 0; j < batches.Count(); j++)
                {
                    var miniList = batches[j];
                    var x = DenseMatrix.OfColumnArrays(miniList.Select(k => k.Item1).ToArray());
                    var y = DenseMatrix.OfColumnArrays(miniList.Select(k => k.Item2).ToArray());

                    var inBatchPred = network.Forward(x);

                    if (j % 500 == 0)
                    {
                        var inBatchmse = Metrics.GetMse(y, inBatchPred);
                        var inBatchprecision = Metrics.GetPrecision(y, inBatchPred);
                        _logger.Log($"                   Batch: {j + 1:D4}/{batches.Count()}, Mse: {inBatchmse:F3}, Precision: {inBatchprecision:P0}");
                    }

                    Backward(network, y);
                }

                var pred = network.Forward(xDev);

                var mse = Metrics.GetMse(yDev, pred);
                var precision = Metrics.GetPrecision(yDev, pred);

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