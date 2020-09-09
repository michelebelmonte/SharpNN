using CSharpNN.CostFunctions;
using CSharpNN.Infrastructure;
using CSharpNN.Trainers;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CSharpNN.Network;

namespace CSharpNN
{
    internal class Program
    {
        private static void DisplayTestPrecision(IEnumerable<double[]> x, IEnumerable<double[]> y, Network.Network network, ILogger logger)
        {
            var xMatrix = DenseMatrix.OfColumnArrays(x);
            var yMatrix = DenseMatrix.OfColumnArrays(y);

            var pred = network.Forward(xMatrix);

            var precision = Metrics.GetPrecision(yMatrix, pred);
            var mse = Metrics.GetMse(yMatrix, pred);
            var confusionMatrix = Metrics.GetConfusionMatrix(yMatrix, pred);
            var histogramVector = Metrics.GetHistogramVector(yMatrix);

            logger.Log();
            logger.Log("Train data histogram:");
            logger.Log("Test data histogram");
            logger.Log(histogramVector.ToVectorString());
            logger.Log("Test data summary");
            logger.Log($"Mse: {mse:F3}");
            logger.Log($"Precision: {precision:P0}");
            logger.Log();
            logger.Log("Confusion matrix:");
            logger.Log(Environment.NewLine + confusionMatrix.ToMatrixString());
        }

        private static string GetDataFolder(string[] args)
        {
            if (args.Length != 1) return string.Empty;

            var folder = args[0];

            if (!Directory.Exists(folder)) return string.Empty;

            return folder;
        }

        private static (IEnumerable<double[]> trainX, IEnumerable<double[]> trainY, IEnumerable<double[]> devX, IEnumerable<double[]> devY) LoadData(string folder, ConsoleLogger logger)
        {
            logger.Log("Loading data...");
            var trainData = MnistReader.ReadTrainingData(folder).ToList();
            var testData = MnistReader.ReadTestData(folder).ToList();
            logger.Log("Loading data done.");

            var trainX = trainData.Select(j => j.x);
            var trainY = trainData.Select(j => j.label);

            var devX = testData.Select(j => j.x);
            var devY = testData.Select(j => j.label);

            return (trainX, trainY, devX, devY);
        }

        private static void Main(string[] args)
        {
            var logger = new ConsoleLogger();

            try
            {
                var dataFolder = GetDataFolder(args);

                if (string.IsNullOrEmpty(dataFolder)) return;

                Control.UseNativeMKL();

                var (trainX, trainY, devX, devY) = LoadData(dataFolder, logger);

                bool repeat;
                do
                {
                    var randomSeed = 13;
                    var network = NetworkBuilder.Build(28 * 28, new LayerOptions(10, new Sigmoid()), new[]
                    {
                        new LayerOptions(30, new Sigmoid()),
                        //new LayerOptions(30, new Sigmoid()),
                        //new LayerOptions(30, new Sigmoid()),
                    }, randomSeed);

                    var trainer = new SgdTrainer(30, 10, 3.0, new QuadraticCostFunction(), logger, randomSeed);

                    var (randomTrainX, randomTrainY) = Shuffler.Shuffle(randomSeed, trainX, trainY);

                    PrintDataHistograms(trainY, devY, logger);

                    trainer.Train(network, randomTrainX, randomTrainY, 0.95);

                    DisplayTestPrecision(devX, devY, network, logger);

                    logger.Log("Press key to exit. \"r\" to repeat...");
                    var answer = Console.ReadKey();

                    repeat = answer.KeyChar == 'r';
                } while (repeat);
            }
            catch (Exception e)
            {
                logger.Log(e.Message);
            }
        }

        private static void PrintDataHistograms(IEnumerable<double[]> trainY, IEnumerable<double[]> devY, ILogger logger)
        {
            var histogramVector = Metrics.GetHistogramVector(DenseMatrix.OfColumnArrays(trainY));

            logger.Log("Train data histogram");
            logger.Log(Environment.NewLine + StringCharts.Histogram(histogramVector.AsArray()));

            histogramVector = Metrics.GetHistogramVector(DenseMatrix.OfColumnArrays(devY));

            logger.Log("Dev data histogram");
            logger.Log(Environment.NewLine + StringCharts.Histogram(histogramVector.AsArray()));
        }
    }
}