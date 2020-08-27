using CSharpNN.CostFunctions;
using CSharpNN.Infrastructure;
using CSharpNN.Trainers;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CSharpNN
{
    internal class Program
    {
        private static void DisplayTestPrecision(IEnumerable<double[]> x, IEnumerable<double[]> y, Network network, ILogger logger)
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

        private static void Main(string[] args)
        {
            var logger = new ConsoleLogger();
            bool repeat;
            do
            {
                try
                {
                    Control.UseNativeMKL();

                    logger.Log("Loading data...");
                    var trainData = MnistReader.ReadTrainingData(@"D:\Development\CSharp\CSharpNN\SharpNN\Data").ToList();
                    var testData = MnistReader.ReadTestData(@"D:\Development\CSharp\CSharpNN\SharpNN\Data").ToList();
                    logger.Log("Loading data done.");

                    var trainX = trainData.Select(j => j.Item1);
                    var trainY = trainData.Select(j => j.Item2);

                    var devX = testData.Select(j => j.Item1);
                    var devY = testData.Select(j => j.Item2);

                    var randomData = Shuffler.Shuffle(trainX, trainY);

                    logger.Log("Loading data done.");

                    var randomTrainX = randomData.Item1;
                    var randomTrainY = randomData.Item2;

                    PrintDataHistograms(randomTrainX, randomTrainY, logger);

                    var network = new Network(trainData.First().Item1.Length, new LayerOptions(10, new Sigmoid()), new[] { new LayerOptions(30, new Sigmoid()), });
                    var trainer = new SgdTrainer(30, 10, 3.0, new MseCostFunction(), logger);

                    trainer.Train(network, randomTrainX, randomTrainY, 0.95);

                    DisplayTestPrecision(devX, devY, network, logger);
                }
                catch (Exception e)
                {
                    logger.Log(e.Message);
                }

                logger.Log("Press key to exit. \"r\" to repeat...");
                var answer = Console.ReadKey();

                repeat = answer.KeyChar == 'r';
            } while (repeat);
        }

        private static void PrintDataHistograms(List<double[]> trainY, List<double[]> devY, ILogger logger)
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