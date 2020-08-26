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
        private static void DisplayTestPrecision(List<Tuple<double[], double[]>> test, Network network, ILogger logger)
        {
            var x = DenseMatrix.OfColumnArrays(test.Select(j => j.Item1).ToArray());
            var y = DenseMatrix.OfColumnArrays(test.Select(j => j.Item2).ToArray());

            var pred = network.Forward(x);

            var precision = Metrics.GetPrecision(y, pred);
            var mse = Metrics.GetMse(y, pred);
            var confusionMatrix = Metrics.GetConfusionMatrix(y, pred);
            var histogramVector = Metrics.GetHistogramVector(y);

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
            var repeat = false;
            var logger = new ConsoleLogger();
            do
            {
                try
                {
                    Control.UseNativeMKL();

                    logger.Log("Loading data...");
                    var data = MnistReader.ReadTrainingData(@"D:\Development\CSharp\CSharpNN\SharpNN\Data").ToList();
                    var test = MnistReader.ReadTestData(@"D:\Development\CSharp\CSharpNN\SharpNN\Data").ToList();
                    logger.Log("Loading data done.");

                    //var network = new Network(data.First().Item1.Length, 10, new Sigmoid(), new[] { 100, 100, 100, }, new IFunction[] { new Sigmoid(), new Sigmoid(), new Sigmoid(), });
                    var network = new Network(data.First().Item1.Length, 10, new Sigmoid(), new[] { 30 },
                        new IFunction[] { new Sigmoid() });
                    var trainer = new SgdTrainer(30, 10, 3.0, new MseCostFunction(), logger);

                    trainer.Train(network, data, 0.95);

                    DisplayTestPrecision(test, network, logger);
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
    }
}