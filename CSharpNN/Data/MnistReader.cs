using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CSharpNN
{
    public static class MnistReader
    {
        private const string TestImages = "t10k-images.idx3-ubyte";
        private const string TestLabels = "t10k-labels.idx1-ubyte";
        private const string TrainImages = "train-images.idx3-ubyte";
        private const string TrainLabels = "train-labels.idx1-ubyte";

        public static int ReadBigInt32(BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static IEnumerable<Tuple<double[], double[]>> ReadTestData(string folder)
        {
            foreach (var item in Read(
                Path.Combine(folder, TestImages),
                Path.Combine(folder, TestLabels)))
            {
                yield return item;
            }
        }

        public static IEnumerable<Tuple<double[], double[]>> ReadTrainingData(string folder)
        {
            foreach (var item in Read(
                Path.Combine(folder, TrainImages),
                Path.Combine(folder, TrainLabels)))
            {
                yield return item;
            }
        }

        private static IEnumerable<Tuple<double[], double[]>> Read(string imagesPath, string labelsPath)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = ReadBigInt32(images);
            int numberOfImages = ReadBigInt32(images);
            int width = ReadBigInt32(images);
            int height = ReadBigInt32(images);

            int magicLabel = ReadBigInt32(labels);
            int numberOfLabels = ReadBigInt32(labels);

            for (int i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height).Select(x => (double)x).Select(x => x / 255.0).ToArray();

                if (bytes.Max() > 1) throw new InvalidOperationException();
                var label = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

                var index = labels.ReadByte();

                label[index] = 1.0;

                yield return new Tuple<double[], double[]>(bytes, label);
            }
        }
    }
}