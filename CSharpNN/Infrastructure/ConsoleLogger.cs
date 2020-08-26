using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CSharpNN.Infrastructure
{
    internal class ConsoleLogger : ILogger
    {
        public void Log(string message)
        {
            Console.WriteLine($"{DateTime.Now:s} - {message}");
        }

        public void Log()
        {
            Console.WriteLine();
        }
    }

    internal class StringCharts
    {
        public static string Histogram(IEnumerable<double> values, char symbol = '-', int maxLength = 50)
        {
            var max = values.Max();

            var normalized = values.Select(x => x / max);

            var stringBuilder = new StringBuilder();

            foreach (var d in normalized)
            {
                var size = (int)(d * maxLength);

                var line = new string(symbol, size);

                stringBuilder.AppendLine(line);
            }

            return stringBuilder.ToString();
        }
    }
}