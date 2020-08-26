namespace CSharpNN.Infrastructure
{
    internal interface ILogger
    {
        void Log(string message);

        void Log();
    }
}