using System.Collections.Generic;

namespace FNN
{
    public static class ActivationFunctions
    {
        private static readonly Dictionary<string, IActivationFunction> FuncDict = new Dictionary<string, IActivationFunction>()
        {
            {"passthrough", new Identity()},
            {"sigmoid", new Sigmoid()},
            {"relu", new ReLU()},
            {"lrelu", new LReLU()}
        };

        public static IActivationFunction GetActivationFunctions(string funcString)
        {
            
            IActivationFunction func;

            if (FuncDict.TryGetValue(funcString, out func))
            {
                return func;
            }
            return new Identity();
        }
    }
}