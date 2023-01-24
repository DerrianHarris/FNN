using System.Collections.Generic;

namespace FNN
{
    public static class LossFunctions
    {
        private static readonly Dictionary<string, ILossFunction> FuncDict = new Dictionary<string, ILossFunction>()
        {
            {"error", new Error()},
            {"mean_squared_error", new MeanSquaredError()},
        };

        public static ILossFunction GetLossFunctions(string funcString)
        {
            ILossFunction func;
            if (FuncDict.TryGetValue(funcString, out func))
            {
                return func;
            }
            return new Error();
        }
    }
}