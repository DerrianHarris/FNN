using System;

namespace FNN
{
    public class ReLU : IActivationFunction
    {
        public double Call(double x)
        {
            return Math.Max(0, x);
        }
    }
}