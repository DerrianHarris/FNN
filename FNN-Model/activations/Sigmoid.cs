using System;

namespace FNN
{
    public class Sigmoid: IActivationFunction
    {
        public double Call( double x )
        {
            return 1/(1+ Math.Exp(-x));
        }
    }
}