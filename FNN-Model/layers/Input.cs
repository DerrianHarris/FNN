using System;
using MathNet.Numerics.LinearAlgebra;

namespace FNN.layers
{
    public class Input: NonTrainableLayer
    {
        public Input(ValueTuple<int,int> inputShape, string name = "input") : base()
        {
            this.Name = name;
            this.Shape = inputShape;
            this.Trainable = false;
        }

        public override Matrix<double> Call(Matrix<double> input)
        {
            return input;
        }

        public override void Build((int, int) input_shape)
        {
        }
    }
}