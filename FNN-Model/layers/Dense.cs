using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace FNN.layers
{
    public class Dense: TrainableLayer
    {
        
        public Dense(int units, string name = "simple_dense",string weightInitializer = "random_normal", string biasInitializer = "zeros", string activationFunc = "") : base(weightInitializer,  biasInitializer, activationFunc)
        {
            this.Name = name;
            this.Shape = (units,1);
        }
        
        public override Matrix<double> Call(Matrix<double> input)
        {
            Matrix<double> output = (Weights * input) + Biases;
            output =  output.Map(ActivationFunction.Call);
            return output;
        }

       
    }
}