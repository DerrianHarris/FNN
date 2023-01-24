using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace FNN.layers
{
    public abstract class TrainableLayer: BaseLayer
    {
     
        
        
        public string WeightInitializer;
        public string BiasInitializer;
        public string ActivationFunc;
        
        public Matrix<double> Weights;
        public Matrix<double> Biases;
        
        public IActivationFunction ActivationFunction;



        public override int TrainableParams => (Weights.RowCount * Weights.ColumnCount) + Biases.ColumnCount;
        public override int NonTrainableParams => 0;
        public override int TotalParams => TrainableParams + NonTrainableParams;

        protected TrainableLayer(string weightInitializer = "random_normal", string biasInitializer = "zeros", string activationFunc = "") : base()
        {
            this.WeightInitializer = weightInitializer;
            this.BiasInitializer = biasInitializer;
            this.ActivationFunc = activationFunc;
            this.Trainable = true;
        }
        
        public override void Build((int,int) input_shape)
        {
            
            (int, int) weight_shape = (Shape.Item1,input_shape.Item1);

            float stddev = 0.05f;
            float mean = 0.0f;
            int seed = 0;
            
            Weights = (Matrix)Initializers.GetInitializer(WeightInitializer).Invoke(weight_shape,stddev,mean,seed);
            Biases = (Matrix)Initializers.GetInitializer(BiasInitializer).Invoke(Shape);

            ActivationFunction = ActivationFunctions.GetActivationFunctions(ActivationFunc);
        }
    }
}