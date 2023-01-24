using MathNet.Numerics.LinearAlgebra;

namespace FNN.layers
{
    public abstract class BaseLayer
    {
        public string Name { get; set; }

        public (int, int) Shape { get; set; }

        public (int, int) OutputShape => (Shape.Item2, Shape.Item1);

        public bool Trainable { get; set; }
        
        public abstract int TrainableParams { get; }
        public abstract int NonTrainableParams { get; }
        public abstract int TotalParams { get; }
        
        protected BaseLayer()
        {
        }
        
        public abstract Matrix<double> Call(Matrix<double> input);
        public abstract void Build((int,int) input_shape);

    }
}