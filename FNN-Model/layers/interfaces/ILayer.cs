using MathNet.Numerics.LinearAlgebra.Single;

namespace FNN.layers.interfaces
{
    public interface ILayer
    {
        Matrix Call();
        void Build();
        int TrainableParams { get; }
        int NonTrainableParams { get; }
        int TotalParams { get; }
    }
}