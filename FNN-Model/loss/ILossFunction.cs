using MathNet.Numerics.LinearAlgebra;

namespace FNN
{
    public interface ILossFunction
    {
        Matrix<double> Call(Matrix<double> x,Matrix<double> y);
    }
}