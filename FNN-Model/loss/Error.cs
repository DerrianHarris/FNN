using MathNet.Numerics.LinearAlgebra;

namespace FNN
{
    public class Error: ILossFunction
    {
        public Matrix<double> Call(Matrix<double> x, Matrix<double> y)
        {
            return y - x;
        }
    }
}