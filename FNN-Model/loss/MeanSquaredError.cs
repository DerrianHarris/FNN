using System;
using MathNet.Numerics.LinearAlgebra;

namespace FNN
{
    public class MeanSquaredError: ILossFunction
    {
        public Matrix<double> Call(Matrix<double> x, Matrix<double> y)
        {
            Matrix<double> error = y - x;
            Matrix<double> sq_error = error.Map((e)=>Math.Pow(e,2));
            return sq_error;
        }
    }
}