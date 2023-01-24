using System;
using System.Collections.Generic;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Random;

namespace FNN
{
    public static class Initializers
    {
        
        public delegate TOut ParamsFunc<TIn, TOut>(params TIn[] args);
        
        private static readonly Dictionary<string, ParamsFunc<object, Matrix<double>>> FuncDict = new Dictionary<string, ParamsFunc<object, Matrix<double>>>()
        {
            {"random_normal", RandomNormal},
            {"zeros", Zeros}
        };
        
        public static ParamsFunc<object, Matrix<double>> GetInitializer(string funcString)
        {
            ParamsFunc<object, Matrix<double>> func;
            if (FuncDict.TryGetValue(funcString, out func))
            {
                return func;
            }
            throw new ArgumentException("Initializer not found: " + funcString);
        }
        
        public static Matrix<double> RandomNormal(object[] args)
        {

            ValueTuple<int, int> shape = (ValueTuple<int, int>) args[0];
            float stddev = (float)args[1];
            float mean = (float)args[2];
            int seed = (int)args[3];
            
            return Matrix.Build.Random(shape.Item1, shape.Item2,
                new Normal(mean, stddev, new SystemRandomSource(seed)));
        }
        
        public static Matrix<double> Zeros(object[] args)
        {
            ValueTuple<int, int> shape = (ValueTuple<int, int>)args[0];
            return Matrix.Build.Dense(shape.Item1, shape.Item2, 0);
        }
    }
}