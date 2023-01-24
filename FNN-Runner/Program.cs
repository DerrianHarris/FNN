using System;

using FNN;
using FNN.layers;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


namespace FNN_Runner
{
    class Program
    {
        static void Main(string[] args)
        {

            Matrix<double>[] features = new[]
            {
                Matrix.Build.DenseOfRowMajor(2, 1, new double[] {0, 0}),
                Matrix.Build.DenseOfRowMajor(2, 1, new double[] {0, 1}),
                Matrix.Build.DenseOfRowMajor(2, 1, new double[] {1, 0}),
                Matrix.Build.DenseOfRowMajor(2, 1, new double[] {1, 1})
            };
            
            Matrix<double>[] labels = new[]
            {
                Matrix.Build.DenseOfRowMajor(1, 1, new double[] {0}),
                Matrix.Build.DenseOfRowMajor(1, 1, new double[] {1}),
                Matrix.Build.DenseOfRowMajor(1, 1, new double[] {1}),
                Matrix.Build.DenseOfRowMajor(1, 1, new double[] {0}),
            };
            
            Model model = new Model();

            model.Add(new Input((2,1)));
            model.Add(new Dense(units:4,activationFunc:"lrelu"));
            model.Add(new Dense(units:1,activationFunc:"lrelu"));
            model.Compile(lossFunction:"error",learningRate:0.01f);
            Console.Out.WriteLine(model.Summary());
            
            model.Fit(features,labels,10000);
            Matrix<double>[] predictions = model.Predict(features);
            
            for (int i = 0; i < predictions.Length; i++)
            {
                Console.Out.WriteLine("------------------------------------");
                Console.Out.WriteLine("Feature: \n{0} \nLabel: \n{1} \nPrediction: \n{2}",features[i].ToMatrixString(),labels[i].ToMatrixString(),predictions[i].ToMatrixString());
                Console.Out.WriteLine("------------------------------------");
            }
        }
    }
}