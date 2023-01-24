using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FNN.layers;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace FNN
{
    public class Model
    {
        public float LearningRate;
        private List<BaseLayer> _layers;
        private Matrix<double>[] _activations;
        
        protected ILossFunction LossFunction;
        private string lossFunc;

        public Model()
        {
            _layers = new List<BaseLayer>();
        }

        public Model(List<BaseLayer> layers)
        {
            this._layers = layers;
        }

        public void Fit(Matrix<double>[] features,Matrix<double>[] labels, int epochs = 1)
        {
            Random rand = new Random();
            int seed = rand.Next();

            Matrix<double>[] features_shuffled = Shuffle(features, seed);
            Matrix<double>[] labels_shuffled = Shuffle(labels, seed);
            

            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < features.Length; j++)
                {
                    Matrix<double> input = features_shuffled[j];
                    Matrix<double> label = labels_shuffled[j];

                    feedForward(input);
                    backPropagation(label);
                }
            }
            
        }
        
        public Matrix<double>[] Evaluate(Matrix<double>[] inputs)
        {
            return null;
        }
        
        public Matrix<double>[] Predict(Matrix<double>[] inputs)
        {
            Matrix<double>[] outputs = new Matrix<double>[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                outputs[i] = feedForward(inputs[i]);
            }
            return outputs;
        }
        
        public void Add(BaseLayer layer)
        {
            this._layers.Add(layer);
        }
        
        public void Compile(float learningRate = 0.01f,string lossFunction = "")
        {
            (int,int) prevLayer = (0,0);
            
            foreach (BaseLayer layer in _layers)
            {
                layer.Build(prevLayer);
                prevLayer = layer.Shape;
            }

            this.lossFunc = lossFunction;
            this.LossFunction = LossFunctions.GetLossFunctions(lossFunc);

            this._activations = new Matrix<double>[_layers.Count];
            this.LearningRate = learningRate;
        }

        public string Summary()
        {
            StringBuilder result = new StringBuilder();
            result.Append("\n---------------------- Model Summary ----------------------");
            result.AppendFormat("\n {0,15}{1,20}{2,15}", "Layer Name","Output Shape","Params #");
            result.Append("\n-----------------------------------------------------------");
            int count = 0;
            foreach (BaseLayer layer in _layers)
            {
                result.AppendFormat("\n {0,-5}{1,-20}{2,-20}{3,-10}", count,layer.Name, layer.OutputShape,layer.TotalParams);
                count++;
            }

            return result.ToString();
        }
        
        private Matrix<double> feedForward(Matrix<double> input)
        {
            Matrix<double> output = input;
            for (int i = 0; i < _layers.Count; i++)
            {
                output = _layers[i].Call(output);
                _activations[i] = output;
            }
            return output;
        }

        private void backPropagation(Matrix<double> inputLabels)
        {
            
            Matrix<double>[] errors = new Matrix<double>[_layers.Count+1];
            Matrix<double>[] weightDeltas = new Matrix<double>[_layers.Count];

            TrainableLayer outputLayer = (TrainableLayer) _layers[^1];
            
            errors[^1] = _activations[^1] - inputLabels;

            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                if (_layers[i].Trainable)
                {
                    TrainableLayer layer = (TrainableLayer) _layers[i];
                    Matrix<double> prevError = errors[i + 1];
                    
                    weightDeltas[i] = prevError.PointwiseMultiply(_activations[i].Map(x => Differentiate.FirstDerivative(layer.ActivationFunction.Call,x)));
                    errors[i] = layer.Weights.Transpose() * weightDeltas[i];
                }
                else
                {
                    errors[i] = errors[i + 1];
                    weightDeltas[i] = weightDeltas[i + 1];
                }
            }

            for (int i = 0; i < _layers.Count; i++)
            {
                if (_layers[i].Trainable)
                {
                    TrainableLayer layer = (TrainableLayer) _layers[i];
                    Matrix<double> deltaBiases = weightDeltas[i] * LearningRate;
                    Matrix<double> deltaWeight =   deltaBiases * _activations[i-1].Transpose();
                    
                    layer.Weights -= deltaWeight;
                    layer.Biases -= deltaBiases;
                }
            }
        }

        public T[] Shuffle<T>(T[] arr, int seed)
        {
            Random random = new Random(seed);
            return arr.OrderBy(x => random.Next()).ToArray();
        }
    }
}