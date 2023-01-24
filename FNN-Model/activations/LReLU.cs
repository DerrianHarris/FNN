namespace FNN
{
    public class LReLU: IActivationFunction
    {
        public double Call(double x)
        {
            return x >= 0 ? x : x * 0.01f;
        }
    }
}