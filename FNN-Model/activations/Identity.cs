namespace FNN
{
    public class Identity: IActivationFunction
    {
        public double Call(double x)
        {
            return x;
        }
    }
}