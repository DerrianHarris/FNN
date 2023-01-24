namespace FNN.layers
{
    public abstract class NonTrainableLayer: BaseLayer
    {
        public override int TrainableParams => 0;
        public override int NonTrainableParams => Shape.Item1;
        public override int TotalParams => TrainableParams + NonTrainableParams;
    }
}