// See https://aka.ms/new-console-template for more information
namespace RandomForest;

public class LabeledData(int label, double[] data)
{
    public int Label { get; private set; } = label;

    public double this[int i] => data[i];

    public int Length => data.Length;

    private readonly double[] data = data;
}
