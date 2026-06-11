// See https://aka.ms/new-console-template for more information
namespace RandomForest;

public class DecisionNode
{
    public int Index;
    public double Threshold;
    public double InfoGain;
    public DecisionNode? LeftNode;
    public DecisionNode? RightNode;
    public int Value;
    public bool IsLeaf;
}
