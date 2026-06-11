// See https://aka.ms/new-console-template for more information
namespace RandomForest;

public class Forest(int numTrees, int bootStrapSize, int minSamplesSplit = 2, int maxDepth = 6, int maxFeatures = 10)
{
    private readonly int numTrees = numTrees;
    private readonly int bootStrapSize = bootStrapSize;
    private readonly int minSamplesSplit = minSamplesSplit;
    private readonly int maxDepth = maxDepth;
    private readonly int maxFeatures = maxFeatures;
    private readonly List<DecisionTree> trees = [];

    public void Train(LabeledData[] data)
    {
        for(int i = 0; i < numTrees; i++)
        {
            var bootStrapped = Bootstrap(data, bootStrapSize);
            var tree = new DecisionTree(minSamplesSplit, maxDepth, maxFeatures);
            tree.Train(bootStrapped);
            trees.Add(tree);
        }
    }

    public Dictionary<int, double> Predict(LabeledData d)
    {
        Dictionary<int, int> classVotes = [];

        foreach(DecisionTree tree in trees)
        {
            int yPred = tree.Predict(d);
            if (!classVotes.TryGetValue(yPred, out int value))
            {
                value = 0;
                classVotes.Add(yPred, value);
            }

            classVotes[yPred] = ++value;
        }


        Dictionary<int, double> classProbs = [];
        
        foreach(var pair in classVotes)
        {
            classProbs.Add(pair.Key, (double)pair.Value / numTrees);
        }

        return classProbs;
    }

    private static LabeledData[] Bootstrap(LabeledData[] data, int numSamples)
    {
        Random rnd = new();
        List<LabeledData> bootstrappedData = [];

        for(int i = 0; i < numSamples; i++)
        {
            int idx = rnd.Next(data.Length);
            bootstrappedData.Add(data[idx]);
        }

        return [.. bootstrappedData];
    }
}
