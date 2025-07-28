using Microsoft.ML.Data;

namespace AzureMLPrediction.Model.Sentiment
{
    public class ReviewInput
    {
        [LoadColumn(0)] public bool Label;
        [LoadColumn(1)] public string ReviewText;
    }
}
