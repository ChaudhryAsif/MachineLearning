using Microsoft.ML.Data;

namespace AzureMLPrediction.Model.Sentiment
{
    public class ReviewOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
