using Microsoft.ML.Data;

namespace AzureMLPrediction.Model
{
    public class ModelTrainingData
    {
        [LoadColumn(0)]
        public float Day { get; set; }

        [LoadColumn(1)]
        public float Month { get; set; }

        [LoadColumn(2)]
        public float Year { get; set; }

        [LoadColumn(3)]
        public float Holiday { get; set; }

        [LoadColumn(4)]
        public float Weekday { get; set; }

        [LoadColumn(5)]
        public float Temp { get; set; }

        [LoadColumn(6)]
        public float Hum { get; set; }

        [LoadColumn(7)]
        public float Count { get; set; }
    }
}
