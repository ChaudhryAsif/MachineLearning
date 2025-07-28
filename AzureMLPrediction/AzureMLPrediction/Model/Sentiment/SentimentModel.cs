using Microsoft.ML;
using System;
using System.IO;

namespace AzureMLPrediction.Model.Sentiment
{
    public class SentimentModel
    {
        private readonly MLContext _mlContext;
        private readonly PredictionEngine<ReviewInput, ReviewOutput> _predEngine;
        private readonly ITransformer _model;

        public SentimentModel()
        {
            _mlContext = new MLContext();

            // Dynamically get the project root path
            string projectRoot = Directory.GetParent(AppContext.BaseDirectory)!.Parent!.Parent!.Parent!.FullName;

            // Get the full path to the data file
            string dataPath = Path.Combine(projectRoot, "Data", "reviews.csv");

            // Check if data file exists
            if (!File.Exists(dataPath))
                throw new FileNotFoundException($"Training data not found at path: {dataPath}");

            // Load training data
            var data = _mlContext.Data.LoadFromTextFile<ReviewInput>(
                path: dataPath, hasHeader: true, separatorChar: ',');

            // Define training pipeline
            var pipeline = _mlContext.Transforms.Text.FeaturizeText("Features", nameof(ReviewInput.ReviewText))
                .Append(_mlContext.BinaryClassification.Trainers.FastTree());

            const string modelPath = "ReviewSentimentModel.zip";

            // Load model if exists; otherwise train and save
            //if (File.Exists(modelPath))
            //{
            //    using var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            //    _model = _mlContext.Model.Load(stream, out _);
            //}
            //else
            //{
                // Train model
                _model = pipeline.Fit(data);

                // Save model
                _mlContext.Model.Save(_model, data.Schema, modelPath);
            //}

            // Create prediction engine
            _predEngine = _mlContext.Model.CreatePredictionEngine<ReviewInput, ReviewOutput>(_model);

#if DEBUG
            var predictions = _model.Transform(data);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions);
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
#endif
        }

        // Returns both prediction and confidence
        public (bool Prediction, float Probability) Predict(string reviewText)
        {
            var input = new ReviewInput { ReviewText = reviewText };
            var result = _predEngine.Predict(input);
            return (result.Prediction, result.Probability);
        }

        // Optional manual evaluation on another dataset
        public double Evaluate(string trainingDataPath)
        {
            var data = _mlContext.Data.LoadFromTextFile<ReviewInput>(
                path: trainingDataPath, hasHeader: true, separatorChar: ',');

            var predictions = _model.Transform(data);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions);
            return metrics.Accuracy;
        }
    }
}
