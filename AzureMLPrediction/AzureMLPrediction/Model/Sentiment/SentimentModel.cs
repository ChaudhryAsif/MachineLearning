using Microsoft.ML;
using Microsoft.ML.Data;

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

            string projectRoot = Directory.GetParent(AppContext.BaseDirectory)!.Parent!.Parent!.Parent!.FullName;
            string dataPath = Path.Combine(projectRoot, "Data", "reviews.csv");

            if (!File.Exists(dataPath))
            {
                Console.WriteLine($"Error: Training data not found at path: {dataPath}");
                throw new FileNotFoundException($"Training data not found at path: {dataPath}");
            }

            var fullData = _mlContext.Data.LoadFromTextFile<ReviewInput>(
                path: dataPath, hasHeader: true, separatorChar: ',');

            LogDataBalance(fullData);

            var pipeline = _mlContext.Transforms.Text.NormalizeText(inputColumnName: nameof(ReviewInput.ReviewText), outputColumnName: "NormalizedText")
                .Append(_mlContext.Transforms.Text.TokenizeIntoWords(inputColumnName: "NormalizedText", outputColumnName: "Tokens"))
                .Append(_mlContext.Transforms.Text.RemoveDefaultStopWords(inputColumnName: "Tokens", outputColumnName: "TokensWithoutStopWords"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "TokensWithoutStopWords", outputColumnName: "Features"))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: nameof(ReviewInput.Label),
                    featureColumnName: "Features"));

            string modelPath = Path.Combine(projectRoot, "Models", "ReviewSentimentModel.zip"); // Changed .csv to .zip

            IDataView data;

            if (File.Exists(modelPath))
            {
                using var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read);
                _model = _mlContext.Model.Load(stream, out _);
                Console.WriteLine("Loaded existing model from ReviewSentimentModel.zip");

                // If model is loaded, we still need to create the data for evaluation
                var splitData = _mlContext.Data.TrainTestSplit(fullData, testFraction: 0.2, seed: 1);
                data = splitData.TestSet;
            }
            else
            {
                Console.WriteLine("Training new model...");

                var splitData = _mlContext.Data.TrainTestSplit(fullData, testFraction: 0.2, seed: 1);
                var trainingData = splitData.TrainSet;
                data = splitData.TestSet;

                _model = pipeline.Fit(trainingData);

                _mlContext.Model.Save(_model, fullData.Schema, modelPath);
                Console.WriteLine("Trained and saved new model to ReviewSentimentModel.zip");
            }

            // Evaluate the model on the test set, regardless of whether it was trained or loaded
            Console.WriteLine("\n--- Model Evaluation on Test Set (current model) ---");
            var testPredictions = _model.Transform(data);
            var metrics = _mlContext.BinaryClassification.Evaluate(testPredictions, labelColumnName: nameof(ReviewInput.Label));
            LogEvaluationMetrics(metrics);
            Console.WriteLine($"--------------------------------------------------\n");

            _predEngine = _mlContext.Model.CreatePredictionEngine<ReviewInput, ReviewOutput>(_model);
        }

        // Returns both prediction and confidence
        public (bool Prediction, float Confidence) Predict(string reviewText)
        {
            var input = new ReviewInput { ReviewText = reviewText };
            var result = _predEngine.Predict(input);
            Console.WriteLine($"Raw Score for '{reviewText}': {result.Score}");

            float confidenceValue;
            if (result.Prediction)
            {
                confidenceValue = result.Probability;
            }
            else
            {
                confidenceValue = 1 - result.Probability;
            }

            return (result.Prediction, confidenceValue);
        }

        // --- Helper Methods for Logging ---
        private void LogDataBalance(IDataView data)
        {
            var positiveCount = data.GetColumn<bool>(nameof(ReviewInput.Label)).Where(label => label).Count();
            var negativeCount = data.GetColumn<bool>(nameof(ReviewInput.Label)).Where(label => !label).Count();
            Console.WriteLine($"\n--- Data Balance ---");
            Console.WriteLine($"Positive reviews in dataset: {positiveCount}");
            Console.WriteLine($"Negative reviews in dataset: {negativeCount}");
            Console.WriteLine($"--------------------\n");
        }

        private void LogEvaluationMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy on Test Set: {metrics.Accuracy:P2}");
            Console.WriteLine($"Area Under ROC Curve (AUC): {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:P2}");
            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision:P2}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:P2}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:P2}");
        }
    }
}
