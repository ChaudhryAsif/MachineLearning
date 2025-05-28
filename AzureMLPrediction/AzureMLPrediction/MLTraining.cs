using AzureMLPrediction.Model;
using Microsoft.ML;

namespace AzureMLPrediction
{
    public class MLTraining
    {
        public void TrainAndSaveModel()
        {
            var context = new MLContext();

            // Load training data from a CSV file into an IDataView
            var trainingData = context.Data.LoadFromTextFile<ModelTrainingData>(
                path: "F:\\Files\\sample_data.csv",
                separatorChar: ',',
                hasHeader: true
            );

            // Build the data processing and training pipeline
            var pipeline = context.Transforms
                .Concatenate(
                    outputColumnName: "Features",
                    nameof(ModelTrainingData.Month),
                    nameof(ModelTrainingData.Year),
                    nameof(ModelTrainingData.Holiday),
                    nameof(ModelTrainingData.Weekday),
                    nameof(ModelTrainingData.Temp),
                    nameof(ModelTrainingData.Hum)
                )
                // Append regression trainer (FastTree is a gradient-boosted decision tree algorithm)
                .Append(context.Regression.Trainers.FastTree(
                    labelColumnName: nameof(ModelTrainingData.Count),
                    featureColumnName: "Features"
                ));

            // Train the model using the pipeline
            var trainedModel = pipeline.Fit(trainingData);

            // Save the trained model to a .zip file
            using (var fileStream = new FileStream("MLModel.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
                context.Model.Save(trainedModel, trainingData.Schema, fileStream);

            Console.WriteLine("Model has been trained and saved to MLModel.zip");
        }
    }
}
