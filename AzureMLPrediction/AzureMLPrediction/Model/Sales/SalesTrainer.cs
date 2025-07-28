using Microsoft.ML;
using System;
using System.IO;

namespace AzureMLPrediction.Model.Sales
{
    public class SalesTrainer
    {
        public void TrainAndSaveModel()
        {
            var context = new MLContext();

            // Dynamically get the project root by going up from bin\Debug\net8.0\
            string projectRoot = Directory.GetParent(AppContext.BaseDirectory)!.Parent!.Parent!.Parent!.FullName;

            // Paths relative to project root
            string dataPath = Path.Combine(projectRoot, "Data", "enhanced_sales_data.csv");
            string modelPath = Path.Combine(projectRoot, "Models", "SalesModel.zip");

            // Load training data from CSV
            var trainingData = context.Data.LoadFromTextFile<SalesModelTrainingData>(
                path: dataPath,
                separatorChar: ',',
                hasHeader: true
            );

            // Build the pipeline
            var pipeline = context.Transforms
                .Categorical.OneHotEncoding(outputColumnName: "RegionEncoded", inputColumnName: nameof(SalesModelTrainingData.Region))
                .Append(context.Transforms.Concatenate(
                    outputColumnName: "Features",
                    nameof(SalesModelTrainingData.Day),
                    nameof(SalesModelTrainingData.Month),
                    nameof(SalesModelTrainingData.Year),
                    nameof(SalesModelTrainingData.IsHoliday),
                    nameof(SalesModelTrainingData.Weekday),
                    nameof(SalesModelTrainingData.Temp),
                    nameof(SalesModelTrainingData.Humidity),
                    nameof(SalesModelTrainingData.IsWeekend),
                    nameof(SalesModelTrainingData.StoreId),
                    nameof(SalesModelTrainingData.Promotion),
                    nameof(SalesModelTrainingData.Discount),
                    "RegionEncoded"
                ))
                .Append(context.Regression.Trainers.FastTree(
                    labelColumnName: nameof(SalesModelTrainingData.Sales),
                    featureColumnName: "Features",
                    numberOfLeaves: 50,
                    numberOfTrees: 200,
                    minimumExampleCountPerLeaf: 1,
                    learningRate: 0.2
                ));

            // Train model
            var trainedModel = pipeline.Fit(trainingData);

            // Ensure Models folder exists
            Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);

            // Save model
            using var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write);
            context.Model.Save(trainedModel, trainingData.Schema, fileStream);

            Console.WriteLine("✅ Sales model trained and saved to: " + modelPath);
        }
    }
}
