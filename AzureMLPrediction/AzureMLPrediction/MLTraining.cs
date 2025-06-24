using AzureMLPrediction.Model;
using Microsoft.ML;

namespace AzureMLPrediction
{
    public class MLTraining
    {
        public void TrainAndSaveModel()
        {
            var context = new MLContext();

            string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "enhanced_sales_data.csv");

            // Load training data from a CSV file into an IDataView
            var trainingData = context.Data.LoadFromTextFile<ModelTrainingData>(
                path: dataPath,
                separatorChar: ',',
                hasHeader: true
            );

            // Build the data processing and training pipeline
            var pipeline = context.Transforms
                          .Categorical.OneHotEncoding(outputColumnName: "RegionEncoded", inputColumnName: nameof(ModelTrainingData.Region))
                          .Append(context.Transforms.Concatenate(
                              outputColumnName: "Features",
                              nameof(ModelTrainingData.Day),
                              nameof(ModelTrainingData.Month),
                              nameof(ModelTrainingData.Year),
                              nameof(ModelTrainingData.IsHoliday),
                              nameof(ModelTrainingData.Weekday),
                              nameof(ModelTrainingData.Temp),
                              nameof(ModelTrainingData.Humidity),
                              nameof(ModelTrainingData.IsWeekend),
                              nameof(ModelTrainingData.StoreId),
                              nameof(ModelTrainingData.Promotion),
                              nameof(ModelTrainingData.Discount),
                              "RegionEncoded" // include the encoded string column
                          ))
                          .Append(context.Regression.Trainers.FastTree(
                                 labelColumnName: nameof(ModelTrainingData.Sales),
                                 featureColumnName: "Features",
                                 numberOfLeaves: 50,
                                 numberOfTrees: 200,
                                 minimumExampleCountPerLeaf: 1,
                                 learningRate: 0.2
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
