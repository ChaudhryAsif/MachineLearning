using Microsoft.ML;

namespace AzureMLPrediction.Model.Sales
{
    public class SalesTrainer
    {
        public void TrainAndSaveModel()
        {
            var context = new MLContext();

            string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "enhanced_sales_data.csv");

            // Load training data from a CSV file into an IDataView
            var trainingData = context.Data.LoadFromTextFile<SalesModelTrainingData>(
                path: dataPath,
                separatorChar: ',',
                hasHeader: true
            );

            // Build the data processing and training pipeline
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
                              "RegionEncoded" // include the encoded string column
                          ))
                          .Append(context.Regression.Trainers.FastTree(
                                 labelColumnName: nameof(SalesModelTrainingData.Sales),
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
