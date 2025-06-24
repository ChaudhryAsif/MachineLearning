using Microsoft.ML;

namespace AzureMLPrediction.Model.Sales
{
    public class SalesPredicter
    {
        public void MLPredictionModel()
        {
            var mlContext = new MLContext();

            // Load trained model
            DataViewSchema modelSchema;
            ITransformer mlModel = mlContext.Model.Load("MLModel.zip", out modelSchema);

            // Load training data to evaluate model
            var trainingData = mlContext.Data.LoadFromTextFile<SalesModelTrainingData>(
                path: "F:\\Files\\enhanced_sales_data.csv",
                separatorChar: ',',
                hasHeader: true);

            // Evaluate the model
            var predictions = mlModel.Transform(trainingData);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Sales");

            Console.WriteLine($"R²: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:#.##}");

            // Create prediction engine
            var predEngine = mlContext.Model.CreatePredictionEngine<SalesModelInput, SalesModelOutput>(mlModel);

            // Input sample
            var input = new SalesModelInput
            {
                Day = 7,
                Month = 14,
                Year = 2023,
                Weekday = 1,
                IsHoliday = 0,
                IsWeekend = 0,           // Saturday
                Temp = 0.33368f,
                Humidity = 0.791486f,
                StoreId = 1,
                Region = "West",        // Use one of: North, South, East, West
                Promotion = 0,
                Discount = 0
            };

            // Predict
            var result = predEngine.Predict(input);
            Console.WriteLine($"Predicted Score: {result.Score}");
        }
    }
}
