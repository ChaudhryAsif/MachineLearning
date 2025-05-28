using AzureMLPrediction.Model;
using Microsoft.ML;

namespace AzureMLPrediction
{
    public class MLPrediction
    {
        public void MLPredictionModel()
        {
            var mlContext = new MLContext();

            // Load model
            DataViewSchema modelSchema;
            ITransformer mlModel = mlContext.Model.Load("MLModel.zip", out modelSchema);

            // Create prediction engine
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Input sample
            var input = new ModelInput
            {
                Day = 5,
                Month = 1,
                Year = 2024,
                Holiday = 0,
                Weekday = 0,
                Temp = 0.36f,
                Hum = 0.69f
            };

            // Predict
            var result = predEngine.Predict(input);
            Console.WriteLine($"Predicted Score: {result.Score}");
        }
    }
}
