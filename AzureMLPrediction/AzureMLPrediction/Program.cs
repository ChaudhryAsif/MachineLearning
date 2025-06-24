using Microsoft.ML;

var mlContext = new MLContext();

// Train and save the ML model
var trainer = new AzureMLPrediction.Model.Sales.SalesTrainer();
trainer.TrainAndSaveModel();

// predictions
var predictor = new AzureMLPrediction.Model.Sales.SalesPredicter();
predictor.MLPredictionModel();

// Load the trained model and inspect the schema
DataViewSchema modelSchema;
ITransformer model = mlContext.Model.Load("MLModel.zip", out modelSchema);

Console.WriteLine("Model Input Schema:");
foreach (var column in modelSchema)
{
    Console.WriteLine($"- Name: {column.Name}, Type: {column.Type}");
}

Console.WriteLine("Model loaded successfully!");
