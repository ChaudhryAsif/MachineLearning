using AzureMLPrediction.Model.Sentiment;
using Microsoft.ML;

//var mlContext = new MLContext();

//// Train and save the ML model
//var trainer = new AzureMLPrediction.Model.Sales.SalesTrainer();
//trainer.TrainAndSaveModel();

//// predictions
//var predictor = new AzureMLPrediction.Model.Sales.SalesPredicter();
//predictor.MLPredictionModel();

//// Load the trained model and inspect the schema
//DataViewSchema modelSchema;
//ITransformer model = mlContext.Model.Load("MLModel.zip", out modelSchema);

//Console.WriteLine("Model Input Schema:");
//foreach (var column in modelSchema)
//{
//    Console.WriteLine($"- Name: {column.Name}, Type: {column.Type}");
//}

//Console.WriteLine("Model loaded successfully!");

var model = new SentimentModel();

var (pred1, prob1) = model.Predict("not good");
var (pred2, prob2) = model.Predict("Worst product ever. Not happy at all.");

Console.WriteLine("Review: not good!");
Console.WriteLine($"Sentiment: {(pred1 ? "Positive" : "Negative")} | Confidence: {prob1:P2}\n");

Console.WriteLine("Review: Worst product ever. Not happy at all.");
Console.WriteLine($"Sentiment: {(pred2 ? "Positive" : "Negative")} | Confidence: {prob2:P2}");
