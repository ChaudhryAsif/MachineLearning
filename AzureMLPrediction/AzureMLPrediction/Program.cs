using AzureMLPrediction.Model.Sentiment;
using DAL.DbContext;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.ML;
using System;

var host = Host.CreateDefaultBuilder(args)
    .ConfigureServices((context, services) =>
    {
        // Register EF Core with SQL Server
        services.AddDbContext<ApplicationDbContext>(options =>
            options.UseSqlServer("Server=.;Database=SentimentDb;Trusted_Connection=True;"));

        // Register SentimentModel as a service
        services.AddTransient<SentimentModel>();
    })
    .Build();

// Seed data and run the sentiment model
using var scope = host.Services.CreateScope();
var services = scope.ServiceProvider;

// Seed the database with example reviews
SeedReviews.SeedData(services.GetRequiredService<ApplicationDbContext>());

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

var model = services.GetRequiredService<SentimentModel>();
//var model = new SentimentModel();

var (pred1, prob1) = model.Predict("not good");
var (pred2, prob2) = model.Predict("Worst product ever. Not happy at all.");

Console.WriteLine("Review: not good!");
Console.WriteLine($"Sentiment: {(pred1 ? "Positive" : "Negative")} | Confidence: {prob1:P2}\n");

Console.WriteLine("Review: Worst product ever. Not happy at all.");
Console.WriteLine($"Sentiment: {(pred2 ? "Positive" : "Negative")} | Confidence: {prob2:P2}");
