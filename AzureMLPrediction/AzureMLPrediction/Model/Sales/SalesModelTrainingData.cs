using Microsoft.ML.Data;

namespace AzureMLPrediction.Model.Sales
{
    public class SalesModelTrainingData
    {
        [LoadColumn(0)] public float Day { get; set; }
        [LoadColumn(1)] public float Month { get; set; }
        [LoadColumn(2)] public float Year { get; set; }
        [LoadColumn(3)] public float Weekday { get; set; }
        [LoadColumn(4)] public float IsHoliday { get; set; }
        [LoadColumn(5)] public float IsWeekend { get; set; }
        [LoadColumn(6)] public float Temp { get; set; }
        [LoadColumn(7)] public float Humidity { get; set; }
        [LoadColumn(8)] public float StoreId { get; set; }
        [LoadColumn(9)] public string Region { get; set; }
        [LoadColumn(10)] public float Promotion { get; set; }
        [LoadColumn(11)] public float Discount { get; set; }
        [LoadColumn(12)] public float Sales { get; set; }  // Label
    }
}
