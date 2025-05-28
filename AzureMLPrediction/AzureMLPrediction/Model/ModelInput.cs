namespace AzureMLPrediction.Model
{
    public class ModelInput
    {
        public float Day { get; set; }
        public float Month { get; set; }
        public float Year { get; set; }
        public float Holiday { get; set; }
        public float Weekday { get; set; }
        public float Temp { get; set; }
        public float Hum { get; set; }
    }

    public class ModelOutput
    {
        public float Score { get; set; }
    }
}
