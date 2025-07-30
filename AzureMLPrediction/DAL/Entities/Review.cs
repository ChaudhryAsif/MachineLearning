namespace DAL.Entities
{
    public class Review
    {
        public int Id { get; set; }
        public bool IsPositive { get; set; }
        public string ReviewText { get; set; } = string.Empty;
    }
}
