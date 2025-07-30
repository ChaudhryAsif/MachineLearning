namespace DTO.Extensions
{
    public static class ListExtensions
    {
        public static bool HasAny<T>(this IEnumerable<T> source)
        {
            return source != null && source.Any();
        }
    }
}
