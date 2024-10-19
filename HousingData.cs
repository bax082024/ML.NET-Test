using Microsoft.ML.Data;


public class HousingData
{
    [LoadColumn(0)]  
    public float Rooms { get; set; }

    [LoadColumn(1)]  
    public float Size { get; set; }

    [LoadColumn(2)]  
    public float Price { get; set; }
}
