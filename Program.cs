using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Data;
using System.Security.Cryptography.X509Certificates;


namespace HousingPricePrediction
{
    class Program
  {
    static void Main(string[] args)
    {
      var context = new MLContext();

      IDataView dataView = context.Data.LoadFromTextFile<HousingData>("Data/housing.csv", separatorChar: ',', hasHeader: true);

      var pipeline = context.Transforms.Concatenate("Features", new[] { "Rooms", "Size" })
        .Append(context.Transforms.NormalizeMinMax("Features"))
        .Append(context.Regression.Trainers.FastTree(labelColumnName: "Price"));

        var model = pipeline.Fit(dataView);

        var predictionEngine = context.Model.CreatePredictionEngine<HousingData, HousingPrediction>(model);

        var newHouse = new HousingData { Rooms = 3, Size = 1600};
        var prediction = predictionEngine.Predict(newHouse);

        Console.WriteLine($"Predicted price for a house with {newHouse.Rooms} rooms and {newHouse.Size} square feet: {prediction.Price}");

        EvaluateModel(context, model, dataView);
    }

    public static void EvaluateModel(MLContext context, ITransformer model, IDataView data)
    {
        var predictions = model.Transform(data);
        var metrics = context.Regression.Evaluate(predictions, labelColumnName: "Price");

        Console.WriteLine($"R-Squared: {metrics.RSquared}");
        Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
    }
  }

  public class HousingPrediction
  {
    [ColumnName("Score")]
    public float Price { get; set; }
  }
}