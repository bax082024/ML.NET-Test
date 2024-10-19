using Microsoft.ML;
using Microsoft.ML.Data;
using System;


namespace HousingPricePrediction
{



    class Program
  {
    static void Main(string[] args)
    {
      var context = new MLContext();

      IDataView dataView = content.Data.LoadFromTextFile<HousingData>("Data/housing.csv", separatorChar: ',', hashHeader: true);




    }
  }
}