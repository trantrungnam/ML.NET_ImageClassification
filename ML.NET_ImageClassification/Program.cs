using Microsoft.ML;
using Microsoft.ML.Data;
using ML.NET_ImageClassification.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ML.NET_ImageClassification
{
    class Program
    {

        static readonly string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        static readonly string _imagesFolder = Path.Combine(_assetsPath, "images");
        static readonly string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
        static readonly string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
        static readonly string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
        static readonly string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }

        static void Main(string[] args)
        {
            // Tạo một môi trường ML.NET mới.
            MLContext mlContext = new MLContext();
            ITransformer model = GenerateModel(mlContext);
            ClassifySingleImage(mlContext, model);

        }

        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
            }
        }

        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
        {
            return File.ReadAllLines(file)
             .Select(line => line.Split('\t'))
             .Select(line => new ImageData()
             {
                 ImagePath = Path.Combine(folder, line[0])
             });
        }

        // Phương thức dùng để dự doán
        public static void ClassifySingleImage(MLContext mlContext, ITransformer model)
        {
            var imageData = new ImageData()
            {
                ImagePath = _predictSingleImage
            };
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            //Dự đoán cho một hình ảnh
            var prediction = predictor.Predict(imageData);
            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
        }

        public static ITransformer GenerateModel(MLContext mlContext)
        {
            // Thêm các estimator để đọc, resize và trích xuất các pixel từ dữ liệu ảnh đưa vào
            // Dữ liệu cần được xử lý thành các numeric vector (định dạng model của Tensenflow).
            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                // Thêm estimator để lấy model tensorflow và đánh điểm cho chúng.
                .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel)
                // Bước này trong pineline là đưa model của tensorflow vào bộ nhớ, sau đó xử lý các giá trị pixel vector thông qua tensorflow model network 
                .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                // Ánh xạ các chuổi label trong dữ liệu training thành các "giá trị integer key".
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                // Thêm thuật toán của ML.NET để train
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                // Ánh xạ "giá trị integer key" dự doán được trở về chuổi string. 
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            // Option 1: Load dữ liệu training từ text file.
            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);

            // Option 2: Load dữ liệu trainning từ Database
            //DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<ImageData>();
            //string connectionString = @"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=<YOUR-DB-FILEPATH>;Database=<YOUR-DB-NAME>;Integrated Security=True;Connect Timeout=30";
            //string sqlCommand = "SELECT Size, CAST(NumBed as REAL) as NumBed, Price FROM House";
            //DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, sqlCommand);
            //IDataView trainingData = loader.Load(dbSource);

            // Option 3: Load dữ liệu training từ memory
            // IDataView data = mlContext.Data.LoadFromEnumerable<HousingData>(inMemoryCollection);

            // Train the model with the data loaded above
            ITransformer model = pipeline.Fit(trainingData);

            IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
            // Schema validation
            IDataView predictions = model.Transform(testData);
            // Tạo một IEnumerable cho các kết quả dự đoán và hiển thị kết quả.
            IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
            DisplayResults(imagePredictionData);
            // Ước lượng độ chính xác của model.
            // tạo phương thức để đánh giá ước lượng cho model
            // so sánh các giá trị dự doán với tập các dữ liệu test.
            // trả về các số liệu do hiệu suất.
            MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(predictions,
              labelColumnName: "LabelKey",
              predictedLabelColumnName: "PredictedLabel");
            // "Log-loss" càng gần 0 thì dự đoán đó càng chính xác.
            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            // "Per Log-loss" càng gần 0 thì dự đoán đó càng chính xác.
            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");
            return model;
        }
    }
}
