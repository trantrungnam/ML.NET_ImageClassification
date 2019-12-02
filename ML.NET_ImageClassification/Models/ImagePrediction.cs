using System;
using System.Collections.Generic;
using System.Text;

namespace ML.NET_ImageClassification.Models
{
    class ImagePrediction: ImageData
    {
        public float[] Score;

        public string PredictedLabelValue;
    }
}
