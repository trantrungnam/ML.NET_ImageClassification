using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML.NET_ImageClassification.Models
{
    class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }
}
