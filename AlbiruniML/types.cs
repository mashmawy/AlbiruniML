using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
   

    public enum ConvDataFormat
    {
        channelsFirst, channelsLast
    }
    public class Conv2DInfo
    {
        public int batchSize;
        public int inHeight;
        public int inWidth;
        public int inChannels;
        public int outHeight;
        public int outWidth;
        public int outChannels;
        public ConvDataFormat dataFormat;
        public int strideHeight;
        public int strideWidth;
        public int dilationHeight;
        public int dilationWidth;
        public int filterHeight;
        public int filterWidth;
        public PadInfo padInfo = new PadInfo();
        public int[] inShape = { 0, 0, 0, 0 };
        public int[] outShape = { 0, 0, 0, 0 };
        public int[] filterShape = { 0, 0, 0, 0 };
    };


    public class computeGradientSliceShapesResutl
    {
        public int[] aBegin;
        public int[] aSize;
        public int[] bBegin;
        public int[] bSize;
    }
    public class SqueezeResult
    {
        public int[] NewShape { get; set; }
        public int[] KeptDims { get; set; }
    }



    public enum roundingMode
    {
        none,
        floor,
        round,
        ceil
    }


    public struct PadInfo
    {
        public int top;
        public int left;
        public int right;
        public int bottom;
        public int alongh;
        public int alongw;
    }


    public struct ValueIndex
    {
        public int index;
        public float value;
    }

    public enum PoolType
    {
        min=0,
        max=1,
        avg=2
    }

     
}



