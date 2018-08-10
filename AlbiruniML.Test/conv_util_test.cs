using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AlbiruniML;

namespace SharpDL.Test
{
    [TestClass]
    public class conv_util_test
    {
        [TestMethod]
        public void conv_1x1_over_1x1_array_with_same_pad()
        {
            ENV.engine = new Engine();
            int[] inShape = new int[] { 1, 1, 1, 1 };
            var stride = 1;
            var dilation = 1;
            var convInfo = Util.computeConv2DInfo(
                 inShape, inShape, new int[] { stride, stride }, new int[] { dilation, dilation }, PadType.same);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 1);
            Assert.AreEqual(convInfo.outWidth, 1);
            Assert.AreEqual(convInfo.outChannels, 1);
        }

        [TestMethod]
        public void conv_2x2_over_3x3_array_with_same_pad()
        {
            ENV.engine = new Engine();

            int[] inShape = new int[] { 1, 3, 3, 1 };
            var stride = 1;
            var dilation = 1;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 2, 2, 1, 1 }, new int[] { stride, stride }, new int[] { dilation, dilation }, PadType.same);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 3);
            Assert.AreEqual(convInfo.outWidth, 3);
            Assert.AreEqual(convInfo.outChannels, 1);
            Assert.AreEqual(convInfo.padInfo.left, 0);
            Assert.AreEqual(convInfo.padInfo.right, 1);
            Assert.AreEqual(convInfo.padInfo.top, 0);
            Assert.AreEqual(convInfo.padInfo.bottom, 1);
        }



        [TestMethod]
        public void conv_2x2_over_3x3_array_with_valid_pad()
        {
            ENV.engine = new Engine();

            int[] inShape = new int[] { 1, 3, 3, 1 };
            var stride = 1;
            var dilation = 1;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 2, 2, 1, 1 }, new int[] { stride, stride }, new int[] { dilation, dilation }, PadType.valid);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 2);
            Assert.AreEqual(convInfo.outWidth, 2);
            Assert.AreEqual(convInfo.outChannels, 1);
        }



        [TestMethod]
        public void conv_3x3_over_5x5_array_with_same_pad_with_stride_2()
        {
            ENV.engine = new Engine();

            int[] inShape = new int[] { 1, 5, 5, 1 };
            var stride = 2;
            var dilation = 1;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 3, 3, 1, 1 }, new int[] { stride, stride }, new int[] { dilation, dilation }, PadType.same);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 3);
            Assert.AreEqual(convInfo.outWidth, 3);
            Assert.AreEqual(convInfo.outChannels, 1);

            Assert.AreEqual(convInfo.padInfo.left, 1);
            Assert.AreEqual(convInfo.padInfo.right, 1);
            Assert.AreEqual(convInfo.padInfo.top, 1);
            Assert.AreEqual(convInfo.padInfo.bottom, 1);


        }



        [TestMethod]
        public void conv_2x2_over_3x3_array_with_valid_pad_with_stride_2()
        {
            ENV.engine = new Engine();

            int[] inShape = new int[] { 1, 3, 3, 1 };
            var stride = 2;
            var dilation = 1;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 2, 2, 1, 1 }, new int[] { stride, stride }, new int[] { dilation, dilation }, PadType.valid);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 1);
            Assert.AreEqual(convInfo.outWidth, 1);
            Assert.AreEqual(convInfo.outChannels, 1);
        }




        [TestMethod]
        public void conv_2x1_over_3x3_array_with_valid_pad_with_stride_1()
        {

            ENV.engine = new Engine();
            int[] inShape = new int[] { 1, 3, 3, 1 };
            var stride = 1;
            var dilation = 1;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 2, 1, 1, 1 }, new int[] { stride, stride }, new int[] { dilation, dilation }, PadType.valid);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 2);
            Assert.AreEqual(convInfo.outWidth, 3);
            Assert.AreEqual(convInfo.outChannels, 1);
        }




        [TestMethod]
        public void conv_2x1_over_3x3_array_with_valid_pad_with_strides_h_eq_2_w_eq_1()
        {

            ENV.engine = new Engine();
            int[] inShape = new int[] { 1, 3, 3, 1 };
            var stride = new int[] { 2, 1 };
            var dilation = 1;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 2, 1, 1, 1 }, stride, new int[] { dilation, dilation }, PadType.valid);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 1);
            Assert.AreEqual(convInfo.outWidth, 3);
            Assert.AreEqual(convInfo.outChannels, 1);
        }



        [TestMethod]
        public void conv_1x2_over_3x3_array_with_valid_pad_with_stride_1()
        {

            ENV.engine = new Engine();
            int[] inShape = new int[] { 1, 3, 3, 1 };
            var stride = new int[] { 1, 1 };
            var dilation = 1;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 1, 2, 1, 1 }, stride, new int[] { dilation, dilation }, PadType.valid);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 3);
            Assert.AreEqual(convInfo.outWidth, 2);
            Assert.AreEqual(convInfo.outChannels, 1);
        }



        [TestMethod]
        public void conv_1x2_over_3x3_array_with_valid_pad_with_stride_1_batch_eq_5()
        {

            ENV.engine = new Engine();
            int[] inShape = new int[] { 5, 3, 3, 1 };
            var stride = new int[] { 1, 1 };
            var dilation = 1;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 1, 2, 1, 1 }, stride, new int[] { dilation, dilation }, PadType.valid);
            Assert.AreEqual(convInfo.batchSize, 5);
            Assert.AreEqual(convInfo.outHeight, 3);
            Assert.AreEqual(convInfo.outWidth, 2);
            Assert.AreEqual(convInfo.outChannels, 1);
        }

        [TestMethod]
        public void conv_2x2_over_3x3_array_with_same_pad_with_dilations_2()
        {

            ENV.engine = new Engine();
            int[] inShape = new int[] { 1, 3, 3, 1 };
            var stride = new int[] { 1, 1 };
            var dilation = 2;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 2, 2, 1, 1 }, stride, new int[] { dilation, dilation }, PadType.same);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 3);
            Assert.AreEqual(convInfo.outWidth, 3);
            Assert.AreEqual(convInfo.outChannels, 1);



            Assert.AreEqual(convInfo.padInfo.left, 1);
            Assert.AreEqual(convInfo.padInfo.right, 1);
            Assert.AreEqual(convInfo.padInfo.top, 1);
            Assert.AreEqual(convInfo.padInfo.bottom, 1);
        }

        [TestMethod]
        public void conv_2x1_over_3x3_array_with_same_pad_with_dilations_2()
        {

            ENV.engine = new Engine();
            int[] inShape = new int[] { 1, 3, 3, 1 };
            var stride = new int[] { 1, 1 };
            var dilation = 2;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 2, 1, 1, 1 }, stride, new int[] { dilation, dilation }, PadType.same);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 3);
            Assert.AreEqual(convInfo.outWidth, 3);
            Assert.AreEqual(convInfo.outChannels, 1);



            Assert.AreEqual(convInfo.padInfo.left, 0);
            Assert.AreEqual(convInfo.padInfo.right, 0);
            Assert.AreEqual(convInfo.padInfo.top, 1);
            Assert.AreEqual(convInfo.padInfo.bottom, 1);
        }


        [TestMethod]
        public void conv_3x4_over_8x8_array_with_same_pad_with_dilations_h_eq_4_w_eq_3()
        {

            ENV.engine = new Engine();
            int[] inShape = new int[] { 1, 8, 8, 1 };
            var stride = new int[] { 1, 1 };
            var dilation = new int[] { 4, 3 };
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 3, 4, 1, 1 }, stride, dilation, PadType.same);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 8);
            Assert.AreEqual(convInfo.outWidth, 8);
            Assert.AreEqual(convInfo.outChannels, 1);



            Assert.AreEqual(convInfo.padInfo.left, 4);
            Assert.AreEqual(convInfo.padInfo.right, 5);
            Assert.AreEqual(convInfo.padInfo.top, 4);
            Assert.AreEqual(convInfo.padInfo.bottom, 4);
        }




        [TestMethod]
        public void conv_2x1_over_3x3_array_with_valid_pad_with_dilations_2()
        {

            ENV.engine = new Engine();
            int[] inShape = new int[] { 1, 3, 3, 1 };
            var stride = new int[] { 1, 1 };
            var dilation = 2;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 2, 1, 1, 1 }, stride, new int[] { dilation, dilation }, PadType.valid);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 1);
            Assert.AreEqual(convInfo.outWidth, 3);
            Assert.AreEqual(convInfo.outChannels, 1);
        }



        [TestMethod]
        public void conv_2x2_over_3x3_array_with_valid_pad_with_dilations_2()
        {

            ENV.engine = new Engine();
            int[] inShape = new int[] { 1, 3, 3, 1 };
            var stride = new int[] { 1, 1 };
            var dilation = 2;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 2, 2, 1, 1 }, stride, new int[] { dilation, dilation }, PadType.valid);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 1);
            Assert.AreEqual(convInfo.outWidth, 1);
            Assert.AreEqual(convInfo.outChannels, 1);
        }




        [TestMethod]
        public void conv_2x2_over_4x4_array_with_valid_pad_with_dilations_2()
        {

            ENV.engine = new Engine();
            int[] inShape = new int[] { 1, 4, 4, 1 };
            var stride = new int[] { 1, 1 };
            var dilation = 2;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 2, 2, 1, 1 }, stride, new int[] { dilation, dilation }, PadType.valid);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 2);
            Assert.AreEqual(convInfo.outWidth, 2);
            Assert.AreEqual(convInfo.outChannels, 1);
        }


        /// <summary>
        /// depthwise
        /// 
        
        /// </summary>
        [TestMethod]
        public void depthwise_filter_1x1_over_1x1_array_with_same_pad()
        {

            ENV.engine = new Engine();
            var inChannels = 1;
            var inShape = new int[] { 1, 1, 1, inChannels };
            var fSize = 1;
            var chMul = 1;
            var stride = new int[] { 1, 1 };
            var dilation = new int[] { 1, 1 };
            var pad = PadType.same;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { fSize, fSize, inChannels, chMul }, stride, dilation, pad, roundingMode.none, true);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 1);
            Assert.AreEqual(convInfo.outWidth, 1);
            Assert.AreEqual(convInfo.outChannels, 1);
        }


        [TestMethod]
        public void depthwise_filter_2x2_over_3x3_array_with_same_pad_chMul_3_depth_2()
        {

            ENV.engine = new Engine();
            var inChannels = 2;
            var batchSize = 1;
            var inSize = 3;
            var inShape = new int[] { batchSize, inSize, inSize, inChannels };
            var fSize = 2;
            var chMul = 3;
            var stride = new int[] { 1, 1 };
            var dilation = new int[] { 1, 1 };
            var pad = PadType.same;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { fSize, fSize, inChannels, chMul }, stride, dilation, pad, roundingMode.none, true);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 3);
            Assert.AreEqual(convInfo.outWidth, 3);
            Assert.AreEqual(convInfo.outChannels, 6);
        }


        [TestMethod]
        public void depthwise_filter_2x2_over_3x3_array_with_valid_pad_chMul_3_depth_2()
        {

            ENV.engine = new Engine();
            var inChannels = 2;
            var batchSize = 1;
            var inSize = 3;
            var inShape = new int[] { batchSize, inSize, inSize, inChannels };
            var fSize = 2;
            var chMul = 3;
            var stride = new int[] { 1, 1 };
            var dilation = new int[] { 1, 1 };
            var pad = PadType.valid;
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { fSize, fSize, inChannels, chMul }, stride, dilation, pad, roundingMode.none, true);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 2);
            Assert.AreEqual(convInfo.outWidth, 2);
            Assert.AreEqual(convInfo.outChannels, 6);
        }

        ///conv_util computeConvInfo channelsFirst
        ///

        [TestMethod]
        public void channelsFirst_conv_2x2_over_3x3_array_with_same_pad()
        {
            ENV.engine = new Engine();
              var inDepth = 2;
var outDepth = 4;
int[] inShape = new int[] { 1, inDepth,3,3 };
            var stride = new int[] { 1, 1 };
            var dilation = new int[] { 1, 1 };
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 2, 2, inDepth, outDepth }, stride, dilation, PadType.same
                 , roundingMode.none,false,ConvDataFormat.channelsFirst);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 3);
            Assert.AreEqual(convInfo.outWidth, 3);
            Assert.AreEqual(convInfo.outChannels, 4);

            Assert.AreEqual(convInfo.outShape[0], 1);
            Assert.AreEqual(convInfo.outShape[1], 4);
            Assert.AreEqual(convInfo.outShape[2], 3);
            Assert.AreEqual(convInfo.outShape[3], 3);



            Assert.AreEqual(convInfo.padInfo.left, 0);
            Assert.AreEqual(convInfo.padInfo.right, 1);
            Assert.AreEqual(convInfo.padInfo.top, 0);
            Assert.AreEqual(convInfo.padInfo.bottom, 1);
        }



        [TestMethod]
        public void channelsFirst_conv_2x2_over_3x3_array_with_valid_pad()
        {
            ENV.engine = new Engine();
            var inDepth = 6;
            var outDepth = 16;
            int[] inShape = new int[] { 1, inDepth, 3, 3 };
            var stride = new int[] { 1, 1 };
            var dilation = new int[] { 1, 1 };
            var convInfo = Util.computeConv2DInfo(
                 inShape, new int[] { 2, 2, inDepth, outDepth }, stride, dilation, PadType.valid
                 , roundingMode.none, false, ConvDataFormat.channelsFirst);
            Assert.AreEqual(convInfo.batchSize, 1);
            Assert.AreEqual(convInfo.outHeight, 2);
            Assert.AreEqual(convInfo.outWidth, 2);
            Assert.AreEqual(convInfo.outChannels, 16);


            Assert.AreEqual(convInfo.outShape[0], 1);
            Assert.AreEqual(convInfo.outShape[1], 16);
            Assert.AreEqual(convInfo.outShape[2], 2);
            Assert.AreEqual(convInfo.outShape[3], 2);


            Assert.AreEqual(convInfo.padInfo.left, 0);
            Assert.AreEqual(convInfo.padInfo.right, 0);
            Assert.AreEqual(convInfo.padInfo.top, 0);
            Assert.AreEqual(convInfo.padInfo.bottom, 0);
          
        }



    }
}
