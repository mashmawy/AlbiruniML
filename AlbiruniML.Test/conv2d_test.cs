using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using alb = AlbiruniML.Ops;
using AlbiruniML;
namespace SharpDL.Test
{
    [TestClass]
    public class conv2d_test
    {
        [TestMethod]
        public void conv2dtest1()
        {
            ENV.engine = new Engine();
            var inputDepth = 1;
            int[] inputShape = new int[] { 2, 2, inputDepth };
            var outputDepth = 1;
            var fSize = 1;
            var pad = 0;
            var stride = 1;

            var x = alb.tensor3d(new float[] { 1, 2, 3, 4 },  2, 2, inputDepth);
            var w = alb.tensor4d(new float[] { 2 }, fSize, fSize, inputDepth, outputDepth);

            var result = alb.conv2d(x, w, new int[] { stride, stride }, PadType.number,
                null, roundingMode.none, new Nullable<int>(pad));
             
                     
            Assert.AreEqual(result.Get(0), 2);
            Assert.AreEqual(result.Get(1), 4);
            Assert.AreEqual(result.Get(2), 6);
            Assert.AreEqual(result.Get(3), 8);
        }


        [TestMethod]
        public void conv2dtest2()
        {
            ENV.engine = new Engine();
            var inputDepth = 1;
            int[] inputShape = new int[] { 2, 2, 2, inputDepth };
            var outputDepth = 1;
            var fSize = 1;
            var pad = 0;
            var stride = 1;

            var x = alb.tensor4d(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 2, 2, 2, inputDepth);
            var w = alb.tensor4d(new float[] { 2 }, fSize, fSize, inputDepth, outputDepth);

            var result = alb.conv2d(x, w, new int[] { stride, stride }, PadType.number,
                null, roundingMode.none, new Nullable<int>(pad));


            Assert.AreEqual(result.Shape[0], 2);
            Assert.AreEqual(result.Shape[1], 2);
            Assert.AreEqual(result.Shape[2], 2);
            Assert.AreEqual(result.Shape[3], 1);

            Assert.AreEqual(result.Get(0), 2);
            Assert.AreEqual(result.Get(1), 4);
            Assert.AreEqual(result.Get(2), 6);
            Assert.AreEqual(result.Get(3), 8);

            Assert.AreEqual(result.Get(4), 10);
            Assert.AreEqual(result.Get(5), 12);
            Assert.AreEqual(result.Get(6), 14);
            Assert.AreEqual(result.Get(7), 16);
        }


        [TestMethod]
        public void conv2dtest3()
        {
            ENV.engine = new Engine();
            var inputDepth = 1; 
            var outputDepth = 1;
            var fSize = 2;
            var pad = 0;
            var stride = 1;

            var x = alb.tensor4d(new float[] { 1, 2, 3, 4, },1, 2, 2, inputDepth);
            var w = alb.tensor4d(new float[] { 3, 1, 5, 0 }, fSize, fSize, inputDepth, outputDepth);

            var result = alb.conv2d(x, w, new int[] { stride, stride }, PadType.number,
                new int[] { 1, 1 }, roundingMode.none, new Nullable<int>(pad));


            Assert.AreEqual(result.Get(0), 20);
        }

        [TestMethod]
        public void conv2dtest4()
        {
            ENV.engine = new Engine();
            var inputDepth = 1;
            int[] inputShape = new int[] { 4, 4, inputDepth };
            var outputDepth = 1;
            var fSize = 2;
            var pad = 0;
            var stride = 1;
            var fSizeDilated = 3;
            var dilation = 2;
            var noDilation = 1;

            var x = alb.tensor3d(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, 4, 4, inputDepth);
            var w = alb.tensor4d(new float[] { 3, 1, 5, 2 }, fSize, fSize, inputDepth, outputDepth);


            var wDilated = alb.tensor4d(new float[] { 3, 0, 1, 0, 0, 0, 5, 0, 2 },
                fSizeDilated, fSizeDilated, inputDepth, outputDepth);

            var result = alb.conv2d(x, w, new int[] { stride, stride }, PadType.number,
                new int[] { dilation, dilation }, roundingMode.none, new Nullable<int>(pad));

            var expectedResult = alb.conv2d(x, wDilated, new int[] { stride, stride }, PadType.number,
           new int[] { noDilation, noDilation }, roundingMode.none, new Nullable<int>(pad));


            Assert.AreEqual(result.Shape.Length, expectedResult.Shape.Length);
            for (int i = 0; i < result.Shape.Length; i++)
            {
                Assert.AreEqual(result.Shape[i], expectedResult.Shape[i]);
            }



            Assert.AreEqual(result.dataSync().Length, expectedResult.dataSync().Length);
            for (int i = 0; i < result.dataSync().Length; i++)
            {
                Assert.AreEqual(result.Get(i), expectedResult.Get(i));
            }
        }

        [TestMethod]
        public void conv2dtest5()
        {
            ENV.engine = new Engine();
            var inputDepth = 1;
            int[] inputShape = new int[] { 3, 3, inputDepth };
            var outputDepth = 1; 
            var pad = 0;
            var stride = 1; 
            var filterSize = 2;

            int[] filterShape = new int[] { filterSize, filterSize, inputDepth, outputDepth };
            var filter = alb.ones(filterShape);

            var x = alb.tensor3d(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 3, 3, inputDepth);
            var dy = alb.tensor3d(new float[] { 3, 1, 2, 0 }, 2, 2, 1);

            var grads = alb.grads(
            (Tensor[] inputs) =>
            {
                var xl = inputs[0];
                var filterl = inputs[1] as Tensor;

                var res = xl.conv2d(filterl, new int[] { stride, stride }, PadType.number,
                               new int[] { 1, 1 }, roundingMode.none, new Nullable<int>(pad));
                return res;
            });

            var gres = grads(new Tensor[] { x, filter }, dy);
            var dx = gres[0];
            var dfilter = gres[1];

             
            Assert.AreEqual(x.Shape.Length, dx.Shape.Length);
            for (int i = 0; i < x.Shape.Length; i++)
            {
                Assert.AreEqual(x.Shape[i], dx.Shape[i]);
            }

            float[] expDx = new float[] { 3, 4, 1, 5, 6, 1, 2, 2, 0 };

            for (int i = 0; i < dx.dataSync().Length; i++)
            {
                Assert.AreEqual(dx.Get(i), expDx[i]);
            }



            Assert.AreEqual(filter.Shape.Length, dfilter.Shape.Length);
            for (int i = 0; i < filter.Shape.Length; i++)
            {
                Assert.AreEqual(filter.Shape[i], dfilter.Shape[i]);
            }



            float[] expDfilter = new float[] { 13, 19, 31, 37 };

            for (int i = 0; i < dfilter.dataSync().Length; i++)
            {
                Assert.AreEqual(dfilter.Get(i), expDfilter[i]);
            }


        }

        [TestMethod]
        public void conv2dtest6()
        {
            ENV.engine = new Engine();
            var inputDepth = 1;
            int[] inputShape = new int[] { 3, 3, inputDepth };
            var outputDepth = 1;
            var pad = 0;
            var stride = 1;
            var filterSize = 2;

            int[] filterShape = new int[] { filterSize, filterSize, inputDepth, outputDepth };
            var filter = alb.ones(filterShape);

            var x = alb.tensor4d(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 2, 3, 3, inputDepth);
            var dy = alb.tensor4d(new float[] { 3, 1, 2, 0, 3, 1, 2, 0 },2, 2, 2, 1);

            var grads = alb.grads(
            (Tensor[] inputs) =>
            {
                var xl = inputs[0];
                var filterl = inputs[1] as Tensor;

                var res = xl.conv2d(filterl, new int[] { stride, stride }, PadType.number,
                               new int[] { 1, 1 }, roundingMode.none, new Nullable<int>(pad));
                return res;
            });

            var gres = grads(new Tensor[] { x, filter }, dy);
            var dx = gres[0];
            var dfilter = gres[1];

             
            Assert.AreEqual(x.Shape.Length, dx.Shape.Length);
            for (int i = 0; i < x.Shape.Length; i++)
            {
                Assert.AreEqual(x.Shape[i], dx.Shape[i]);
            }

            float[] expDx = new float[] { 3, 4, 1, 5, 6, 1, 2, 2, 0, 3, 4, 1, 5, 6, 1, 2, 2, 0 };

            for (int i = 0; i < dx.dataSync().Length; i++)
            {
                Assert.AreEqual(dx.Get(i), expDx[i]);
            }



            Assert.AreEqual(filter.Shape.Length, dfilter.Shape.Length);
            for (int i = 0; i < filter.Shape.Length; i++)
            {
                Assert.AreEqual(filter.Shape[i], dfilter.Shape[i]);
            }



            float[] expDfilter = new float[] { 13 * 2, 19 * 2, 31 * 2, 37 * 2 };

            for (int i = 0; i < dfilter.dataSync().Length; i++)
            {
                Assert.AreEqual(dfilter.Get(i), expDfilter[i]);
            }


        }

    }
}
