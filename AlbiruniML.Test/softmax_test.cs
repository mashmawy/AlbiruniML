using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using alb = AlbiruniML.Ops;
using AlbiruniML;

namespace SharpDL.Test
{
    [TestClass]
    public class softmax_test
    {
        [TestMethod]
        public void softmax_regular_test()
        {
            ENV.engine = new Engine();
            var y = alb.softmax(alb.tensor1d(new float[] { 2, 1, 3 }));
            var res = new float[] { 0.24472847f, 0.09003057f, 0.66524095f };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(y.dataSync()[i], 5), (float)Math.Round(res[i], 5));
            }
        }
        [TestMethod]
        public void softmax_overflow()
        {
            ENV.engine = new Engine();
            var y = alb.softmax(alb.tensor1d(new float[] { 1000, 1000 }));
            var res = new float[] { 0.49999f, 0.49999f };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(y.dataSync()[i], 5), (float)Math.Round(res[i], 5));
            }
        }
        [TestMethod]
        public void softmax_underflow()
        {
            ENV.engine = new Engine();
            var y = alb.softmax(alb.tensor1d(new float[] { -1000, -1000 }));
            var res = new float[] { 0.49999f, 0.49999f };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(y.dataSync()[i], 5), (float)Math.Round(res[i], 5));
            }
        }
        [TestMethod]
        public void softmax_huge_difference()
        {
            ENV.engine = new Engine();
            var y = alb.softmax(alb.tensor1d(new float[] { -1000, +1000 }));
            var res = new float[] { 0, 1f };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(y.dataSync()[i], 5), (float)Math.Round(res[i], 5));
            }
        }
        [TestMethod]
        public void softmax_Propagates_NAN()
        {
            ENV.engine = new Engine();
            var y = alb.softmax(alb.tensor1d(new float[] { 2, 1, float.NaN }));
            var res = new float[] { float.NaN, float.NaN, float.NaN };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(y.dataSync()[i], 5), (float)Math.Round(res[i], 5));
            }
        }

        [TestMethod]
        public void softmax_2D_dim1()
        {
            ENV.engine = new Engine();
            var y = alb.softmax(alb.tensor2d(new float[] { 2, 1, 3, 1, 3, 2 }, 2, 3), 1);
            var res = new float[] { 0.24472847f, 0.09003057f, 0.66524095f,
                0.09003057f, 0.66524095f, 0.24472847f };
            Assert.AreEqual(y.Rank, 2);
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(y.dataSync()[i], 5), (float)Math.Round(res[i], 5));
            }
        }


        [TestMethod]
        public void softmax_2D_implicit_dim1()
        {
            ENV.engine = new Engine();
            var y = alb.softmax(alb.tensor2d(new float[] { 2, 1, 3, 1, 3, 2 }, 2, 3));
            var res = new float[] { 0.24472847f, 0.09003057f, 0.66524095f,
                0.09003057f, 0.66524095f, 0.24472847f };
            Assert.AreEqual(y.Rank, 2);
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(y.dataSync()[i], 5), (float)Math.Round(res[i], 5));
            }
        }



        [TestMethod]
        public void softmax_1D_gradient()
        {
            ENV.engine = new Engine();
            var x = alb.tensor1d(new float[] { 10, 0, -1 });
            var y = alb.softmax(x);
            var dy = alb.tensor1d(new float[] { 1, 2, 3 });
            var dx = alb.grad((Tensor a) => a.softmax())(x, dy);
            var totalSum = alb.sum(alb.mul(dy, y));
            var res = new float[]{
                     (dy.Get(0) - totalSum.Get()) * y.Get(0),
      (dy.Get(1) - totalSum.Get()) * y.Get(1),
      (dy.Get(2) - totalSum.Get()) * y.Get(2)
            };
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(dx.dataSync()[i], 5), (float)Math.Round(res[i], 5));
            }
        }

        [TestMethod]
        public void softmax_2D_gradient()
        {
            ENV.engine = new Engine();
            var x = alb.tensor2d(new float[] { 10, 0, -1, 5, 4, 3 }, 2, 3);
            var y = alb.softmax(x);
            var dy = alb.tensor2d(new float[] { 3, 2, 1, 1, 2, 3 }, 2, 3);
            var dx = alb.grad((Tensor a) => a.softmax())(x, dy);
            var totalSum = alb.sum(alb.mul(dy, y), new int[] { -1 });
            var res = new float[]{
                     (dy.Get(0, 0) - totalSum.Get(0)) * y.Get(0, 0),
                      (dy.Get(0, 1) - totalSum.Get(0)) * y.Get(0, 1),
                      (dy.Get(0, 2) - totalSum.Get(0)) * y.Get(0, 2),
                      (dy.Get(1, 0) - totalSum.Get(1)) * y.Get(1, 0),
                      (dy.Get(1, 1) - totalSum.Get(1)) * y.Get(1, 1),
                      (dy.Get(1, 2) - totalSum.Get(1)) * y.Get(1, 2)
            };
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(dx.dataSync()[i], 5), (float)Math.Round(res[i], 5));
            }
        }


        [TestMethod]
        public void softmaxCrossEntropy_1D()
        {
            ENV.engine = new Engine();
            var logits = alb.tensor1d(new float[] { 1, 2, 3 });
            var label = alb.tensor1d(new float[] { 0.3f, 0.6f, 0.1f });
            var softmaxLogits = alb.softmax(logits);
            var y = alb.loss.softmaxCrossEntropy(label, logits);

            Assert.AreEqual((float)Math.Round(y.Get(), 3), (float)Math.Round(
        -Math.Log(softmaxLogits.Get(0)) * label.Get(0) +
            -Math.Log(softmaxLogits.Get(1)) * label.Get(1) +
            -Math.Log(softmaxLogits.Get(2)) * label.Get(2), 3));
        }
        [TestMethod]
        public void softmaxCrossEntropy_2D()
        {
            ENV.engine = new Engine();
            var logits = alb.tensor2d(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
            var label = alb.tensor2d(new float[] { 0.3f, 0.6f, 0.1f, 0.2f, 0.3f, 0.5f }, 2, 3);
            var softmaxLogits = alb.softmax(logits);
            var y = alb.loss.softmaxCrossEntropy(label, logits);
            var res = new float[]{
                (float)-Math.Log(softmaxLogits.Get(0, 0)) * label.Get(0, 0) +
           (float)-Math.Log(softmaxLogits.Get(0, 1)) * label.Get(0, 1) +
           (float)-Math.Log(softmaxLogits.Get(0, 2)) * label.Get(0, 2),
       (float)-Math.Log(softmaxLogits.Get(1, 0)) * label.Get(1, 0) +
           (float)-Math.Log(softmaxLogits.Get(1, 1)) * label.Get(1, 1) +
           (float)-Math.Log(softmaxLogits.Get(1, 2)) * label.Get(1, 2)
            };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(y.Get(i), 3), (float)Math.Round(res[i], 3));
            }

        }

        [TestMethod]
        public void softmaxCrossEntropy_2D_dim1()
        {
            ENV.engine = new Engine();
            var logits = alb.tensor2d(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
            var label = alb.tensor2d(new float[] { 0.3f, 0.6f, 0.1f, 0.2f, 0.3f, 0.5f }, 2, 3);
            var softmaxLogits = alb.softmax(logits);
            var y = alb.loss.softmaxCrossEntropy(label, logits, 1);
            var res = new float[]{
                (float)-Math.Log(softmaxLogits.Get(0, 0)) * label.Get(0, 0) +
          (float)-Math.Log(softmaxLogits.Get(0, 1)) * label.Get(0, 1) +
          (float)-Math.Log(softmaxLogits.Get(0, 2)) * label.Get(0, 2),
      (float)-Math.Log(softmaxLogits.Get(1, 0)) * label.Get(1, 0) +
          (float)-Math.Log(softmaxLogits.Get(1, 1)) * label.Get(1, 1) +
          (float)-Math.Log(softmaxLogits.Get(1, 2)) * label.Get(1, 2)
            };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(y.Get(i), 3), (float)Math.Round(res[i], 3));
            }

        }


        [TestMethod]
        public void softmaxCrossEntropy_1D_gradient()
        {
            ENV.engine = new Engine();
            var logits = alb.tensor1d(new float[] { 1, 2, 3 });
            var labels = alb.tensor1d(new float[] { 0.3f, 0.6f, 0.1f });
            var softmaxLogits = alb.softmax(logits);
            var dy = alb.scalar(2);

            var grads = alb.grads(
               (Tensor[] x) => alb.loss.softmaxCrossEntropy(x[0], x[1]));

            var d = grads(new Tensor[] { labels, logits }, dy);
            var dlabels = d[0];
            var dlogits = d[1];
            var dres = new float[]{
                    dy.Get() * (softmaxLogits.Get(0) - labels.Get(0)),
      dy.Get() * (softmaxLogits.Get(1) - labels.Get(1)),
      dy.Get() * (softmaxLogits.Get(2) - labels.Get(2))
            };
            var lres = new float[]{
                     dy.Get() * (labels.Get(0) - softmaxLogits.Get(0)),
      dy.Get() * (labels.Get(1) - softmaxLogits.Get(1)),
      dy.Get() * (labels.Get(2) - softmaxLogits.Get(2))
            };

            for (int i = 0; i < dres.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(dlogits.dataSync()[i], 5), (float)Math.Round(dres[i], 5));
            }


            for (int i = 0; i < lres.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(dlabels.dataSync()[i], 5), (float)Math.Round(lres[i], 5));
            }
        }


        [TestMethod]
        public void softmaxCrossEntropy_2D_gradient()
        {
            ENV.engine = new Engine();
            var logits = alb.tensor2d(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
            var labels = alb.tensor2d(new float[] { 0.3f, 0.6f, 0.1f, .2f, .3f, .5f }, 2, 3);
            var softmaxLogits = alb.softmax(logits);
            var dy = alb.tensor1d(new float[] { 2, 4 });

            var grads = alb.grads(
               (Tensor[] x) => alb.loss.softmaxCrossEntropy(x[0], x[1]));

            var d = grads(new Tensor[] { labels, logits }, dy);
            var dlabels = d[0];
            var dlogits = d[1];
            var dres = new float[]{
 dy.Get(0) * (softmaxLogits.Get(0, 0) - labels.Get(0, 0)),
      dy.Get(0) * (softmaxLogits.Get(0, 1) - labels.Get(0, 1)),
      dy.Get(0) * (softmaxLogits.Get(0, 2) - labels.Get(0, 2)),
      dy.Get(1) * (softmaxLogits.Get(1, 0) - labels.Get(1, 0)),
      dy.Get(1) * (softmaxLogits.Get(1, 1) - labels.Get(1, 1)),
      dy.Get(1) * (softmaxLogits.Get(1, 2) - labels.Get(1, 2))
            };
          
            for (int i = 0; i < dres.Length; i++)
            {
                Assert.AreEqual((float)Math.Round(dlogits.dataSync()[i], 5), (float)Math.Round(dres[i], 5));
            }


             
        }
    }
}
