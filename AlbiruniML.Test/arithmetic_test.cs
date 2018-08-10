using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using alb = AlbiruniML.Ops;
using AlbiruniML;
namespace SharpDL.Test
{
    [TestClass]
    public class arithmetic_test
    {
        [TestMethod]
        public void div_same_shape()
        {
            var a = alb.tensor2d(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
            var c = alb.tensor2d(new float[] { 1, 2, 3, 4, 2, 5 }, 2, 3);

            var r = alb.div(a, c).dataSync();
            float[] res = new float[] { 1, 1, 1, 1, 2.5f, 6 / 5f };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }
        [TestMethod]
        public void div_broadcasts()
        {
            var a = alb.tensor1d(new float[] { -5, -4, 3, 2 });
            var c = alb.scalar(2);

            var r = alb.div(a, c).dataSync();
            float[] res = new float[] { -2.5f, -2, 1.5f, 1 };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }
        [TestMethod]
        public void div_broadcasting_same_rank_Tensors_different_shape()
        {
            var a = alb.tensor2d(new float[] { 1, 2, -3, -4 }, 2, 2);
            var c = alb.tensor2d(new float[] { 2, 3 }, 2, 1);

            var r = alb.div(a, c);
            var expected = r.dataSync();

            float[] res = new float[] { 1f / 2f, 1, -1, -4f / 3f };
            int[] shape = alb.shape(2, 2);// new int[] { 2, 2 };

            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(r.Shape[i], shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(expected[i], res[i]);
            }
        }
        [TestMethod]
        public void div_broadcasting_2D_1D()
        {
            var a = alb.tensor2d(new float[] { 1, 2, -3, -4 }, 2, 2);
            var c = alb.tensor1d(new float[] { 1, 2 });

            var r = alb.div(a, c);
            var expected = r.dataSync();

            float[] res = new float[] { 1, 1, -3, -2 };
            int[] shape = new int[] { 2, 2 };

            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(r.Shape[i], shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(expected[i], res[i]);
            }
        }
        [TestMethod]
        public void div_scalar_divided_by_array()
        {
            var a = alb.scalar(2);
            var c = alb.tensor2d(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);

            var r = alb.div(a, c).dataSync();
            float[] res = new float[] { 2f / 1f, 2f / 2f, 2f / 3f, 2f / 4f, 2f / 5f, 2f / 6f };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }
        [TestMethod]
        public void div_array_divided_by_scalar()
        {
            var c = alb.scalar(2);
            var a = alb.tensor2d(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);

            var r = alb.div(a, c).dataSync();
            float[] res = new float[] { 1f / 2f, 2f / 2f, 3f / 2f, 4f / 2f, 5f / 2f, 6f / 2f };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }
        [TestMethod]
        public void div_gradient_scalar()
        {
            var a = alb.scalar(5);
            var b = alb.scalar(2);
            var dy = alb.scalar(4);

            var grads = alb.grads((Tensor[] x) => { return x[0] / x[1]; });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];

            Assert.AreEqual(da.Get(0), 4f / 2f);
            Assert.AreEqual(db.Get(0), -4f * 5f / (2f * 2f));

        }
        [TestMethod]
        public void div_gradient_tensor1d()
        {
            var a = alb.tensor1d(new float[] { 1, 2, 3 });
            var b = alb.tensor1d(new float[] { 3, 4, 5 });
            var dy = alb.tensor1d(new float[] { 1, 10, 20 });

            var grads = alb.grads((Tensor[] x) => { return x[0]/x[1]; });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 1f / 3f, 10f / 4f, 20f / 5f };
            var dbRes = new float[] { -1f * 1f / 9f, -10f * 2f / 16f, -20f * 3f / 25f };

            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }

        [TestMethod]
        public void div_gradient_tensor2d()
        {
            var a = alb.tensor2d(new float[] { 3, 1, 2, 3 }, 2, 2);
            var b = alb.tensor2d(new float[] { 1, 3, 4, 5 }, 2, 2);
            var dy = alb.tensor2d(new float[] { 1, 10, 15, 20 }, 2, 2);

            var grads = alb.grads((Tensor[] x) => { return x[0]/x[1]; });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 1f / 1f, 10f / 3f, 15f / 4f, 20f / 5f };
            var dbRes = new float[] { -1f * 3f / 1f, -10f * 1f / 9f, -15f * 2f / 16f, -20f * 3f / 25f };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }

        [TestMethod]
        public void div_gradient_scalar_tensor1d()
        {
            var a = alb.scalar(2);
            var b = alb.tensor1d(new float[] { 3, 4, 5 });
            var dy = alb.tensor1d(new float[] { 6, 7, 8 });

            var grads = alb.grads((Tensor[] x) => { return x[0]/x[1]; });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 6f / 3f + 7f / 4f + 8f / 5f };
            var dbRes = new float[] { -6f * 2f / 9f, -7f * 2f / 16f, -8f * 2f / 25f };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }

        [TestMethod]
        public void div_gradient_tensor2d_scalar()
        {
            var a = alb.tensor2d(new float[] { 2, 3, 4, 5 }, 2, 2);

            var b = alb.scalar(2);
            var dy = alb.tensor2d(new float[] { 6, 7, 8, 9 }, 2, 2);

            var grads = alb.grads((Tensor[] x) => { return x[0]/x[1]; });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 6f / 2f, 7f / 2f, 8f / 2f, 9f / 2f };
            var dbRes = new float[] { -6f * 2f / 4f + -7f * 3f / 4f + -8f * 4f / 4f + -9f * 5f / 4f };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }

        [TestMethod]
        public void div_gradient_tensor2d_tensor2d_broadcast()
        {
            var a = alb.tensor2d(new float[] { 3, 4 }, 2, 1);

            var b = alb.tensor2d(new float[] { 2, 3, 4, 5 }, 2, 2);
            var dy = alb.tensor2d(new float[] { 6, 7, 8, 9 }, 2, 2);

            var grads = alb.grads((Tensor[] x) => { return x[0]/x[1]; });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 6f / 2f + 7f / 3f, 8f / 4f + 9f / 5f };
            var dbRes = new float[] { -6f * 3f / 4f, -7f * 3f / 9f, -8f * 4f / 16f, -9f * 4f / 25f };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual((float)Math.Round(da.Get(i), 5), (float)Math.Round(daRes[i], 5));
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }

        [TestMethod]
        public void mul_same_shape()
        {
            var a = alb.tensor2d(new float[] { 1, 2, -3, -4 }, 2, 2);
            var c = alb.tensor2d(new float[] { 5, 3, 4, -7 }, 2, 2);

            var r = alb.mulStrict(a, c);
            float[] res = new float[] { 5, 6, -12, 28 };
            int[] shape = new int[] { 2, 2 };

            Assert.AreEqual(r.Shape.Length, shape.Length);
            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(r.Shape[i], shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }
        [TestMethod]
        public void mul_broadcasting()
        {
            var a = alb.tensor2d(new float[] { 1, 2, -3, -4 }, 2, 2);
            var b = alb.scalar(2);

            var r = a*b;
            float[] res = new float[] { 2, 4, -6, -8 };
            int[] shape = new int[] { 2, 2 };

            Assert.AreEqual(r.Shape.Length, shape.Length);
            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(r.Shape[i], shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }

        [TestMethod]
        public void mul_broadcasting_same_rank_Tensors_different_shape()
        {
            var a = alb.tensor2d(new float[] { 1, 2, -3, -4 }, 2, 2);
            var b = alb.tensor2d(new float[] { 2, 3 }, 2, 1);

            var r = a*b;
            float[] res = new float[] { 2, 4, -9, -12 };
            int[] shape = new int[] { 2, 2 };

            Assert.AreEqual(r.Shape.Length, shape.Length);
            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(r.Shape[i], shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }
        [TestMethod]
        public void mul_broadcasting_2D_1D()
        {
            var a = alb.tensor2d(new float[] { 1, 2, -3, -4 }, 2, 2);
            var b = alb.tensor1d(new float[] { 1, 2 });

            var r = a*b;
            float[] res = new float[] { 1, 4, -3, -8 };
            int[] shape = new int[] { 2, 2 };

            Assert.AreEqual(r.Shape.Length, shape.Length);
            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(r.Shape[i], shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }

        [TestMethod]
        public void mul_gradient_scalar()
        {
            var a = alb.scalar(5);
            var b = alb.scalar(2);
            var dy = alb.scalar(4);

            var grads = alb.grads((Tensor[] x) => { return x[0]*x[1]; });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];

            Assert.AreEqual(da.Get(0), b.Get() * dy.Get());
            Assert.AreEqual(db.Get(0), a.Get() * dy.Get());

        }

        [TestMethod]
        public void mul_gradient_tensor1d()
        {
            var a = alb.tensor1d(new float[] { 1, 2, 3 });
            var b = alb.tensor1d(new float[] { 3, 4, 5 });
            var dy = alb.tensor1d(new float[] { 1, 10, 20 });

            var grads = alb.grads((Tensor[] x) => { return x[0]*x[1]; });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 3 * 1, 4 * 10, 5 * 20 };
            var dbRes = new float[] { 1 * 1, 2 * 10, 3 * 20 };

            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }


        [TestMethod]
        public void mul_gradient_tensor2d()
        {
            var a = alb.tensor2d(new float[] { 3, 1, 2, 3 }, 2, 2);
            var b = alb.tensor2d(new float[] { 1, 3, 4, 5 }, 2, 2);
            var dy = alb.tensor2d(new float[] { 1, 10, 15, 20 }, 2, 2);

            var grads = alb.grads((Tensor[] x) => { return x[0]*x[1]; });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 1 * 1, 3 * 10, 4 * 15, 5 * 20 };
            var dbRes = new float[] { 3 * 1, 1 * 10, 2 * 15, 3 * 20 };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }


        [TestMethod]
        public void mul_gradient_scalar_tensor1d()
        {
            var a = alb.scalar(2);
            var b = alb.tensor1d(new float[] { 3, 4, 5 });
            var dy = alb.tensor1d(new float[] { 6, 7, 8 });

            var grads = alb.grads((Tensor[] x) => { return x[0]*x[1]; });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 3 * 6 + 4 * 7 + 5 * 8 };
            var dbRes = new float[] { 2 * 6, 2 * 7, 2 * 8 };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }

        [TestMethod]
        public void mul_gradient_tensor2d_scalar()
        {
            var a = alb.tensor2d(new float[] { 2, 3, 4, 5 }, 2, 2);

            var b = alb.scalar(2);
            var dy = alb.tensor2d(new float[] { 6, 7, 8, 9 }, 2, 2);

            var grads = alb.grads((Tensor[] x) => { return x[0]*x[1]; });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 2 * 6, 2 * 7, 2 * 8, 2 * 9 };
            var dbRes = new float[] { 2 * 6 + 3 * 7 + 4 * 8 + 5 * 9 };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }


        [TestMethod]
        public void mul_gradient_tensor2d_tensor2d_broadcast()
        {
            var a = alb.tensor2d(new float[] { 3, 4 }, 2, 1);

            var b = alb.tensor2d(new float[] { 2, 3, 4, 5 }, 2, 2);
            var dy = alb.tensor2d(new float[] { 6, 7, 8, 9 }, 2, 2);

            var grads = alb.grads((Tensor[] x) => { return x[0]*x[1]; });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 2 * 6 + 3 * 7, 4 * 8 + 5 * 9 };
            var dbRes = new float[] { 6 * 3, 7 * 3, 8 * 4, 9 * 4 };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual((float)Math.Round(da.Get(i), 5), (float)Math.Round(daRes[i], 5));
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }

        [TestMethod]
        public void pow_same_shape()
        {
            var a = alb.tensor2d(new float[] { 1, -2, -3, 0, 7, 1 }, 2, 3);
            var c = alb.tensor2d(new float[] { 5, 3, 4, 5, 2, -3 }, 2, 3);

            var r = alb.pow(a, c).dataSync();
            float[] res = new float[] { 1, -8, 81, 0, 49, 1 };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(Math.Ceiling(r[i]), Math.Ceiling(res[i]));
            }
        }

        [TestMethod]
        public void pow_different_shape()
        {
            var a = alb.tensor2d(new float[] { 1, -2, -3, 0, 7, 1 }, 2, 3);
            var c = alb.scalar(2);

            var r = alb.pow(a, c).dataSync();
            float[] res = new float[] { 1, 4, 9, 0, 49, 1 };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(System.Math.Ceiling(r[i]), System.Math.Ceiling(res[i]));
            }
        }


        [TestMethod]
        public void pow_non_int32_exponent()
        {
            var a = alb.tensor1d(new float[] { 2, 4 });
            var c = alb.tensor1d(new float[] { .5f, 1.2f });

            var r = alb.pow(a, c).dataSync();
            float[] res = new float[] { (float)Math.Pow(2, 0.5f), (float)Math.Pow(4, 1.2f) };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }


        [TestMethod]
        public void pow_broadcasting_same_rank_Tensors_different_shape()
        {
            var a = alb.tensor2d(new float[] { 1, 2, -3, -4 }, 2, 2);
            var b = alb.tensor2d(new float[] { 2, 1 }, 2, 1);

            var r = alb.pow(a, b);
            float[] res = new float[] { 1, 4, -3, -4 };
            int[] shape = new int[] { 2, 2 };

            Assert.AreEqual(r.Shape.Length, shape.Length);
            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(r.Shape[i], shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }

        [TestMethod]
        public void pow_broadcasting_2D_1D()
        {
            var a = alb.tensor2d(new float[] { 1, 2, -3, -4 }, 2, 2);
            var b = alb.tensor1d(new float[] { 1, 2 });

            var r = alb.pow(a, b);
            float[] res = new float[] { 1, 4, -3, 16 };
            int[] shape = new int[] { 2, 2 };

            Assert.AreEqual(r.Shape.Length, shape.Length);
            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(r.Shape[i], shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }


        [TestMethod]
        public void pow_gradient_scalar()
        {
            var a = alb.scalar(5);
            var b = alb.scalar(2);
            var dy = alb.scalar(3);

            var grads = alb.grads((Tensor[] x) => { return alb.pow(x[0], x[1]); });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];

            Assert.AreEqual(Math.Ceiling( da.Get(0)),Math.Ceiling( 2f * 5f * 3f));
            Assert.AreEqual((float)Math.Round(db.Get(0), 4), (float)Math.Round(
                3f * (float)Math.Pow(5, 2) * (float)Math.Log(5), 4));

        }

        [TestMethod]
        public void pow_gradient_scalar_fractional()
        {
            var a = alb.scalar(4.0f);
            var b = alb.scalar(1.5f);
            var dy = alb.scalar(3.0f);

            var grads = alb.grads((Tensor[] x) => { return alb.pow(x[0], x[1]); });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];

            Assert.AreEqual(da.Get(0), 1.5 * Math.Pow(4, 0.5) * 3);
            Assert.AreEqual((float)Math.Round(db.Get(0), 4),
                (float)Math.Round(3.0f * (float)Math.Pow(4, 1.5f) * (float)Math.Log(4.0), 4));

        }



        [TestMethod]
        public void pow_gradient_tensor()
        {
            var a = alb.tensor1d(new float[] { -1, .5f, 2 });
            var b = alb.tensor1d(new float[] { 3, 2, -1 });
            var dy = alb.tensor1d(new float[] { 1, 5, 10 });

            var grads = alb.grads((Tensor[] x) => { return alb.pow(x[0], x[1]); });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];

            var daRes = new float[] {  3 * (float)Math.Pow(-1, 2) * 1, 2 *  (float)Math.Pow(.5, 1) * 5,
          -1 *  (float)Math.Pow(2, -2) * 10 };
            var dbRes = new float[] {  float.NaN, 5 *  (float)Math.Pow(.5, 2) * (float) Math.Log(.5),
      10 *  (float)Math.Pow(2, -1) *  (float)Math.Log(2) };

            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }
         

        [TestMethod]
        public void pow_gradient_scalar_tensor1d()
        {
            var a = alb.scalar(2);
            var b = alb.tensor1d(new float[] { 3, 4, 5 });
            var dy = alb.tensor1d(new float[] { 6, 7, 8 });

            var grads = alb.grads((Tensor[] x) => { return alb.pow(x[0], x[1]); });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 6 * 3 * (float)Math.Pow(2, 2) + 7 * 4 * (float)Math.Pow(2, 3) + 8 * 5 * (float)Math.Pow(2, 4) };
            var dbRes = new float[] {  6 * (float)Math.Pow(2, 3) * (float)Math.Log(2), 7 * (float)Math.Pow(2, 4) * (float)Math.Log(2),
      8 * (float)Math.Pow(2, 5) * (float)Math.Log(2)};


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }

        [TestMethod]
        public void pow_gradient_tensor2d_scalar()
        {
            var a = alb.tensor2d(new float[] { 2, 3, 4, 5 }, 2, 2);

            var b = alb.scalar(2);
            var dy = alb.tensor2d(new float[] { 6, 7, 8, 9 }, 2, 2);

            var grads = alb.grads((Tensor[] x) => { return alb.pow(x[0], x[1]); });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 6 * 2 * (float)Math.Pow(2, 1), 7 * 2 * (float)Math.Pow(3, 1), 8 * 2 * (float)Math.Pow(4, 1),
      9 * 2 * (float)Math.Pow(5, 1) };
            var dbRes = new float[] { 6 * (float)Math.Pow(2, 2) * (float)Math.Log(2) + 7 * (float)Math.Pow(3, 2) * (float)Math.Log(3) +
         8 * (float)Math.Pow(4, 2) * (float)Math.Log(4) + 9 * (float)Math.Pow(5, 2) * (float)Math.Log(5) };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(Math.Ceiling( da.Get(i)), Math.Ceiling( daRes[i]));
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(Math.Round( db.Get(i),2),Math.Round( dbRes[i],2));
            }
        }



        [TestMethod]
        public void pow_gradient_tensor2d_tensor2d_broadcast()
        {
            var a = alb.tensor2d(new float[] { 3, 4 }, 2, 1);

            var b = alb.tensor2d(new float[] { 2, 3, 4, 5 }, 2, 2);
            var dy = alb.tensor2d(new float[] { 6, 7, 8, 9 }, 2, 2);

            var grads = alb.grads((Tensor[] x) => { return alb.pow(x[0], x[1]); });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] {  6 * 2 * (float)Math.Pow(3, 1) + 7 * 3 * (float)Math.Pow(3, 2),
      8 * 4 * (float)Math.Pow(4, 3) + 9 * 5 *(float) Math.Pow(4, 4)};
            var dbRes = new float[] {6 * (float)Math.Pow(3, 2) *(float) Math.Log(3), 7 * (float)Math.Pow(3, 3) *(float) Math.Log(3),
      8 * (float)Math.Pow(4, 4) * (float)Math.Log(4), 9 * (float)Math.Pow(4, 5) * (float)Math.Log(4) };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual((float)Math.Round(da.Get(i), 5), (float)Math.Round(daRes[i], 5));
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual((float)Math.Round(db.Get(i), 4), (float)Math.Round(dbRes[i], 4));

            }
        }


        [TestMethod]
        public void add_test_1()
        {
            var c = alb.scalar(5);
            var a = alb.tensor1d(new float[] { 1, 2, 3 });

            var r = alb.add(a, c).dataSync();
            float[] res = new float[] { 6, 7, 8 };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }

        [TestMethod]
        public void add_propagates_nan()
        {
            var c = alb.scalar(float.NaN);
            var a = alb.tensor1d(new float[] { 1, 2, 3 });

            var r = alb.add(a, c).dataSync();
            float[] res = new float[] { float.NaN, float.NaN, float.NaN };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }


        [TestMethod]
        public void add_broadcasting_same_rank_Tensors_different_shape()
        {
            var a = alb.tensor2d(new float[] { 1, 2, -3, -4 }, 2, 2);
            var b = alb.tensor2d(new float[] { 2, 3 }, 2, 1);

            var r = alb.add(a, b);
            float[] res = new float[] { 3, 4, 0, -1 };
            int[] shape = new int[] { 2, 2 };

            Assert.AreEqual(r.Shape.Length, shape.Length);
            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(r.Shape[i], shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }


        [TestMethod]
        public void add_broadcasting_2D_1D()
        {
            var a = alb.tensor2d(new float[] { 1, 2, -3, -4 }, 2, 2);
            var c = alb.tensor1d(new float[] { 1, 2 });

            var r = alb.add(a, c);
            var expected = r.dataSync();

            float[] res = new float[] { 2, 4, -2, -2 };
            int[] shape = new int[] { 2, 2 };

            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(r.Shape[i], shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(expected[i], res[i]);
            }
        }


        [TestMethod]
        public void add_test_2()
        {
            var a = alb.tensor1d(new float[] { 2, 5, 1 });
            var c = alb.tensor1d(new float[] { 4, 2, -1 });

            var r = alb.add(a, c).dataSync();
            float[] res = new float[] { 6, 7, 0 };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }

        [TestMethod]
        public void add_test_2_propagates_nan()
        {
            var a = alb.tensor1d(new float[] { 2, 5, float.NaN });
            var c = alb.tensor1d(new float[] { 4, 2, -1 });

            var r = alb.add(a, c).dataSync();
            float[] res = new float[] { 6, 7, float.NaN };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }

        [TestMethod]
        public void add_broadcasts_2d_scalar()
        {
            var a = alb.tensor2d(new float[] { 1, 2, 3, 4, 5, 6 },  2, 3);
            var c = alb.scalar(2);

            var r = alb.add(a, c);
            float[] res = new float[] { 3, 4, 5, 6, 7, 8 };

            int[] shape = new int[] { 2, 3 };

            Assert.AreEqual(shape.Length, r.Shape.Length);

            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(shape[i], r.Shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }
        [TestMethod]
        public void add_broadcasts_scalar_1d()
        {
            var a = alb.scalar(2);
            var c = alb.tensor1d(new float[] { 1, 2, 3, 4, 5, 6 });

            var r = alb.add(a, c).dataSync();
            float[] res = new float[] { 3, 4, 5, 6, 7, 8 };
            Assert.AreEqual(r.Length, 6);
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }


        [TestMethod]
        public void add_broadcasts_2d_2d_one_dim()
        {
            var a = alb.tensor2d(new float[] { 1, 2, 5 }, 1, 3);
            var c = alb.tensor2d(new float[] { 7, 3 }, 2,1);

            var r = alb.add(a, c);
            float[] res = new float[] { 8, 9, 12, 4, 5, 8 };

            int[] shape = new int[] { 2, 3 };

            Assert.AreEqual(shape.Length, r.Shape.Length);

            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(shape[i], r.Shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }

        [TestMethod]
        public void add_broadcasts_2d_2d_inner_dim_of_b ()
        {
            var a = alb.tensor2d(new float[] { 1, 2, 5, 4, 5, 6 }, 2, 3);
            var c = alb.tensor2d(new float[] { 7, 3 }, 2, 1);

            var r = alb.add(a, c);
            float[] res = new float[] { 8, 9, 12, 7, 8, 9 };

            int[] shape = new int[] { 2, 3 };

            Assert.AreEqual(shape.Length, r.Shape.Length);

            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(shape[i], r.Shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }

        [TestMethod]
        public void add_broadcasts_3d_scalar()
        {
            var a = alb.tensor3d(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3, 1);
            var c = alb.scalar(-1);

            var r = alb.add(a, c);
            float[] res = new float[] { 0, 1, 2, 3, 4, 5 };

            int[] shape = new int[] { 2, 3, 1 };

            Assert.AreEqual(shape.Length, r.Shape.Length);

            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(shape[i], r.Shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }



        [TestMethod]
        public void add_gradient_scalar_tensor1d()
        {
            var a = alb.scalar(2);
            var b = alb.tensor1d(new float[] { 3, 4, 5 });
            var dy = alb.tensor1d(new float[] { 7, 8, 9 });

            var grads = alb.grads((Tensor[] x) => { return alb.add(x[0], x[1]); });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 7 + 8 + 9 };
            var dbRes = new float[] { 7, 8, 9 };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }



        [TestMethod]
        public void add_gradient_tensor2d_tensor2d_broadcast()
        {
            var a = alb.tensor2d(new float[] { 2, 3 }, 2, 1);

            var b = alb.tensor2d(new float[] { 4, 5, 6, 7 }, 2, 2);
            var dy = alb.tensor2d(new float[] { 5, 4, 3, 2 }, 2, 2);

            var grads = alb.grads((Tensor[] x) => { return alb.add(x[0], x[1]); });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 5 + 4, 3 + 2 };
            var dbRes = new float[] { 5, 4, 3, 2 };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual((float)Math.Round(da.Get(i), 5), (float)Math.Round(daRes[i], 5));
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual((float)Math.Round(db.Get(i), 4), (float)Math.Round(dbRes[i], 4));

            }
        }















        [TestMethod]
        public void sub_test_1()
        {
            var c = alb.scalar(5);
            var a = alb.tensor1d(new float[] { 7, 2, 3 });

            var r = alb.sub(c,a).dataSync();
            float[] res = new float[] { -2, 3, 2 };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }

        [TestMethod]
        public void sub_propagates_nan()
        {
            var c = alb.scalar(5);
            var a = alb.tensor1d(new float[] { 1, float.NaN, 3 });

            var r = alb.sub(a, c).dataSync();
            float[] res = new float[] { -4, float.NaN, -2 };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }


        [TestMethod]
        public void sub_broadcasting_same_rank_Tensors_different_shape()
        {
            var a = alb.tensor2d(new float[] { 1, 2, -3, -4 }, 2, 2);
            var b = alb.tensor2d(new float[] { 2, 3 }, 2, 1);

            var r = alb.sub(a, b);
            float[] res = new float[] { -1, 0, -6, -7 };
            int[] shape = new int[] { 2, 2 };

            Assert.AreEqual(r.Shape.Length, shape.Length);
            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(r.Shape[i], shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }


        [TestMethod]
        public void sub_broadcasting_2D_1D()
        {
            var a = alb.tensor2d(new float[] { 1, 2, -3, -4 }, 2, 2);
            var c = alb.tensor1d(new float[] { 1, 2 });

            var r = alb.sub(a, c);
            var expected = r.dataSync();

            float[] res = new float[] { 0, 0, -4, -6 };
            int[] shape = new int[] { 2, 2 };

            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(r.Shape[i], shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(expected[i], res[i]);
            }
        }


        [TestMethod]
        public void sub_test_2()
        {
            var a = alb.tensor1d(new float[] { 2, 5, 1 });
            var c = alb.tensor1d(new float[] { 4, 2, -1 });

            var r = alb.sub(a, c).dataSync();
            float[] res = new float[] { -2, 3, 2 };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }

        [TestMethod]
        public void sub_test_2_propagates_nan()
        {
            var a = alb.tensor1d(new float[] { 2, 5, 1 });
            var c = alb.tensor1d(new float[] { 4, float.NaN, -1 });

            var r = alb.sub(a, c).dataSync();
            float[] res = new float[] { -2, float.NaN, 2 };

            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }

        [TestMethod]
        public void sub_broadcasts_2d_scalar()
        {
            var a = alb.tensor2d(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
            var c = alb.scalar(2);

            var r = alb.sub(a, c);
            float[] res = new float[] { -1, 0, 1, 2, 3, 4 };

            int[] shape = new int[] { 2, 3 };

            Assert.AreEqual(shape.Length, r.Shape.Length);

            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(shape[i], r.Shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }
        [TestMethod]
        public void sub_broadcasts_scalar_1d()
        {
            var a = alb.scalar(2);
            var c = alb.tensor1d(new float[] { 1, 2, 3, 4, 5, 6 });

            var r = alb.sub(a, c).dataSync();
            float[] res = new float[] { 1, 0, -1, -2, -3, -4 };
            Assert.AreEqual(r.Length, 6);
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r[i], res[i]);
            }
        }


        [TestMethod]
        public void sub_broadcasts_2d_2d_one_dim()
        {
            var a = alb.tensor2d(new float[] { 1, 2, 5 }, 1, 3);
            var c = alb.tensor2d(new float[] { 7, 3 }, 2, 1);

            var r = alb.sub(a, c);
            float[] res = new float[] { -6, -5, -2, -2, -1, 2 };

            int[] shape = new int[] { 2, 3 };

            Assert.AreEqual(shape.Length, r.Shape.Length);

            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(shape[i], r.Shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }

        [TestMethod]
        public void sub_broadcasts_2d_2d_inner_dim_of_b()
        {
            var a = alb.tensor2d(new float[] { 1, 2, 5, 4, 5, 6 }, 2, 3);
            var c = alb.tensor2d(new float[] { 7, 3 }, 2, 1);

            var r = alb.sub(a, c);
            float[] res = new float[] { -6, -5, -2, 1, 2, 3 };

            int[] shape = new int[] { 2, 3 };

            Assert.AreEqual(shape.Length, r.Shape.Length);

            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(shape[i], r.Shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }

        [TestMethod]
        public void sub_broadcasts_3d_scalar()
        {
            var a = alb.tensor3d(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3, 1);
            var c = alb.scalar(-1);

            var r = alb.sub(a, c);
            float[] res = new float[] { 2, 3, 4, 5, 6, 7 };

            int[] shape = new int[] { 2, 3, 1 };

            Assert.AreEqual(shape.Length, r.Shape.Length);

            for (int i = 0; i < shape.Length; i++)
            {
                Assert.AreEqual(shape[i], r.Shape[i]);
            }
            for (int i = 0; i < res.Length; i++)
            {
                Assert.AreEqual(r.dataSync()[i], res[i]);
            }
        }


        [TestMethod]
        public void sub_gradient_basic_tensor2d()
        {
            var a = alb.tensor2d(new float[] { 0, 1, 2, 3 }, 2, 2);
            var b = alb.tensor2d(new float[] { 3, 2, 1, 0 }, 2, 2);
            var dy = alb.tensor2d(new float[] { 1, 10, 15, 20 }, 2, 2);

            var grads = alb.grads((Tensor[] x) => { return alb.sub(x[0], x[1]); });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 1, 10, 15, 20 };
            var dbRes = new float[] { -1, -10, -15, -20 };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }
        [TestMethod]
        public void sub_gradient_tensor1d_scalar()
        {
            var b = alb.scalar(2);
            var a = alb.tensor1d(new float[] { 3, 4, 5 });
            var dy = alb.tensor1d(new float[] { 7, 8, 9 });

            var grads = alb.grads((Tensor[] x) => { return alb.sub(x[0], x[1]); });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 7, 8, 9 };
            var dbRes = new float[] { -7 - 8 - 9 };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }

        [TestMethod]
        public void sub_gradient_scalar_tensor1d()
        {
            var a = alb.scalar(2);
            var b = alb.tensor1d(new float[] { 3, 4, 5 });
            var dy = alb.tensor1d(new float[] { 7, 8, 9 });

            var grads = alb.grads((Tensor[] x) => { return alb.sub(x[0], x[1]); });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 7 + 8 + 9 };
            var dbRes = new float[] { -7, -8, -9 };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual(da.Get(i), daRes[i]);
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual(db.Get(i), dbRes[i]);
            }
        }



        [TestMethod]
        public void sub_gradient_tensor2d_tensor2d_broadcast()
        {
            var a = alb.tensor2d(new float[] { 4, 5, 6, 7 }, 2, 2);
            var b = alb.tensor2d(new float[] { 2, 3 }, 2, 1);

            var dy = alb.tensor2d(new float[] { 5, 4, 3, 2 }, 2, 2);

            var grads = alb.grads((Tensor[] x) => { return alb.sub(x[0], x[1]); });
            var d = grads(new Tensor[] { a, b }, dy);
            var da = d[0];
            var db = d[1];


            var daRes = new float[] { 5, 4, 3, 2 };
            var dbRes = new float[] { -5 - 4, -3 - 2 };


            Assert.AreEqual(a.Shape.Length, da.Shape.Length);
            for (int i = 0; i < a.Shape.Length; i++)
            {
                Assert.AreEqual(a.Shape[i], da.Shape[i]);
            }


            for (int i = 0; i < daRes.Length; i++)
            {

                Assert.AreEqual((float)Math.Round(da.Get(i), 5), (float)Math.Round(daRes[i], 5));
            }

            Assert.AreEqual(b.Shape.Length, db.Shape.Length);
            for (int i = 0; i < b.Shape.Length; i++)
            {
                Assert.AreEqual(b.Shape[i], db.Shape[i]);
            }

            for (int i = 0; i < dbRes.Length; i++)
            {

                Assert.AreEqual((float)Math.Round(db.Get(i), 4), (float)Math.Round(dbRes[i], 4));

            }
        }


    }
}
