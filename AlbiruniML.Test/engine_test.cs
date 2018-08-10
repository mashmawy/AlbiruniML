using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using alb = AlbiruniML.Ops;
using AlbiruniML;
using System.Collections.Generic;
namespace AlbiruniML.Test
{
    [TestClass]
    public class engine_test
    {
        [TestMethod]
        public void gradients()
        {
            ENV.engine = new Engine();
            var a = alb.tensor2d(new float[] { -1, 2, -3, 10, -20, 30 }, 2, 3);
            var b = alb.tensor2d(new float[] { 2, -3, 4, -1, 2, -3 }, 3, 2);

            var grads = alb.grads((Tensor[] x) =>
            {
                var aI = x[0];
                var bI = x[1];
                // m = dot(a, b)
                // y = relu(m)
                // e = sum(y)
                var m = alb.matMul(aI, bI);
                var y = alb.relu(m);
                return alb.sum(y);

            })(new Tensor[] { a, b }, null);

            var da = grads[0];
            var db = grads[1];

            // de/dy = 1
            // dy/dm = step(m)
            // de/dm = de/dy * dy/dm = step(m)
            var dedm = alb.step(alb.matMul(a, b));

            // de/da = dot(de/dy, bT)

            AssertTools.ArrayIsEqual(da.Shape, a.Shape);

            var transposeA = false;
            var transposeB = true;

            AssertTools.TensorIsEqual(da, alb.matMul(dedm, b, transposeA, transposeB));




            AssertTools.ArrayIsEqual(db.Shape, b.Shape);
            transposeA = true;
            transposeB = false;

            AssertTools.TensorIsEqual(db, alb.matMul(a, dedm, transposeA, transposeB));

        }


        [TestMethod]
        public void gradf()
        {
            ENV.engine = new Engine();
            var grad = alb.grad(x => x.square());
            var result = grad(alb.tensor1d(new float[] { .1f, .2f }), null);
            AssertTools.ArrayIsEqual(result.dataSync(), new float[] { .2f, .4f });
        }
        [TestMethod]
        public void calling_gradf_twice_works()
        {
            ENV.engine = new Engine();
            var grad = alb.grad(x => x.square());
            var result = grad(alb.tensor1d(new float[] { .1f, .2f }), null);
            var result2 = grad(alb.tensor1d(new float[] { .1f, .4f }), null);
            AssertTools.ArrayIsEqual(result.dataSync(), new float[] { .2f, .4f });
            AssertTools.ArrayIsEqual(result2.dataSync(), new float[] { .2f, .8f });
        }

        [TestMethod]
        public void gradsf()
        {
            ENV.engine = new Engine();
            var grad = alb.grads(x => x[0].square());
            var result = grad(new Tensor[]{alb.tensor1d(new float[] { .1f, .2f })}, null);
            AssertTools.ArrayIsEqual(result[0].dataSync(), new float[] { .2f, .4f });
        }
        [TestMethod]
        public void calling_gradsf_twice_works()
        {
            ENV.engine = new Engine();
            var grad = alb.grads(x => x[0].square());
            var result = grad(new Tensor[] { alb.tensor1d(new float[] { .1f, .2f }) }, null);
            var result2 = grad(new Tensor[] { alb.tensor1d(new float[] { .1f, .4f }) }, null);
            AssertTools.ArrayIsEqual(result[0].dataSync(), new float[] { .2f, .4f });
            AssertTools.ArrayIsEqual(result2[0].dataSync(), new float[] { .2f, .8f });
        }



        [TestMethod]
        public void works_with_reshape()
        {

            ENV.engine = new Engine();
            var a = alb.data(1, 2, 3, 4).ToTensor(2, 2);
            var exponent =alb.data(2, 2, 2, 2).ToTensor();

            var da = alb.grad(aI =>
            {
                var b = aI.flatten();
                var m = alb.pow(b, exponent);
                return alb.sum(m);
            })(a, null);

            AssertTools.ArrayIsEqual(da.Shape, new int[] { 2, 2 });
            AssertTools.ArrayIsEqual(da.dataSync(), new float[] { 2, 4, 6, 8 }); 
        }




        [TestMethod]
        public void does_not_error_if_irrelevant_pruned_ops_are_missing_grads()
        {
            ENV.engine = new Engine();
            var a = alb.tensor1d(new float[] { 1, 1 });
            var b = alb.tensor1d(new float[] { 0, 1 });


            var da = alb.grad(aI =>
            {
                // Logical has no gradients, but it is irrelevant.
                aI.logicalAnd(b);
                return aI.sum();
            })(a, null);
            AssertTools.ArrayIsEqual(da.dataSync(), new float[] { 1, 1 });
        }
        [TestMethod]
        public void errors_if_relevant_ops_are_missing_grads()
        {
            ENV.engine = new Engine();
            var a = alb.tensor1d(new float[] { 1, 1 });
            var b = alb.tensor1d(new float[] { 0, 1 });

            
            var da = alb.grad(aI =>
            {
                // Logical has no gradients, but it is irrelevant.
                return aI.logicalAnd(b);
                   
            }) ;

            try
            {
                da(a, null);
                Assert.AreEqual(false, true); 
            }
            catch (Exception)
            {
                Assert.AreEqual(true, true);
               
            }

        }


        //higher-order gradients
        [TestMethod]
        public void grad_grad_f()
        {
            ENV.engine = new Engine();

            var gradgrad = alb.grad((Tensor x) =>
            {

                return alb.grad(x2 => x2.mul(x2).mul(x2))(x, null);

            }
            );


            var result = gradgrad(alb.tensor1d(new float[] { .1f, .2f }), null);
            AssertTools.ArrayIsEqual(result.dataSync(), new float[] { .6f, 1.2f });
        }
        [TestMethod]
        public void grads_grads_f()
        {
            ENV.engine = new Engine();

            var gradgrad = alb.grads((Tensor[] x) =>
            {

                return alb.grads(x2 => x2[0].mul(x2[0]).mul(x2[0]))(x, null)[0];

            }
            );


            var result = gradgrad(new Tensor[] { alb.tensor1d(new float[] { .1f, .2f }) }, null);
            AssertTools.ArrayIsEqual(result[0].dataSync(), new float[] { .6f, 1.2f });
        }


        ///second order derivative through customGradient
        [TestMethod]
        public void second_order_derivative_through_customGradient()
        {
            ENV.engine = new Engine();
            var a = alb.scalar(3);
            var b = alb.scalar(2);

            var dy = alb.scalar(5);

            var customPow = alb.customGrad(aI =>
            {
                var value = alb.pow(a, b);
                Func<Tensor, List<Tensor>> gradFunc =
                    (Tensor dyI) =>
                    {
                        return new List<Tensor>() { 
                            dyI.mul(aI[0]) };

                    };
                return new CustomGradientResults()
                {
                    gradFunc = gradFunc,
                    value = value
                };
            });

            var dda = alb.grad((Tensor xI) =>
            {
                return alb.grad(xI2 => customPow(new Tensor[] { xI2 }))(xI, null);
            }  )(a, dy); 

            AssertTools.ArrayIsEqual(dda.Shape, a.Shape);

            // First order: dy * a. Second order: dy.
            AssertTools.TensorIsEqual(dda, dy);

        }


        [TestMethod]
        public void calling_gradient_of_custom_op_twice_works()
        {

            ENV.engine = new Engine();

            var customOp = alb.customGrad(x =>
            {
                // Override gradient of our custom x ^ 2 op to be dy * abs(x);
                return new CustomGradientResults()
                {
                    value = x[0].square(),
                    gradFunc = dy => new List<Tensor>() { dy.mul(x[0].abs()) }
                };
            });
            var x2 = alb.tensor1d(new float[] { -1, -2, 3 });
            var grad = alb.grad(x3 => customOp(new Tensor[] { x3 }));



            AssertTools.ArrayIsEqual(grad(x2, null).dataSync(), new float[] { 1, 2, 3 });
            AssertTools.ArrayIsEqual(grad(x2, null).dataSync(), new float[] { 1, 2, 3 });
        }

        
    }
}
