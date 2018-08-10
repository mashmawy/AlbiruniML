using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using alb = AlbiruniML.Ops;
using AlbiruniML;
using System.Collections.Generic;
namespace AlbiruniML.Test
{
    [TestClass]
    public class array_ops_test
    {
        [TestMethod]
        public void zeros1D()
        {

            ENV.engine = new Engine();
            var a = alb.zeros(new int[] { 3 });
            AssertTools.ArrayIsEqual(a.Shape, new int[] { 3 });
            AssertTools.ArrayIsEqual(a.dataSync(), new float[] { 0, 0, 0 });
        }
        [TestMethod]
        public void zeros2D()
        {

            ENV.engine = new Engine();
            var a = alb.zeros(new int[] { 3, 2 });
            AssertTools.ArrayIsEqual(a.Shape, new int[] { 3, 2 });
            AssertTools.ArrayIsEqual(a.dataSync(), new float[] { 0, 0, 0, 0, 0, 0 });
        }
        [TestMethod]
        public void zeros3D()
        {

            var a = alb.zeros(new int[] { 2, 2, 2 });
            AssertTools.ArrayIsEqual(a.Shape, new int[] { 2, 2, 2 });
            AssertTools.ArrayIsEqual(a.dataSync(), new float[] { 0, 0, 0, 0, 0, 0, 0, 0 });
        }

        [TestMethod]
        public void zeros4D()
        {

            var a = alb.zeros(new int[] { 3, 2, 1, 1 });
            AssertTools.ArrayIsEqual(a.Shape, new int[] { 3, 2, 1, 1 });
            AssertTools.ArrayIsEqual(a.dataSync(), new float[] { 0, 0, 0, 0, 0, 0 });
        }




        [TestMethod]
        public void ones1D()
        {

            var a = alb.ones(new int[] { 3 });
            AssertTools.ArrayIsEqual(a.Shape, new int[] { 3 });
            AssertTools.ArrayIsEqual(a.dataSync(), new float[] { 1, 1, 1 });
        }
        [TestMethod]
        public void ones2D()
        {

            var a = alb.ones(new int[] { 3, 2 });
            AssertTools.ArrayIsEqual(a.Shape, new int[] { 3, 2 });
            AssertTools.ArrayIsEqual(a.dataSync(), new float[] { 1, 1, 1, 1, 1, 1 });
        }
        [TestMethod]
        public void ones3D()
        {

            var a = alb.ones(new int[] { 2, 2, 2 });
            AssertTools.ArrayIsEqual(a.Shape, new int[] { 2, 2, 2 });
            AssertTools.ArrayIsEqual(a.dataSync(), new float[] { 1, 1, 1, 1, 1, 1, 1, 1 });
        }

        [TestMethod]
        public void ones4D()
        {

            var a = alb.ones(new int[] { 3, 2, 1, 1 });
            AssertTools.ArrayIsEqual(a.Shape, new int[] { 3, 2, 1, 1 });
            AssertTools.ArrayIsEqual(a.dataSync(), new float[] { 1, 1, 1, 1, 1, 1 });
        }


        [TestMethod]
        public void zerosLike1d()
        {

            var a = alb.tensor1d(new float[] { 1, 2, 3 });
            var b = alb.zerosLike(a);
            AssertTools.ArrayIsEqual(b.Shape, new int[] { 3 });
            AssertTools.ArrayIsEqual(b.dataSync(), new float[] { 0, 0, 0 });
        }


        [TestMethod]
        public void zerosLike2d()
        {

            var a = alb.tensor2d(new float[] { 1, 2, 3, 4 }, 2, 2);
            var b = alb.zerosLike(a);
            AssertTools.ArrayIsEqual(b.Shape, new int[] { 2, 2 });
            AssertTools.ArrayIsEqual(b.dataSync(), new float[] { 0, 0, 0, 0 });
        }


        [TestMethod]
        public void zerosLike3d()
        {

            var a = alb.tensor3d(new float[] { 1, 2, 3, 4 }, 2, 2, 1);
            var b = alb.zerosLike(a);
            AssertTools.ArrayIsEqual(b.Shape, new int[] { 2, 2, 1 });
            AssertTools.ArrayIsEqual(b.dataSync(), new float[] { 0, 0, 0, 0 });
        }

        [TestMethod]
        public void zerosLike4d()
        {

            var a = alb.tensor4d(new float[] { 1, 2, 3, 4 }, 2, 2, 1, 1);
            var b = alb.zerosLike(a);
            AssertTools.ArrayIsEqual(b.Shape, new int[] { 2, 2, 1, 1 });
            AssertTools.ArrayIsEqual(b.dataSync(), new float[] { 0, 0, 0, 0 });
        }









        [TestMethod]
        public void onesLike1d()
        {

            var a = alb.tensor1d(new float[] { 1, 2, 3 });
            var b = alb.onesLike(a);
            AssertTools.ArrayIsEqual(b.Shape, new int[] { 3 });
            AssertTools.ArrayIsEqual(b.dataSync(), new float[] { 1, 1, 1 });
        }


        [TestMethod]
        public void onesLike2d()
        {

            var a = alb.tensor2d(new float[] { 1, 2, 3, 4 }, 2, 2);
            var b = alb.onesLike(a);
            AssertTools.ArrayIsEqual(b.Shape, new int[] { 2, 2 });
            AssertTools.ArrayIsEqual(b.dataSync(), new float[] { 1, 1, 1, 1 });
        }


        [TestMethod]
        public void onesLike3d()
        {

            var a = alb.tensor3d(new float[] { 1, 2, 3, 4 }, 2, 2, 1);
            var b = alb.onesLike(a);
            AssertTools.ArrayIsEqual(b.Shape, new int[] { 2, 2, 1 });
            AssertTools.ArrayIsEqual(b.dataSync(), new float[] { 1, 1, 1, 1 });
        }

        [TestMethod]
        public void onesLike4d()
        {

            var a = alb.tensor4d(new float[] { 1, 2, 3, 4 }, 2, 2, 1, 1);
            var b = alb.onesLike(a);
            AssertTools.ArrayIsEqual(b.Shape, new int[] { 2, 2, 1, 1 });
            AssertTools.ArrayIsEqual(b.dataSync(), new float[] { 1, 1, 1, 1 });
        }



        [TestMethod]
        public void should_return_a_random_1D()
        {
            int[] shape = new int[] { 10 };
            var result = alb.rand(shape, () => Util.randUniform(0, 2));
            AssertTools.ValuesInRange(result.dataSync(), 0, 2);

            result = alb.rand(shape, () => Util.randUniform(0, 1.5));
            AssertTools.ValuesInRange(result.dataSync(), 0, 1.5f);


        }


        [TestMethod]
        public void should_return_a_random_2D()
        {
            int[] shape = new int[] { 3, 4 };
            var result = alb.rand(shape, () => Util.randUniform(0, 2));
            AssertTools.ValuesInRange(result.dataSync(), 0, 2);
        }



        [TestMethod]
        public void should_return_a_random_3D()
        {
            int[] shape = new int[] { 3, 4, 5 };
            var result = alb.rand(shape, () => Util.randUniform(0, 2));
            AssertTools.ValuesInRange(result.dataSync(), 0, 2);
        }


        [TestMethod]
        public void should_return_a_random_4D()
        {
            int[] shape = new int[] { 3, 4, 5, 6 };
            var result = alb.rand(shape, () => Util.randUniform(0, 2));
            AssertTools.ValuesInRange(result.dataSync(), 0, 2);

        }


        [TestMethod]
        public void eye1by1()
        {
            AssertTools.ArrayIsEqual(alb.eye(1).dataSync(), alb.tensor2d(new float[] { 1 }, 1, 1).dataSync());
        }


        [TestMethod]
        public void eye2by2()
        {
            AssertTools.ArrayIsEqual(alb.eye(2).dataSync(), alb.tensor2d(new float[] { 1, 0, 0, 1 }, 2, 2).dataSync());
        }


        [TestMethod]
        public void eye3by3()
        {
            AssertTools.ArrayIsEqual(alb.eye(3).dataSync(), alb.tensor2d(new float[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 }, 3, 3).dataSync());
        }



        [TestMethod]
        public void eye3by4()
        {
            AssertTools.ArrayIsEqual(alb.eye(3, 4).dataSync(), alb.tensor2d(new float[] { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 }, 3, 4).dataSync());
        }

        [TestMethod]
        public void eye4by3()
        {
            AssertTools.ArrayIsEqual(alb.eye(4, 3).dataSync(),
                alb.tensor2d(new float[] { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 }, 4, 3).dataSync());
        }


        [TestMethod]
        public void eye1dbatchShape()
        {
            AssertTools.ArrayIsEqual(alb.eye(2, 2, new int[] { 3 }).dataSync(),
                alb.tensor3d(new float[] { 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1 }, 2, 2, 3).dataSync());
        }



        [TestMethod]
        public void eye2dbatchShape()
        {
            AssertTools.ArrayIsEqual(alb.eye(2, 2, new int[] { 2, 3 }).dataSync(),
                alb.tensor4d(new float[] { 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1 }, 2, 2, 2, 3).dataSync());
        }


        [TestMethod]
        public void tile1d()
        {
            var t = alb.tensor1d(alb.data(1, 2, 3));
            var t2 = alb.tile(t, alb.shape(2));
            AssertTools.ArrayIsEqual(t2.Shape, alb.shape(6));
            AssertTools.ArrayIsEqual(t2, alb.data(1, 2, 3, 1, 2, 3));
        }

        [TestMethod]
        public void tile2d()
        {
            var t = alb.tensor2d(alb.data(1, 11, 2, 22), 2, 2);
            var t2 = alb.tile(t, alb.shape(1, 2));
            AssertTools.ArrayIsEqual(t2.Shape, alb.shape(2, 4));
            AssertTools.ArrayIsEqual(t2, alb.data(1, 11, 1, 11, 2, 22, 2, 22));

            t2 = alb.tile(t, alb.shape(2, 1));
            AssertTools.ArrayIsEqual(t2.Shape, alb.shape(4, 2));
            AssertTools.ArrayIsEqual(t2, alb.data(1, 11, 2, 22, 1, 11, 2, 22));


            t2 = alb.tile(t, alb.shape(2, 2));
            AssertTools.ArrayIsEqual(t2.Shape, alb.shape(4, 4));
            AssertTools.ArrayIsEqual(t2, alb.data(1, 11, 1, 11, 2, 22, 2, 22, 1, 11, 1, 11, 2, 22, 2, 22));
        }


        [TestMethod]
        public void tile3D()
        {
            var t = alb.tensor3d(alb.data(1, 2, 3, 4, 5, 6, 7, 8), 2, 2, 2);
            var t2 = alb.tile(t, alb.shape(1, 2, 1));
            AssertTools.ArrayIsEqual(t2.Shape, alb.shape(2, 4, 2));
            AssertTools.ArrayIsEqual(t2, alb.data(1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8));

        }



        [TestMethod]
        public void tilepropagatesNaNs()
        {
            var t = alb.tensor1d(alb.data(1, 2, float.NaN));
            var t2 = alb.tile(t, alb.shape(2));
            AssertTools.ArrayIsEqual(t2.Shape, alb.shape(6));
            AssertTools.ArrayIsEqual(t2, alb.data(1, 2, float.NaN, 1, 2, float.NaN));


        }

        [TestMethod]
        public void tile1dgradient()
        {
            var t = alb.tensor1d(alb.data(1, 2, 3));
            var dy = alb.tensor1d(alb.data(0.1f, 0.2f, 0.3f, 1, 2, 3, 10, 20, 30));

            var grad = alb.grad((Tensor x) => { return alb.tile(x, alb.shape(3)); });
            var d = grad(t, dy);
            AssertTools.ArrayIsEqual(d, alb.data(11.1f, 22.2f, 33.3f));
        }



        [TestMethod]
        public void tile2dgradient()
        {
            var t = alb.tensor2d(alb.data(1, 2, 3, 4), 2, 2);
            var dy = alb.tensor2d(alb.data(1, 2, 10, 20, 3, 4, 30, 40), 2, 4);

            var grad = alb.grad((Tensor x) => { return alb.tile(x, alb.shape(1, 2)); });
            var d = grad(t, dy);
            AssertTools.TensorIsEqual(d, alb.data(11, 22, 33, 44).ToTensor(2, 2));
        }






        [TestMethod]
        public void gather1d()
        {
            var t = alb.tensor1d(alb.data(1, 2, 3));
            var t2 = alb.gather(t, alb.tensor1d(alb.data(0, 2, 0, 1)), 0);
            AssertTools.ArrayIsEqual(t2.Shape, alb.shape(4));
            AssertTools.ArrayIsEqual(t2, alb.data(1, 3, 1, 2));
        }

        [TestMethod]
        public void gather2d()
        {
            var t = alb.tensor2d(alb.data(1, 11, 2, 22), 2, 2);
            var t2 = alb.gather(t, alb.tensor1d(alb.data(1, 0, 0, 1)), 0);
            AssertTools.ArrayIsEqual(t2.Shape, alb.shape(4, 2));
            AssertTools.ArrayIsEqual(t2, alb.data(2, 22, 1, 11, 1, 11, 2, 22));

            t2 = alb.gather(t, alb.tensor1d(alb.data(1, 0, 0, 1)), 1);
            AssertTools.ArrayIsEqual(t2.Shape, alb.shape(2, 4));
            AssertTools.ArrayIsEqual(t2, alb.data(11, 1, 1, 11, 22, 2, 2, 22));

        }


        [TestMethod]
        public void gather3D()
        {
            var t = alb.tensor3d(alb.data(1, 2, 3, 4, 5, 6, 7, 8), 2, 2, 2);
            var t2 = alb.gather(t, alb.tensor1d(alb.data(1, 0, 0, 1)), 2);
            AssertTools.ArrayIsEqual(t2.Shape, alb.shape(2, 2, 4));
            AssertTools.ArrayIsEqual(t2, alb.data(2, 1, 1, 2, 4, 3, 3, 4, 6, 5, 5, 6, 8, 7, 7, 8));

        }


        [TestMethod]
        public void gatherpropagatesNaNs()
        {
            var t = alb.tensor1d(alb.data(1, 2, float.NaN));
            var t2 = alb.gather(t, alb.tensor1d(alb.data(0, 2, 0, 1)), 0);
            AssertTools.ArrayIsEqual(t2.Shape, alb.shape(4));
            AssertTools.ArrayIsEqual(t2, alb.data(1, float.NaN, 1, 2));

        }




        [TestMethod]
        public void gatherchaining()
        {
            var x = alb.zeros(alb.shape(2, 4, 6));
            // [0, 2, 4]
            var indices = alb.range(0, 6, 2);
            var axis = 2;
            AssertTools.ArrayIsEqual(x.gather(indices, axis).Shape, alb.shape(2, 4, 3));

        }
        [TestMethod]
        public void gather1dgradient()
        {
            var t = alb.tensor1d(alb.data(1, 2, 3));

            var indices = alb.tensor1d(alb.data(0, 2, 0, 1));
            var dy = alb.tensor1d(alb.data(3, 4, 5, 6));

            var grad = alb.grad((Tensor x) => { return alb.gather(x, indices); });
            var d = grad(t, dy);
            AssertTools.ArrayIsEqual(d, alb.data(8, 6, 4));
        }

        [TestMethod]
        public void gather2dgradient4_1()
        {
            var t = alb.tensor2d(alb.data(1, 11, 2, 22), 4, 1);
            var indices = alb.tensor1d(alb.data(1, 0, 0, 1));
            var dy = alb.tensor2d(alb.data(23, 7, 19, 13), 4, 1);
            var axis = 0;

            var grad = alb.grad((Tensor x) => { return alb.gather(x, indices, axis); });
            var d = grad(t, dy);


            AssertTools.ArrayIsEqual(t.Shape, d.Shape);

            AssertTools.TensorIsEqual(d, alb.data(26, 36, 0, 0).ToTensor(4, 1));
        }

        [TestMethod]
        public void gather2dgradient2_2()
        {
            var t = alb.tensor2d(alb.data(1, 11, 2, 22), 2, 2);
            var indices = alb.tensor1d(alb.data(1, 0, 0, 1));
            var dy = alb.tensor2d(alb.data(3, 4, 5, 6, 7, 8, 9, 10), 2, 4);
            var axis = 1;

            var grad = alb.grad((Tensor x) => { return alb.gather(x, indices, axis); });
            var d = grad(t, dy);


            AssertTools.ArrayIsEqual(t.Shape, d.Shape);

            AssertTools.TensorIsEqual(d, alb.data(9, 9, 17, 17).ToTensor(4, 1));
        }
        [TestMethod]
        public void oneHotDepth2diagonal()
        {
            var indices = alb.tensor1d(alb.data(0, 1));
            var res = alb.oneHot(indices, 2);
            AssertTools.ArrayIsEqual(res.Shape, alb.shape(2, 2));

            AssertTools.TensorIsEqual(res, alb.data(1, 0, 0, 1).ToTensor(2, 2));
        }
        [TestMethod]
        public void oneHotDepth2transposeddiagonal()
        {
            var indices = alb.tensor1d(alb.data(1, 0));
            var res = alb.oneHot(indices, 2);
            AssertTools.ArrayIsEqual(res.Shape, alb.shape(2, 2));

            AssertTools.TensorIsEqual(res, alb.data(0, 1, 1, 0).ToTensor(2, 2));
        }
        [TestMethod]
        public void oneHotDepth3events4()
        {
            var indices = alb.tensor1d(alb.data(2, 1, 2, 0));
            var res = alb.oneHot(indices, 3);
            AssertTools.ArrayIsEqual(res.Shape, alb.shape(4, 3));

            AssertTools.TensorIsEqual(res, alb.data(0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0).ToTensor(4, 3));
        }
        [TestMethod]
        public void oneHotDepth2onValue3offValue_minus_2()
        {
            var indices = alb.tensor1d(alb.data(0, 1));
            var res = alb.oneHot(indices, 2, 3, -2);
            AssertTools.ArrayIsEqual(res.Shape, alb.shape(2, 2));

            AssertTools.TensorIsEqual(res, alb.data(3, -2, -2, 3).ToTensor(2, 2));
        }
        [TestMethod]
        public void oneHotOut_of_range_events_do_not_trigger_onValue()
        {
            ENV.engine = new Engine();
            var indices = alb.tensor1d(alb.data(-1, 5, 12345));
            var res = alb.oneHot(indices, 5);
            AssertTools.ArrayIsEqual(res.Shape, alb.shape(3, 5));

            AssertTools.TensorIsEqual(res, alb.data(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).ToTensor(3, 5));
        }
    }
}
