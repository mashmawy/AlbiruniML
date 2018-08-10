using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using alb = AlbiruniML.Ops;
using AlbiruniML;
namespace SharpDL.Test
{
    [TestClass]
    public class matmul_test
    {
        [TestMethod]
        public void TestMethod1()
        {
            ENV.engine = new Engine();
            var a = alb.tensor2d(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
            var b = alb.tensor2d(new float[] { 1, 0, 2, 4, 3, 0 }, 2, 3);
           
            var transposeA = false;
            var transposeB = true;
            var c = alb.matMul(a, b, transposeA, transposeB);

            var expected = new float[] { 7, 10, 16, 31 };

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(c.dataSync()[i], expected[i]);
            }
        }
    }
}
