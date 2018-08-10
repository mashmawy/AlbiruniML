using AlbiruniML;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML.Test
{
    public static class AssertTools
    {

        public static void ValuesInRange(float[] t1, float min, float max)
        {
            for (int i = 0; i < t1.Length; i++)
            {
                Assert.AreEqual(true, t1[i] >= min);
                Assert.AreEqual(true, t1[i] <= max);
            }
        }

        public static void NamedTensorIsEqual(Dictionary<string,Tensor> d1,Dictionary<string,Tensor> d2)
        {
            foreach (var item in d1)
            {
                Assert.AreEqual(true, d2.ContainsKey(item.Key));
                Assert.AreEqual(true, d2.ContainsValue(item.Value));
            }

        }
        public static void TapeIsEqual(TapeNode[] t1, TapeNode[] t2)
        {
            for (int i = 0; i < t1.Length; i++)
            {
                TapeNodeIsEqual(t1[i], t2[i]);
            }
        }
        public static void TapeNodeIsEqual(TapeNode t1, TapeNode t2)
        {
            Assert.AreEqual(t1.id, t2.id);
            Assert.AreEqual(t1.name, t2.name);
            NamedTensorIsEqual(t1.inputs, t2.inputs);
            TensorIsEqual(t1.output, t2.output);
        }
        public static void TensorIsEqual(Tensor  t1, Tensor  t2)
        {
            var arr1 = t1.dataSync();
            var arr2 = t2.dataSync();

            ArrayIsEqual(arr1, arr2);
        }
        public static void ArrayIsEqual<T>( T[] arr1,T[] arr2)
        {
            Assert.AreEqual(arr1.Length, arr2.Length);
            for (int i = 0; i < arr1.Length; i++)
            {
                Assert.AreEqual(arr1[i], arr2[i]);
            }
        }
    }
}
