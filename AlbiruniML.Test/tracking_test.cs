using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using alb = AlbiruniML.Ops; 
using AlbiruniML;
namespace AlbiruniML.Test
{
    [TestClass]
    public class tracking_test
    { 
        public void tidyreturntensors()
        {
            ENV.engine = new Engine();
            int current = ENV.engine.memory().numTensors;
            alb.tidy(() =>
            {
                current = ENV.engine.memory().numTensors;
                var a = alb.tensor1d(alb.data(1, 2, 3));
                var b = alb.tensor1d(alb.data(0, 0, 0));
                Assert.AreEqual(current + 2, ENV.engine.memory().numTensors);
               
                alb.tidy(() =>
                {
                    var result = alb.tidy(() =>
                    {
                        b = alb.addStrict(a, b);
                        b = alb.addStrict(a, b);
                        b = alb.addStrict(a, b);
                        return alb.add(a, b);
                    });

                    // result is new. All intermediates should be disposed. 
                    Assert.AreEqual(current + 2 + 1, ENV.engine.memory().numTensors);
                    AssertTools.ArrayIsEqual(result.ToArray(), alb.data(4, 8, 12));
                   
                });

                // a, b are still here, result should be disposed.
                 Assert.AreEqual(ENV.engine.memory().numTensors, current+2);
              
            });

            Assert.AreEqual(current + 0, ENV.engine.memory().numTensors); 
        }

         
        public void multiple_disposes_does_not_affect_num_arrays()
        {
            ENV.engine = new Engine();
             
            Assert.AreEqual(0, ENV.engine.memory().numTensors);
            var a = alb.tensor1d(alb.data(1, 2, 3));
            var b = alb.tensor1d(alb.data(1, 2, 3)); 
            Assert.AreEqual(2, ENV.engine.memory().numTensors);
            a.dispose();
            a.dispose(); 
            Assert.AreEqual(1, ENV.engine.memory().numTensors);
            b.dispose(); 
            Assert.AreEqual(0, ENV.engine.memory().numTensors);
        }
    }
}
