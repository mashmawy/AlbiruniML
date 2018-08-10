using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using alb = AlbiruniML.Ops;
using AlbiruniML;
namespace SharpDL.Test
{
    [TestClass]
    public class lstm_test
    {
        [TestMethod]
        public void MultiRNNCell_with_2_BasicLSTMCells()
        {
            ENV.engine = new Engine();
            var lstmKernel1 = alb.tensor2d(
      new float[]{
          0.26242125034332275f, -0.8787832260131836f, 0.781475305557251f,
          1.337337851524353f, 0.6180247068405151f, -0.2760246992111206f,
          -0.11299663782119751f, -0.46332040429115295f, -0.1765323281288147f,
          0.6807947158813477f, -0.8326982855796814f, 0.6732975244522095f
        },
       3, 4);


            var lstmBias1 = alb.tensor1d(
          new float[] { 1.090713620185852f, -0.8282332420349121f, 0f, 1.0889357328414917f });

            var lstmKernel2 = alb.tensor2d(
      new float[]{
          -1.893059492111206f, -1.0185645818710327f, -0.6270437240600586f,
          -2.1829540729522705f, -0.4583775997161865f, -0.5454602241516113f,
          -0.3114445209503174f, 0.8450229167938232f
        },
       2, 4);
            var lstmBias2 = alb.tensor1d(new float[] { 0.9906240105628967f, 0.6248329877853394f, 0f, 1.0224634408950806f });

            var forgetBias = alb.scalar(1.0f);



            LSTMCellFunc lstm1 = (Tensor data, Tensor c, Tensor h) =>
        alb.basicLSTMCell(forgetBias, lstmKernel1, lstmBias1, data, c, h);


            LSTMCellFunc lstm2 = (Tensor data, Tensor c, Tensor h) =>
        alb.basicLSTMCell(forgetBias, lstmKernel2, lstmBias2, data, c, h);


            Tensor[] cs = new Tensor[]{
                      alb.zeros(new int[]{1, lstmBias1.Shape[0] / 4}),
      alb.zeros (new int[]{1, lstmBias2.Shape[0] / 4})
            };

            Tensor[] hs = new Tensor[]{
                      alb.zeros(new int[]{1, lstmBias1.Shape[0] / 4}),
      alb.zeros (new int[]{1, lstmBias2.Shape[0] / 4})
            };

            var onehot = alb.buffer(new int[] { 1, 2 });
            onehot.Set(1.0f, 0, 0);
            var output = alb.multiRNNCell(new LSTMCellFunc[] { lstm1, lstm2 },
                onehot.toTensor(), cs, hs);


            Assert.AreEqual(output[0].State.Get(), -0.7440074682235718f);
            Assert.AreEqual(output[1].State.Get(), 0.7460772395133972f);
            Assert.AreEqual(output[0].Output.Get(), -0.5802832245826721f);
            Assert.AreEqual(output[1].Output.Get(), 0.5745711922645569f);
        }

        [TestMethod]
        public void basicLSTMCell_with_batch_2()
        {
            ENV.engine = new Engine();
            var lstmKernel = alb.randomNormal(new int[] { 3, 4 });
            var lstmBias = alb.randomNormal(new int[] { 4 });
            var forgetBias = alb.scalar(1.0f);



            var data = alb.randomNormal(new int[] { 1, 2 });
            var batchedData = alb.concat2d(new Tensor[] { data, data }, 0);  // 2x2
            var c = alb.randomNormal(new int[] { 1, 1 });
            var batchedC = alb.concat2d(new Tensor[] { c, c }, 0);  // 2x1
            var h = alb.randomNormal(new int[] { 1, 1 });
            var batchedH = alb.concat2d(new Tensor[] { h, h }, 0);  // 2x1


            var outs = alb.basicLSTMCell(
        forgetBias, lstmKernel, lstmBias, batchedData, batchedC, batchedH);
            var newC = outs.State;
            var newH = outs.Output;

            Assert.AreEqual(newC.Get(0, 0), newC.Get(1, 0));
            Assert.AreEqual(newH.Get(0, 0), newH.Get(1, 0));
        }

    }
}
