using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public class LSTMResult
    {
        public Tensor State { get; set; }
        public Tensor Output { get; set; }
        public LSTMResult(Tensor state, Tensor output)
        {
            this.State = state;
            this.Output = output;
        }
    }
    public delegate LSTMResult LSTMCellFunc(Tensor data, Tensor c, Tensor h);
    public static partial class Ops
    {

        /// <summary>
        ///  Computes the next states and outputs of a stack of LSTMCells.
        /// 
        ///  Each cell output is used as input to the next cell.
        /// 
        ///  Returns `[cellState, cellOutput]`.
        /// 
        ///  Derived from tf.contrib.rn.MultiRNNCell.
        /// </summary>
        /// <param name="lstmCells">Array of LSTMCell functions.</param>
        /// <param name="data">The input to the cell.</param>
        /// <param name="c">Array of previous cell states.</param>
        /// <param name="h">Array of previous cell outputs.</param>
        /// <returns></returns>
        public static LSTMResult[] multiRNNCell(LSTMCellFunc[] lstmCells, Tensor data, Tensor[] c, Tensor[] h)
        {
            var input = data;
            List<LSTMResult> newStates = new List<LSTMResult>();
            for (var i = 0; i < lstmCells.Length; i++)
            {
                var output = lstmCells[i](input, c[i], h[i]);
                newStates.Add(output);
                input = output.Output;
            }

            return newStates.ToArray();
        }


        /// <summary>
        /// Computes the next state and output of a BasicLSTMCell.
        /// Returns `[newC, newH]`.
        ///
        /// Derived from tf.contrib.rnn.BasicLSTMCell.
        /// </summary>
        /// <param name="forgetBias">Forget bias for the cell.</param>
        /// <param name="lstmKernel">The weights for the cell.</param>
        /// <param name="lstmBias">The bias for the cell.</param>
        /// <param name="data">The input to the cell.</param>
        /// <param name="c">Previous cell state.</param>
        /// <param name="h">Previous cell output.</param>
        /// <returns></returns>
        public static LSTMResult basicLSTMCell(Tensor forgetBias, Tensor lstmKernel, Tensor lstmBias, Tensor data, Tensor c, Tensor h)
        {
            var combined = data.concat(h, 1);
            var weighted = combined.matMul(lstmKernel);
            var res = weighted + lstmBias;

            // i = input_gate, j = new_input, f = forget_gate, o = output_gate
            var batchSize = res.Shape[0];
            var sliceCols = res.Shape[1] / 4;
            var sliceSize =Ops.shape( batchSize, sliceCols) ;
            var i = res.slice(Ops.shape(0, 0), sliceSize).sigmoid();
            var j = res.slice(Ops.shape(0, sliceCols), sliceSize).tanh();
            var f = (res.slice(Ops.shape(0, sliceCols * 2), sliceSize) + forgetBias).sigmoid();
            var o = res.slice(Ops.shape(0, sliceCols * 3), sliceSize).sigmoid();

            var newC = (i * j) + (  c *  f );
            var newH = newC.tanh() * (o);

            return new LSTMResult(newC, newH);
        }
        /// <summary>
        /// Computes the next state and output of a basicGRUCell.
        /// Returns `[newC, newH]`. 
        /// </summary>
        /// <param name="forgetBias">Forget bias for the cell.</param>
        /// <param name="gruKernel">The weights for the cell.</param>
        /// <param name="gruBias">The bias for the cell.</param>
        /// <param name="data">The input to the cell.</param>
        /// <param name="c">Previous cell state.</param>
        /// <param name="h">Previous cell output.</param>
        /// <returns></returns>
        public static LSTMResult basicGRUCell( Tensor gruKernel, Tensor gruBias, Tensor data,   Tensor h)
        {
            var combined = data.concat(h, 1);
            var weighted = combined.matMul(gruKernel);
            var res = weighted + gruBias;

            // i = input_gate, j = new_input, f = forget_gate, o = output_gate
            var batchSize = res.Shape[0];
            var sliceCols = res.Shape[1] / 3;
            var sliceSize = Ops.shape(batchSize, sliceCols);
            var z = res.slice(Ops.shape(0, 0), sliceSize).sigmoid();
            var r = res.slice(Ops.shape(0, sliceCols), sliceSize).sigmoid();
            var hd = (res.slice(Ops.shape(0, sliceCols * 2), sliceSize));
            hd = hd.mulStrict(r).tanh();

            var newC = ((1 - z) * h) + (z * hd);
            var newH = newC;

            return new LSTMResult(newC, newH);
        }
    }
}
