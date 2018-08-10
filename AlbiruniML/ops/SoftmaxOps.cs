using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public static partial class Ops
    {
        /// <summary>
        /// Computes the softmax normalized vector given the logits.
        /// </summary>
        /// <param name="logitst">The logits array.</param>
        /// <param name="dim">The dimension softmax would be performed on. Defaults to `-1`
        ///   which indicates the last dimension.</param>
        /// <returns></returns>
        public static Tensor softmax(this Tensor logitst, int dim = -1)
        {
            if (dim == -1)
            {
                dim = logitst.Rank - 1;
            }
            var customOp =  customGrad(
                (Tensor[] x) =>
                {
                    var logits = x[0];
                    var keepDims = true;
                    var lse = logits.logSumExp(new int[] { dim }, keepDims);
                    
                    var logResult = logits.sub(lse);
                    var y = logResult.exp();


                    CustomGradientResults res = new CustomGradientResults();
                    res.value = y;
                    res.gradFunc = (Tensor dy) =>
                    {
                        var dyTimesY = dy.mul(y);
                        var thesum = dyTimesY.sum(new int[] { dim }, true);
                        var themully = thesum.mul(y);
                        var theres = dyTimesY.sub(themully);
                        return new List<Tensor>() {
                           theres
                        };


                    };
                    return res;
                }
                );
            return customOp(new Tensor[] { logitst });
        }

    }
}
