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
        ///Compute the moving average of a variable.
        ///
        /// Without zeroDebias, the moving average operation is defined by:
        ///   `v += delta`
        /// where
        ///   `delta = (1 - decay) * (x - v)`
        ///
        /// With zeroDebias (default), the `delta` term is scaled to debias the
        /// effect of the (assumed) zero-initialization of `v`.
        ///   `delta /= (1 - decay ^ step)`
        ///
        /// For more details on the zero-debiasing algorithm, see:
        ///   https://arxiv.org/abs/1412.6980
        ///
        /// Note that this function is completely stateless and does not keep track of
        /// step count. The step count needs to be maintained by the caller and passed
        /// in as `step`.
        /// </summary>
        /// <param name="v">The current moving average value.</param>
        /// <param name="x">New input value, must have the same shape and dtype as `v`.</param>
        /// <param name="decay">The decay factor. Typical values are 0.95 and 0.99.</param>
        /// <param name="step">Step count.</param>
        /// <param name="zeroDebias">Whether zeroDebias is to be performed (default: `true`).</param>
        /// <returns>The new moving average value.</returns>
        public static Tensor movingAverage(this Tensor v, Tensor x, Tensor decay, Tensor step = null, bool zeroDebias = true)
        {
            var one = Ops.scalar(1);
            var oneMinusDecay = one.sub(decay);

            var update = x.sub(v).mul(oneMinusDecay);
            if (zeroDebias)
            {
                Util.assert(
                    step != null, "When using zeroDebias: true, step is required.");

                update = update.div(one.sub(pow(decay, step)));
            }
            return v.add(update);
        }
    }
}
