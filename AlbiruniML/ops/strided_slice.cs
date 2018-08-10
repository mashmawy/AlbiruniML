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
        /// Extracts a strided slice of a tensor.
        ///
        /// Roughly speaking, this op extracts a slice of size (end-begin)/stride from
        /// the given input_ tensor. Starting at the location specified by begin the
        /// slice continues by adding stride to the index until all dimensions are not
        /// less than end. Note that a stride can be negative, which causes a reverse
        /// slice.
        /// </summary>
        /// <param name="x">The tensor to stride slice.</param>
        /// <param name="begin">The coordinates to start the slice from.</param>
        /// <param name="end">The coordinates to end the slice at.</param>
        /// <param name="strides">The size of the slice.</param>
        /// <param name="beginMask">If the ith bit of begin_mask is set, begin[i] is ignored
        ///  and the fullest possible range in that dimension is used instead.</param>
        /// <param name="endMask">If the ith bit of end_mask is set, end[i] is ignored
        ///  and the fullest possible range in that dimension is used instead.</param>
        /// <returns></returns>
        public static Tensor stridedSlice(this Tensor x, int[] begin, int[] end,
            int[] strides, int beginMask = 0, int endMask = 0)
        {
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.stridedSlice(
x, begin, end, strides, beginMask, endMask);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs);
        }
    }
}
