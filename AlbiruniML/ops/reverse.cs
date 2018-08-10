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
        ///  Reverses a `1D Tensor` along a specified axis
        /// </summary>
        /// <param name="x">The input tensor to be reversed.</param> 
        /// <returns></returns>
        public static Tensor reverse1d(this Tensor x)
        {
            Util.assert(x.Rank == 1, "Error in reverse1D: x must be rank 1");
            return reverse(x, new int[] { 0 });
        }

        /// <summary>
        ///  Reverses a `2D Tensor` along a specified axis
        /// </summary>
        /// <param name="x">The input tensor to be reversed.</param>
        /// <param name="axis">The set of dimensions to reverse. Must be in the
        ///  range [-rank(x), rank(x)). Defaults to all axes.</param>
        /// <returns></returns>
        public static Tensor reverse2d(this Tensor x, int[] axis)
        {
            Util.assert(x.Rank == 2, "Error in reverse2D: x must be rank 2");
            return reverse(x, axis);
        }

        /// <summary>
        ///  Reverses a `3D Tensor` along a specified axis
        /// </summary>
        /// <param name="x">The input tensor to be reversed.</param>
        /// <param name="axis">The set of dimensions to reverse. Must be in the
        ///  range [-rank(x), rank(x)). Defaults to all axes.</param>
        /// <returns></returns>
        public static Tensor reverse3d(this Tensor x, int[] axis)
        {
            Util.assert(x.Rank == 3, "Error in reverse3D: x must be rank 3");
            return reverse(x, axis);
        }
        /// <summary>
        ///  Reverses a `4D Tensor` along a specified axis
        /// </summary>
        /// <param name="x">The input tensor to be reversed.</param>
        /// <param name="axis">The set of dimensions to reverse. Must be in the
        ///  range [-rank(x), rank(x)). Defaults to all axes.</param>
        /// <returns></returns>
        public static Tensor reverse4d(this Tensor x, int[] axis)
        {
            Util.assert(x.Rank == 4, "Error in reverse4D: x must be rank 4");
            return reverse(x, axis);
        }

        /// <summary>
        /// Reverses a `Tensor` along a specified axis.
        /// </summary>
        /// <param name="x">The input tensor to be reversed.</param>
        /// <param name="axis">The set of dimensions to reverse. Must be in the
        ///  range [-rank(x), rank(x)). Defaults to all axes.</param>
        /// <returns></returns>
        public static Tensor reverse(this Tensor x, int[] axis)
        {
            if (x.Rank == 0)
            {
                return x.clone();
            }
            var axes = Util.parseAxisParam(axis, x.Shape);

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (  Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () => { return dy.reverse(axes); });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return
                    bk.reverse(x, axes);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            var res = e.runKernel(f, inputs, grad);
            return res.reshapeAs(x);
        }


    }
}
