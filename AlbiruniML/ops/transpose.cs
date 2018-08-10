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
        ///  Transposes the `Tensor`. Permutes the dimensions according to `perm`.
        ///  The returned `Tensor`'s dimension `i` will correspond to the input
        /// dimension `perm[i]`. If `perm` is not given, it is set to `[n-1...0]`,
        /// where `n` is the rank of the input `Tensor`. Hence by default, this
        /// operation performs a regular matrix transpose on 2-D input `Tensor`s.
        /// </summary>
        /// <param name="x">The tensor to transpose.</param>
        /// <param name="perm">The permutation of the dimensions of a.</param>
        /// <returns></returns>
        public static Tensor transpose(this Tensor x, int[] perm = null) 
        {
            if (perm == null)
            {
                perm = x.Shape.Select((s, i) => i).ToArray();
              perm=  perm.Reverse().ToArray();
            }
            if (x.Rank <= 1)
            {
                return x.clone();
            }

            if ( x.Rank != perm.Length)
            {
                throw new Exception("Error in transpose: rank of input " + x.Rank.ToString() +
            "must match length of perm " + perm.Length + " .");
            }
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var undoPerm = Util.getUndoAxesPermutation(perm);
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () => { return dy.transpose(undoPerm); });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
              
                return bk.transpose(x, perm);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad) ;
        }
    }
}
