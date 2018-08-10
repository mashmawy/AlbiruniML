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
        /// Gram-Schmidt orthogonalization.
        /// </summary>
        /// <param name="xs">The vectors to be orthogonalized, in one of the two following
        /// formats:
        /// - An Array of `Tensor1D`.
        /// - A `Tensor2D`, i.e., a matrix, in which case the vectors are the rows
        ///   of `xs`.
        /// In each case, all the vectors must have the same length and the length
        /// must be greater than or equal to the number of vectors.</param>
        /// <returns> The orthogonalized and normalized vectors or matrix.
        ///  Orthogonalization means that the vectors or the rows of the matrix
        ///  are orthogonal (zero inner products). Normalization means that each
        ///  vector or each row of the matrix has an L2 norm that equals `1`.</returns>
        public static Tensor gramSchmidt(this Tensor  xs)
        {
            List<Tensor> ys = new List<Tensor>();
            var xs1d = xs.split(  xs.Shape[0], 0).Select(x => squeeze(x, new int[]{0})).ToArray();
            for (var i = 0; i < xs1d.Length; ++i)
            {
                ys.Add(tidy( () =>
                {
                    var x = xs1d[i];
                    if (i > 0)
                    {
                        for (var j = 0; j < i; ++j)
                        {
                            var proj = sum(ys[j].mulStrict(x)).mul(ys[j]);
                            x = x.sub(proj);
                        }
                    }
                    return x.div(norm(x, NormType.euclidean));
                }));
            }
            return stack(ys.ToArray(), 0);
        }

        /// <summary>
        /// Gram-Schmidt orthogonalization.
        /// </summary>
        /// <param name="xs">The vectors to be orthogonalized, in one of the two following
        /// formats:
        /// - An Array of `Tensor1D`.
        /// - A `Tensor2D`, i.e., a matrix, in which case the vectors are the rows
        ///   of `xs`.
        /// In each case, all the vectors must have the same length and the length
        /// must be greater than or equal to the number of vectors.</param>
        /// <returns> The orthogonalized and normalized vectors or matrix.
        ///  Orthogonalization means that the vectors or the rows of the matrix
        ///  are orthogonal (zero inner products). Normalization means that each
        ///  vector or each row of the matrix has an L2 norm that equals `1`.</returns>
        public static Tensor[] gramSchmidt(Tensor[] xs)
        { 
            List<Tensor> ys = new List<Tensor>();
            var xs1d = xs;
            for (var i = 0; i < xs.Length; ++i)
            {
                ys.Add(tidy( () =>
                {
                    var x = xs1d[i];
                    if (i > 0)
                    {
                        for (var j = 0; j < i; ++j)
                        {
                            var proj = sum(ys[j].mulStrict(x)).mul(ys[j]);
                            x = x.sub(proj);
                        }
                    }
                    return x.div(norm(x, NormType.euclidean));
                }));
            }
            return ys.ToArray(); 
        }
    }
}
