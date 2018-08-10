using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public enum NormType
    {
        One,
        Inf,
        NegativeInf,
        Two,
        euclidean,
        fro
    }
    public static partial class Ops
    {
        /// <summary>
        /// Computes the norm of scalar, vectors, and matrices.
        ///This function can compute several different vector norms (the 1-norm, the
        ///Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0)
        ///and matrix norms (Frobenius, 1-norm, and inf-norm).
        /// </summary>
        /// <param name="x">The input array.</param>
        /// <param name="ord">Optional. Order of the norm. Supported norm types are
        /// following:
        ///
        ///  | ord          | norm for matrices         | norm for vectors
        ///  |--------------|---------------------------|---------------------
        ///  |euclidean     |euclidean norm             |2-norm
        ///  |fro           |Frobenius norm	            |
        ///  |Inf           |max(sum(abs(x), axis=1))   |max(abs(x))
        ///  |NegativeInf   |min(sum(abs(x), axis=1))   |min(abs(x))
        ///  |One           |max(sum(abs(x), axis=0))   |sum(abs(x))
        ///  |Two           |                           |sum(abs(x)^2)^1/2*</param>
        /// <param name="axis">Optional. If axis is null (the default), the input is
        /// considered a vector and a single vector norm is computed over the entire
        /// set of values in the Tensor, i.e. norm(x, ord) is equivalent
        /// to norm(x.reshape([-1]), ord). If axis is a integer, the input
        /// is considered a batch of vectors, and axis determines the axis in x
        /// over which to compute vector norms. If axis is a 2-tuple of integer it is
        /// considered a batch of matrices and axis determines the axes in NDArray
        /// over which to compute a matrix norm.</param>
        /// <param name="keepDims"> Optional. If true, the norm have the same dimensionality
        /// as the input.</param>
        /// <returns></returns>
        public static Tensor norm(this Tensor x, NormType ord, int[] axis = null, bool keepDims = false)
        {
            var norm = normImpl(x, ord, axis);
            var keepDimsShape = norm.Shape;
            if (keepDims)
            {
                var axes = Util.parseAxisParam(axis, x.Shape);
                keepDimsShape = Util.expandShapeToKeepDim(norm.Shape, axes);
            }
            return norm.reshape(keepDimsShape);

        }
        private static Tensor normImpl(  Tensor x, NormType p, int[] axis = null)
        {
            if (x.Rank == 0)
            {
                return x.abs();
            }

            // consider vector when no axis is specified
            if (x.Rank != 1 && axis == null)
            {
                return normImpl(x.reshape(new int[] { -1 }), p, axis);
            }

            // vector
            if (x.Rank == 1 || axis.Length == 1)
            {
                if (p == NormType.One)
                {
                    return x.abs().sum(axis);
                }
                if (p == NormType.Inf)
                {
                    return x.abs().max(axis);
                }
                if (p == NormType.NegativeInf)
                {
                    return x.abs().min(axis);
                }
                if (p == NormType.euclidean || p == NormType.Two)
                {
                    // norm(x, 2) = sum(abs(xi) ^ 2) ^ 1/2
                    return x.abs().pow(Ops.scalar(2)).sum(axis).sqrt();
                }

            }
            // matrix (assumption axis[0] < axis[1])
            if (axis.Length == 2)
            {
                if (p == NormType.One)
                {
                    return x.abs().sum(new int[] { axis[0] }).max(new int[] { axis[1] - 1 });
                }
                if (p == NormType.Inf)
                {
                    return x.abs().sum(new int[] { axis[1] }).max(new int[] { axis[0] });
                }
                if (p == NormType.NegativeInf)
                {
                    return x.abs().sum(new int[] { axis[1] }).min(new int[] { axis[0] });
                }
                if (p == NormType.fro || p == NormType.euclidean)
                {
                    // norm(x) = sqrt(sum(pow(x, 2)))
                    return x.square().sum(axis).sqrt();
                }

            }
            throw new Exception("Error in norm: invalid axis");

        }
    }
}
