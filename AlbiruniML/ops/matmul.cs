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
        /// Computes the dot product of two matrices, A * B. These must be matrices.
        /// </summary>
        /// <param name="a">First matrix in dot product operation.</param>
        /// <param name="b">Second matrix in dot product operation.</param>
        /// <param name="transposeA">If true, "a" is transposed before multiplication.</param>
        /// <param name="transposeB">If true, "b" is transposed before multiplication.</param>
        /// <returns></returns>
        public static Tensor matMul(this Tensor a, Tensor b, bool transposeA = false, bool transposeB = false)
        {
            var innerShapeA = transposeA ? a.Shape[0] : a.Shape[1];
            var innerShapeB = transposeB ? b.Shape[1] : b.Shape[0];


            Util.assert(
        a.Rank == 2 && b.Rank == 2,
        "Error in matMul: inputs must be Rank 2, got ranks " + a.Rank.ToString() +
            " and " + b.Rank.ToString() + ".");

            Util.assert(
                innerShapeA == innerShapeB,
                "Error in matMul: inner shapes (" + innerShapeA.ToString() + ") and (" +
                    innerShapeB.ToString() + " ) of Tensors with shapes " + a.Shape.ToString() + " and " +
                    b.Shape.ToString() + " and transposeA=" + transposeA.ToString() +
        " and transposeB=" + transposeB.ToString() + " must match.");
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                if (!transposeA && !transposeB)
                {

                    g.gradient.Add("a", () => { return dy.matMul(b, false, true); });
                    g.gradient.Add("b", () =>
                    {
                        return a.matMul(dy, true, false);
                    });

                }
                else if (!transposeA && transposeB)
                {
                    g.gradient.Add("a", () => { return dy.matMul(b, false, false); });
                    g.gradient.Add("b", () =>
                    {
                        return dy.matMul(a, true, false);
                    });
                }
                else if (transposeA && !transposeB)
                {
                    g.gradient.Add("a", () => { return b.matMul(dy, false, true); });
                    g.gradient.Add("b", () =>
                    {
                        return a.matMul(dy, false, false);
                    });
                }
                else
                {
                    g.gradient.Add("a", () => { return b.matMul(dy, true, true); });
                    g.gradient.Add("b", () =>
                    {
                        return dy.matMul(a, true, true);
                    });
                }

                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.matMul(a, b, transposeA, transposeB);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs, grad);

        }

        /// <summary>
        ///  Computes the dot product of a vector and a matrix, v * B.
        /// </summary>
        /// <param name="v">The vector in dot product operation.</param>
        /// <param name="matrix">The matrix in dot product operation.</param>
        /// <returns></returns>
        public static Tensor vectorTimesMatrix(this Tensor v, Tensor matrix)
        {
            Util.assert(
        v.Rank == 1,
        "Error in vectorTimesMatrix: first input must be Rank 1, but got " +
            "Rank " + v.Rank.ToString() + ".");
            Util.assert(
                matrix.Rank == 2,
                "Error in vectorTimesMatrix: second input must be Rank 2, but got " +
                    "Rank " + matrix.Rank.ToString() + ".");
            Util.assert(
                v.Size == matrix.Shape[0],
                "Error in vectorTimesMatrix: size of vector (" + v.Size.ToString() + ") " +
        "must match first dimension of matrix (" + matrix.Shape[0].ToString() + ")");


            return v.as2D(1, -1).matMul(matrix).as1D();
        }



        /// <summary>
        /// Computes the dot product of a matrix and vector, A * v.
        /// </summary>
        /// <param name="matrix">The matrix in dot product operation.</param>
        /// <param name="v">The vector in dot product operation.</param>
        /// <returns></returns>
        public static Tensor matrixTimesVector(this Tensor matrix, Tensor v)
        {
            Util.assert(
        v.Rank == 1,
        "Error in matrixTimesVector: second input must Rank 1, but got " +
            "Rank " + v.Rank.ToString() + ".");
            Util.assert(
                matrix.Rank == 2,
                "Error in matrixTimesVector: first input must be a Rank 2, but got " +
                    "Rank " + matrix.Rank.ToString() + ".");
            Util.assert(
                v.Size == matrix.Shape[1],
                "Error in matrixTimesVector: size of first Rank 1 input " + v.Size.ToString() + " " +
                    "must match inner dimension of second Rank 2 input");
            return matrix.matMul(v.as2D(-1, 1)).as1D();
        }

        /// <summary>
        ///  Computes the dot product of two vectors, v1 * v2.
        /// </summary>
        /// <param name="v1">The first vector in the dot product operation.</param>
        /// <param name="v2">The second vector in the dot product operation.</param>
        /// <returns></returns>
        public static Tensor dotProduct(this Tensor v1, Tensor v2)
        {
            Util.assert(
        v1.Rank == 1 && v2.Rank == 1,
        "Error in dotProduct: inputs must be Rank 1, but got ranks " +
            v1.Rank.ToString() + " and " + v2.Rank.ToString() + ".");
            Util.assert(
                v1.Size == v2.Size,
                "Error in dotProduct: size of inputs (" + v1.Size.ToString() + ") and (" +
                    v2.Size.ToString() + ") must match.");
            return v1.as2D(1, -1).matMul(v2.as2D(-1, 1)).asScalar();
        }

        /// <summary>
        /// Computes the outer product of two vectors, v1 and v2.
        /// </summary>
        /// <param name="v1">The first vector in the outer product operation.</param>
        /// <param name="v2">The second vector in the dot product operation.</param>
        /// <returns></returns>
        public static Tensor outerProduct(this Tensor v1, Tensor v2)
        {
            Util.assert(
    v1.Rank == 1 && v2.Rank == 1,
    "Error in outerProduct: inputs must be Rank 1, but got ranks " +
        v1.Rank.ToString() + " and " + v2.Rank.ToString() + ".");

            return v1.as2D(-1, 1).matMul(v2.as2D(1, -1));
        }



        /// <summary>
        /// Computes the dot product of two matrices and/or vectors, t1 and t2.
        /// </summary>
        /// <param name="t1">The first tensor in the dot operation.</param>
        /// <param name="t2">The second tensor in the dot operation.</param>
        /// <returns></returns>
        public static Tensor dot(this Tensor t1, Tensor t2)
        {
            Util.assert(
        (t1.Rank == 1 || t1.Rank == 2) && (t2.Rank == 1 || t2.Rank == 2),
        "Error in dot: inputs must all be Rank 1 or 2, but got ranks " +
            t1.Rank.ToString() + " and " + t2.Rank.ToString() + ".");

            var t1Inner = (t1.Rank == 1 ? t1.Size : t1.Shape[1]);
            var t2Inner = (t2.Rank == 1 ? t2.Size : t2.Shape[0]);

            Util.assert(
                t1Inner == t2Inner,
                "Error in dot: inner dimensions of inputs must match, but got " +
                    t1Inner.ToString() + " and " + t2Inner.ToString() + ".");

            if (t1.Rank == 1 && t2.Rank == 1)
            {
                return t1.as2D(1, -1).matMul(t2.as2D(-1, 1)).asScalar();
            }
            else if (t1.Rank == 1 && t2.Rank == 2)
            {
                return t1.as2D(1, -1).matMul(t2.as2D(t2.Shape[0], t2.Shape[1])).as1D();
            }
            else if (t1.Rank == 2 && t2.Rank == 1)
            {
                return t1.matMul(t2.as2D(-1, 1)).as1D();
            }
            else
            {
                return t1.matMul(t2.as2D(t2.Shape[0], t2.Shape[1]));
            }
        }
    }
}
