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
        /// Adds two `Tensor`s element-wise, A + B. Supports broadcasting.
        /// 
        /// We also expose `addStrict` which has the same signature as this op and
        /// asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first `Tensor` to add.</param>
        /// <param name="b">The second `Tensor` to add. Must have the same type as `a`.</param>
        /// <returns></returns>
        public static Tensor add(this Tensor a, Tensor b)
        {
            var outShape =
         Util.assertAndGetBroadcastShape(a.Shape, b.Shape);

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("a", () =>
                {
                    var res = dy;
                    var reduceAxes = Util.getReductionAxes(a.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        res = res.sum(reduceAxes);
                    }
                    return res.reshape(a.Shape);

                });
                g.gradient.Add("b", () =>
                {
                    var res = dy;
                    var reduceAxes = Util.getReductionAxes(b.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        res = res.sum(reduceAxes);
                    }
                    return res.reshape(b.Shape);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.Add(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs, grad);

        }

        /// <summary>
        /// Adds two `Tensor`s element-wise, A + B.
        /// 
        /// Inputs must be the same shape. For broadcasting support, use add() instead.
        /// </summary>
        /// <typeparam name="Tensor">Tensor extends Tensor</typeparam>
        /// <param name="a">The first Tensor to add element-wise.</param>
        /// <param name="b">The second Tensor to add element-wise.</param>
        /// <returns></returns>
        public static Tensor addStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.add(b);
        }

        /// <summary>
        ///Subtracts two `Tensor`s element-wise, A - B. Supports broadcasting.
        ///
        /// We also expose `subStrict` which has the same signature as this op and
        /// asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first `Tensor` to subtract from.</param>
        /// <param name="b">The second `Tensor` to be subtracted. Must have the same dtype as `a`.</param>
        /// <returns></returns>
        public static Tensor sub(this Tensor a, Tensor b)
        {
            var outShape =
         Util.assertAndGetBroadcastShape(a.Shape, b.Shape);

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("a", () =>
                {
                    var res = dy;
                    var reduceAxes = Util.getReductionAxes(a.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        res = res.sum(reduceAxes);
                    }
                    return res.reshape(a.Shape);

                });
                g.gradient.Add("b", () =>
                {
                    var res = dy;
                    var reduceAxes = Util.getReductionAxes(b.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        res = res.sum(reduceAxes);
                    }
                    return res.neg().reshape(b.Shape);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.Subtract(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs, grad);

        }

        /// <summary>
        /// Subtracts two `Tensor`s element-wise, A - B. Inputs must
        /// be the same shape.
        /// 
        /// For broadcasting support, use sub() instead.
        /// </summary>
        /// <param name="a">The first Tensor to subtract element-wise.</param>
        /// <param name="b">The second Tensor to subtract element-wise.</param>
        /// <returns></returns>
        public static Tensor subStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.sub(b);
        }

        /// <summary>
        /// Computes the power of one `Tensor` to another. Supports broadcasting.
        /// 
        /// Given a `Tensor` x and a `Tensor` y, this operation computes x^y for
        /// corresponding elements in x and y. The result's dtype will be the upcasted
        /// type of the `base` and `exp` dtypes.
        /// </summary>
        /// <param name="baset">The base `Tensor` to pow element-wise.</param>
        /// <param name="exp">The exponent `Tensor` to pow element-wise.</param>
        /// <returns></returns>
        public static Tensor pow(this Tensor baset, Tensor exp)
        {
            var outShape =
        Util.assertAndGetBroadcastShape(baset.Shape, exp.Shape);
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var y = s[0];
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("baset", () =>
                {
                    var res = dy.mul(exp.mul(y.div(baset)));
                    var reduceAxes =
                        Util.getReductionAxes(baset.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        res = res.sum(reduceAxes);
                    }
                    return res.reshape(baset.Shape);
                });
                g.gradient.Add("exp", () =>
                {
                    var res = dy.mul(y.mul(baset.log()));
                    var reduceAxes = Util.getReductionAxes(exp.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        res = res.sum(reduceAxes);
                    }
                    return res.reshape(exp.Shape);
                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {

                return saved(bk.Pow(baset, exp));
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("baset", baset);
            inputs.Add("exp", exp);
            return e.runKernel(f, inputs, grad);

        }

        /// <summary>
        /// Computes the power of one `Tensor` to another. Inputs must
        /// be the same shape.
        ///
        /// For broadcasting support, use pow() instead.
        /// </summary>
        /// <param name="a">The base tensor to pow element-wise</param>
        /// <param name="b">The exponent tensor to pow element-wise.</param>
        /// <returns></returns>
        public static Tensor powStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.pow(b);
        }

        /// <summary>
        /// Multiplies two `Tensor`s element-wise, A * B. Supports broadcasting.
        /// 
        /// We also expose `mulStrict` which has the same signature as this op and
        /// asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first tensor to multiply.</param>
        /// <param name="b">The second tensor to multiply.</param>
        /// <returns></returns>
        public static Tensor mul(this Tensor a, Tensor b)
        {
            var outShape =
         Util.assertAndGetBroadcastShape(a.Shape, b.Shape);

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("a", () =>
                {
                    var res = dy.mul(b);
                    var reduceAxes = Util.getReductionAxes(a.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        return res.sum(reduceAxes).reshape(a.Shape);
                    }
                    return res;

                });
                g.gradient.Add("b", () =>
                {
                    var res = dy.mul(a);
                    var reduceAxes = Util.getReductionAxes(b.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        return res.sum(reduceAxes).reshape(b.Shape);
                    }
                    return res;

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.Multiply(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs, grad);

        }

        /// <summary>
        /// Multiplies two `Tensor`s element-wise, A * B.
        /// Inputs must be the same shape. For broadcasting support, use mul().
        /// </summary>
        /// <param name="a">The first tensor to multiply.</param>
        /// <param name="b">The second tensor to multiply.</param>
        /// <returns></returns>
        public static Tensor mulStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.mul(b);
        }


        /// <summary>
        /// Divides two `Tensor`s element-wise, A / B. Supports broadcasting.
        ///
        /// We also expose `divStrict` which has the same signature as this op and
        /// asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first tensor as the numerator.</param>
        /// <param name="b">The second tensor as the denominator.</param>
        /// <returns></returns>
        public static Tensor div(this Tensor a, Tensor b)
        {
            var outShape =
         Util.assertAndGetBroadcastShape(a.Shape, b.Shape);

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("a", () =>
                {
                    var res = dy.div(b);
                    var reduceAxes = Util.getReductionAxes(a.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        return res.sum(reduceAxes).reshape(a.Shape);
                    }
                    return res;

                });
                g.gradient.Add("b", () =>
                {
                    var res = dy.mul(a);
                    var reduceAxes = Util.getReductionAxes(b.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        res = res.sum(reduceAxes).reshape(b.Shape);
                    }
                    var tmp = b.square();
                    return res.div(tmp).neg();

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.Divide(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs, grad);

        }

        /// <summary>
        ///  Divides two `Tensor`s element-wise, A / B. Inputs must
        ///  be the same shape.
        /// </summary>
        /// <param name="a">The first tensor as the numerator for element-wise division.</param>
        /// <param name="b">The second tensor as the denominator for element-wise division.</param>
        /// <returns></returns>
        public static Tensor divStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.div(b);
        }


        /// <summary>
        /// Returns the mod of a and b element-wise.
        /// `floor(x / y) * y + mod(x, y) = x`
        /// Supports broadcasting. 
        /// We also expose `modStrict` which has the same signature as this op and
        /// asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor. Must have the same type as `a`.</param>
        /// <returns></returns>
        public static Tensor mod(this Tensor a, Tensor b)
        {
            var outShape =
         Util.assertAndGetBroadcastShape(a.Shape, b.Shape);

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("a", () =>
                {
                    var reduceAxes = Util.getReductionAxes(a.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        return dy.sum(reduceAxes).reshape(a.Shape);
                    }
                    return dy;

                });
                g.gradient.Add("b", () =>
                {
                    var res = dy.mul(a.div(b).floor().neg());
                    var reduceAxes = Util.getReductionAxes(b.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        res = res.sum(reduceAxes).reshape(b.Shape);
                    }
                    return res;

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.mod(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs, grad);

        }
         
        /// <summary>
        /// Returns the mod of a and b (a less than b ? a : b) element-wise. Inputs must
        /// be the same shape. For broadcasting support, use mod().
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        /// <returns></returns>
        public static Tensor modStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.mod(b);
        }
         
        /// <summary>
        /// Returns the min of a and b (`a less than b ? a : b`) element-wise.
        /// Supports broadcasting.
        ///
        /// We also expose `minimumStrict` which has the same signature as this op and
        /// asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        /// <returns></returns>
        public static Tensor minimum(this Tensor a, Tensor b)
        {

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("a", () =>
                {
                    return dy.mul(a.lessEqual(b));

                });
                g.gradient.Add("b", () =>
                {
                    return dy.mul(a.greater(b));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.minimum(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs, grad);

        }

        /// <summary>
        /// Returns the min of a and b (`a les than b ? a : b`) element-wise. Inputs must
        /// be the same shape. For broadcasting support, use minimum().
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        /// <returns></returns>
        public static Tensor minimumStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.minimum(b);
        }

        /// <summary>
        /// Returns the max of a and b (`a > b ? a : b`) element-wise.
        /// Supports broadcasting.
        ///
        /// We also expose `maximumStrict` which has the same signature as this op and
        /// asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        /// <returns></returns>
        public static Tensor maximum(this Tensor a, Tensor b)
        {

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("a", () =>
                {
                    return dy.mul(a.greaterEqual(b));

                });
                g.gradient.Add("b", () =>
                {
                    return dy.mul(a.less(b));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.maximum(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs, grad);

        }

        /// <summary>
        /// Returns the max of a and b (`a > b ? a : b`) element-wise. Inputs must
        /// be the same shape. For broadcasting support, use maximum().
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        /// <returns></returns>
        public static Tensor maximumStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.maximum(b);
        }

        /// <summary>
        /// Returns (a - b) * (a - b) element-wise.
        /// Supports broadcasting.
        /// 
        /// We also expose `squaredDifferenceStrict` which has the same signature as
        /// this op and asserts that `a` and `b` are the same shape (does not
        /// broadcast).
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        /// <returns></returns>
        public static Tensor squaredDifference(this Tensor a, Tensor b)
        {

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var two = scalar(2);
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("a", () =>
                {
                    return dy.mul(a.sub(b).mul(two));

                });
                g.gradient.Add("b", () =>
                {
                    return dy.mul(b.sub(a).mul(two));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.squaredDifference(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        /// Returns (a - b) * (a - b) element-wise.
        /// 
        /// Inputs must be the same shape. For broadcasting support, use
        /// squaredDifference() instead
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        /// <returns></returns>
        public static Tensor squaredDifferenceStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.squaredDifference(b);
        }

        /// <summary>
        ///    * Computes arctangent of `Tensor`s a / b element-wise: `atan2(a, b)`.
        /// Supports broadcasting.
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        /// <returns></returns>
        public static Tensor atan2(this Tensor a, Tensor b)
        {
            var outShape =
         Util.assertAndGetBroadcastShape(a.Shape, b.Shape);

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("a", () =>
                {
                    var d = add(square(a), square(b));
                    var res = dy.mul(b.div(d));
                    var reduceAxes = Util.getReductionAxes(a.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        res = res.sum(reduceAxes);
                    }
                    return res.reshape(a.Shape);
                });
                g.gradient.Add("b", () =>
                {
                    var d = add(square(a), square(b));
                    var res = neg(dy.mul(a.div(d)));
                    var reduceAxes = Util.getReductionAxes(b.Shape, outShape);
                    if (reduceAxes.Length > 0)
                    {
                        res = res.sum(reduceAxes);
                    }
                    return res.reshape(b.Shape);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.atan2(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs, grad);

        }



    }
}
