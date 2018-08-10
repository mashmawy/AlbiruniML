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
        ///  Computes `-1 * x` element-wise.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor neg(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            { 
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.neg();

                }); 
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.neg(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x); 
            return e.runKernel(f, inputs, grad);
        }
        
        /// <summary>
        /// Computes ceiling of input `Tensor` element-wise: `ceil(x)`
        /// </summary>
        /// <param name="x">The input Tensor.</param>
        /// <returns></returns>
        public static Tensor ceil(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            { 
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return zerosLike(dy);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.ceil(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
       
        /// <summary>
        ///  Computes floor of input `Tensor` element-wise: `floor(x)`.
        /// </summary>
        /// <param name="x">The input Tensor.</param>
        /// <returns></returns>
        public static Tensor floor(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            { 
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return zerosLike(dy);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.floor(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        /// Returns an element-wise indication of the sign of a number.
        /// </summary>
        /// <param name="x">The input Tensor.</param>
        /// <returns></returns>
        public static Tensor sign(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            { 
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return zerosLike(dy);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.sign(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        ///    * Computes round of input `Tensor` element-wise: `round(x)`.
        /// It implements banker's rounding.
        /// </summary>
        /// <param name="x">The input Tensor.</param>
        /// <returns></returns>
        public static Tensor round(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return zerosLike(dy);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.round(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }




        /// <summary>
        /// Computes exponential of the input `Tensor` element-wise. `e ^ x`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor exp(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var y = s[0];

                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                  return  dy.mulStrict(y);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return saved(bk.exp(x));
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
       
        /// <summary>
        /// Computes exponential of the input `Tensor` minus one element-wise.
        /// `e ^ x - 1`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor expm1(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            { 

                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.mulStrict(x.exp());

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.expm1(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        /// Computes natural logarithm of the input `Tensor` element-wise: `ln(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor log(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            { 
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.divStrict(x);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.log(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        /// Computes natural logarithm of the input `Tensor` plus one
        /// element-wise: `ln(1 + x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor log1p(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            { 
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.divStrict(x.add(scalar(1)));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.log1p(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
       
        /// <summary>
        /// Computes square root of the input `Tensor` element-wise: `y = sqrt(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor sqrt(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            { 

                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.divStrict(x.sqrt().mul(scalar(2)));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.sqrt(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
        
        /// <summary>
        ///  Computes reciprocal of square root of the input `Tensor` element-wise:
        /// `y = 1 / sqrt(x)`
        /// </summary>
        /// <param name="x"> The input tensor.</param>
        /// <returns></returns>
        public static Tensor rsqrt(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            { 

                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.divStrict(x.sqrt().mul(scalar(2)));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.rsqrt(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
        
        /// <summary>
        /// Computes square of `x` element-wise: `x ^ 2`
        /// </summary>
        /// <param name="x">The input Tensor.</param>
        /// <returns></returns>
        public static Tensor square(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            { 

                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    var dx = dy.mulStrict(x.mul(scalar(2)));
                    return  dx;

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.square(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }


        /// <summary>
        ///  Computes reciprocal of x element-wise: `1 / x`
        /// `y = 1 / sqrt(x)`
        /// </summary>
        /// <param name="x"> The input tensor.</param>
        /// <returns></returns>
        public static Tensor reciprocal(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {

                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.divStrict(x.square().neg());

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.reciprocal(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
        
        /// <summary>
        /// Computes absolute value element-wise: `abs(x)`
        /// </summary>
        /// <param name="x">The input Tensor.</param>
        /// <returns></returns>
        public static Tensor abs(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            { 

                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.mulStrict(x.step(-1));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.abs(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        ///  Clips values element-wise. `max(min(x, clipValueMax), clipValueMin)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="clipValueMin">Lower-bound of range to be clipped to.</param>
        /// <param name="clipValueMax">Upper-bound of range to be clipped to.</param>
        /// <returns></returns>
        public static Tensor clipByValue(this Tensor x, float clipValueMin, float clipValueMax)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var y = s[0];

                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.where(
                     x.greater(scalar(clipValueMin))
                         .logicalAnd(x.less(scalar(clipValueMax))),
                     zerosLike(dy));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.clip(x, clipValueMin, clipValueMax);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        ///  Computes rectified linear element-wise: `max(x, 0)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor relu(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.mulStrict(stepRes);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.relu(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        ///  Computes exponential linear element-wise, `x > 0 ? e ^ x - 1 : 0`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor elu(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    var y = s[0];
                    ForwardFunc fg = (IBackend bk, Func<Tensor, Tensor> saved) =>
                    {
                        return bk.eluDer(dy, y);
                    };
                    var inputsg = new Dictionary<string, Tensor>();
                    inputsg.Add("dy", dy);
                    inputsg.Add("y", y);
                    return ENV.engine.runKernel(fg, inputsg);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> save) =>
            {
              return  save( bk.elu(x));
              
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

       /// <summary>
        /// Computes scaled exponential linear element-wise.
       /// </summary>
        /// <param name="x">The input tensor.</param>
       /// <returns></returns>
        public static Tensor selu(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    var mask = x.greater(scalar(0));
                    var scaleAlpha = scalar((float)Util.SELU_SCALEALPHA);
                    var scale = scalar((float)Util.SELU_SCALE);
                    var greaterThanZeroDer = dy.mul(scale);
                    var lessEqualZeroDer = dy.mul(scaleAlpha).mul(x.exp());

                    return where(mask, greaterThanZeroDer, lessEqualZeroDer);
                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.selu(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        ///  Computes leaky rectified linear element-wise.
        ///       See
        /// [http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf](
        ///     http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="alpha">The scaling factor for negative values, defaults to 0.2.</param>
        /// <returns></returns>
        public static Tensor leakyRelu(this Tensor x, float alpha = 0.2f)
        {

            return  maximum(scalar(alpha).mul(x), x);
        }


        /// <summary>
        /// Computes leaky rectified linear element-wise with parametric alphas.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="alpha">Scaling factor for negative values.</param>
        /// <returns></returns>
        public static Tensor prelu(this Tensor x, Tensor alpha)
        {

            var zero = Ops.scalar(0);
            return Ops.maximum(zero, x).add(alpha.mul(Ops.minimum(zero, x)));
        }


        /// <summary>
        ///  Computes sigmoid element-wise, `1 / (1 + exp(-x))`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor sigmoid(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    var y = s[0];
                    return dy.mulStrict(y.mul(scalar(1).sub(y)));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return saved( bk.sigmoid(x));
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }


        /// <summary>
        /// Computes log sigmoid of the input `Tensor` element-wise:
        /// `logSigmoid(x)`. For numerical stability, we use `-alb.softplus(-x)`.
        /// </summary>
        /// <param name="x"> The input tensor.</param>
        /// <returns></returns>
        public static Tensor logSigmoid(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.mulStrict(x.neg().sigmoid());

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.softplus(x.neg()).neg();
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }



        /// <summary>
        /// Computes softplus of the input `Tensor` element-wise: `log(exp(x) + 1)`
        /// </summary>
        /// <param name="x"> The input tensor.</param>
        /// <returns></returns>
        public static Tensor softplus(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.mulStrict(x.sigmoid());

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.softplus(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }


        /// <summary>
        /// Computes sin of the input Tensor element-wise: `sin(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor sin(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return x.cos().mulStrict(dy);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.sin(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        ///  Computes cos of the input `Tensor` element-wise: `cos(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor cos(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return x.sin().neg().mulStrict(dy);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.cos(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
       
        /// <summary>
        /// Computes tan of the input `Tensor` element-wise, `tan(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor tan(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.divStrict(x.cos().square());

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.tan(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
       
        /// <summary>
        /// Computes asin of the input `Tensor` element-wise: `asin(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor asin(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.divStrict(sqrt(scalar(1).sub(x.square())));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.asin(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
        
        /// <summary>
        /// Computes acos of the input `Tensor` element-wise: `acos(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor acos(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.divStrict( sqrt( scalar(1).sub(x.square())))
                .neg();

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.acos(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
       
        /// <summary>
        /// Computes atan of the input `Tensor` element-wise: `atan(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor atan(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.divStrict( scalar(1).add(x.square()));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.atan(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
       
        /// <summary>
        /// Computes hyperbolic sin of the input `Tensor` element-wise: `sinh(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor sinh(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return x.cosh().mulStrict(dy);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.sinh(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
       
        /// <summary>
        ///  Computes hyperbolic cos of the input `Tensor` element-wise: `cosh(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor cosh(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return x.sinh().mulStrict(dy);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.cosh(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        /// Computes hyperbolic tangent of the input `Tensor` element-wise: `tanh(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor tanh(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    var y = s[0];
                    return scalar(1).sub(y.square()).mulStrict(dy);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return saved(bk.tanh(x));
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
      
        /// <summary>
        /// Computes inverse hyperbolic sin of the input `Tensor` element-wise:
        /// `asinh(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor asinh(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.divStrict( sqrt(Ops.scalar(1).add(x.square())));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return  bk.asinh(x) ;
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        ///  Computes the inverse hyperbolic cos of the input `Tensor` element-wise:
        ///`acosh(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor acosh(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.divStrict(sqrt(x.square().sub(Ops.scalar(1))));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.acosh(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        /// Computes inverse hyperbolic tan of the input `Tensor` element-wise:
        /// `atanh(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor atanh(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.divStrict(Ops.scalar(1).sub(x.square()));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.atanh(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }


        /// <summary>
        /// Computes gause error function of the input `Tensor` element-wise:
        /// `erf(x)`
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static Tensor erf(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var stepRes = x.step();
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return dy.mulStrict(Ops.scalar(2f / (float)Math.Sqrt(Math.PI))
.mul(x.square().neg().exp()));

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.erf(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }
       
        /// <summary>
        /// Computes step of the input `Tensor` element-wise: `x > 0 ? 1 : alpha * x`
        /// </summary>
        /// <param name="x"> The input tensor.</param>
        /// <param name="alpha">The gradient when input is negative.</param>
        /// <returns></returns>
        public static Tensor step(this Tensor x, float alpha = 0.0f)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var two = scalar(2);
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return zerosLike(dy);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.step(x,alpha);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

    }
}
