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
        /// Creates a new variable with the provided initial value.
        /// </summary>
        /// <param name="initialValue">Initial value for the tensor.</param>
        /// <param name="trainable">If true, optimizers are allowed to update it.</param>
        /// <param name="name">Name of the variable. Defaults to a unique id.</param>
        /// <returns></returns>
        public static Variable variable(this Tensor initialValue, bool trainable = true, string name=null)
        {
            return new Variable(initialValue, trainable, name);
        }

        /// <summary>
        /// Returns the truth value of (a != b) element-wise. Supports broadcasting.
        /// 
        /// We also expose `notEqualStrict` which has the same signature as this op and
        /// asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>
        public static Tensor notEqual(this Tensor a, Tensor b)
        { 
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.notEqual(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs);

        }

        /// <summary>
        /// Strict version of `notEqual` that forces `a` and `b` to be of the same
        /// shape.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>
        public static Tensor notEqualStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.notEqual(b);
        }
         

        /// <summary>
        /// Returns the truth value of (a less than b) element-wise. Supports broadcasting.
        /// 
        /// We also expose `lessStrict` which has the same signature as this op and
        /// asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>
        public static Tensor less(this Tensor a, Tensor b)
        {
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.less(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs);

        }

        /// <summary>
        /// Strict version of `less` that forces `a` and `b` to be of the same shape.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>
        public static Tensor lessStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.less(b);
        }


        /// <summary>
        ///  Returns the truth value of (a == b) element-wise. Supports broadcasting.
        /// 
        /// We also expose `equalStrict` which has the same signature as this op
        /// and asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>
        public static Tensor equal(this Tensor a, Tensor b)
        {
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.equal(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs);

        }

        /// <summary>
        /// Strict version of `equal` that forces `a` and `b` to be of the same shape.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>
        public static Tensor equalStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.equal(b);
        }

        /// <summary>
        /// Returns the truth value of (a less than = b) element-wise. Supports broadcasting.
        /// 
        /// We also expose `lessEqualStrict` which has the same signature as this op
        /// and asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>
        public static Tensor lessEqual(this Tensor a, Tensor b)
        {
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.lessEqual(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs);

        }

        /// <summary>
        /// Strict version of `lessEqual` that forces `a` and `b` to be of the same shape.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>
        public static Tensor lessEqualStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.lessEqual(b);
        }

        /// <summary>
        /// Returns the truth value of (a > b) element-wise. Supports broadcasting.
        ///
        /// We also expose `greaterStrict` which has the same signature as this
        /// op and asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>
        public static Tensor greater(this Tensor a, Tensor b)
        {
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.greater(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs);

        }

        /// <summary>
        /// Strict version of `greater` that forces `a` and `b` to be of the same shape.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>       /// <summary>
        /// Strict version of `greater` that forces `a` and `b` to be of the same shape.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>
        public static Tensor greaterStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.greater(b);
        }

        /// <summary>
        /// Returns the truth value of (a >= b) element-wise. Supports broadcasting.
        /// 
        /// We also expose `greaterEqualStrict` which has the same signature as this
        /// op and asserts that `a` and `b` are the same shape (does not broadcast).
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>
        public static Tensor greaterEqual(this Tensor a, Tensor b)
        {
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.greaterEqual(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs);

        }

        /// <summary>
        /// Strict version of `greaterEqual` that forces `a` and `b` to be of the same shape.
        /// </summary>
        /// <param name="a">The first input tensor.</param>
        /// <param name="b">The second input tensor.</param>
        /// <returns></returns>
        public static Tensor greaterEqualStrict(this Tensor a, Tensor b)
        {
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("shape dose not match");
                }
            }
            return a.greaterEqual(b);
        }
    }
}
