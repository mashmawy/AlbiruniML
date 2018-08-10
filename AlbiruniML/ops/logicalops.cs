using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public static partial class Ops
    {
        public static Tensor logicalNot(this Tensor x)
        {
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.logicalNot(x);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs);

        }
        public static Tensor logicalAnd(this Tensor a, Tensor b)
        {
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.logicalAnd(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs);

        }

        public static Tensor logicalOr(this Tensor a, Tensor b)
        {
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.logicalOr(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs);

        }

        public static Tensor logicalXor(this Tensor a, Tensor b)
        {
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.logicalXor(a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs);

        }

        public static Tensor where(this Tensor condition, Tensor a, Tensor b)
        {
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.where(condition, a, b);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("condition", condition);
            inputs.Add("a", a);
            inputs.Add("b", b);
            return e.runKernel(f, inputs);

        }
    }
}
