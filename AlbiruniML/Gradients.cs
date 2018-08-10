using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public static partial class Ops
    {
        public static List<Tensor> gradScope(string name, Func<List<Tensor>> fn)
        {
            return Ops.tidy(name, fn, true);

        }
         

        public static Func<Tensor, Tensor, Tensor> grad(Func<Tensor, Tensor> f)
        {

            return (Tensor x, Tensor dy) =>
            {
                if (dy != null)
                {

                    var d = ENV.engine.gradients(() => f(x), new Tensor[] { x }, dy);
                    d.value.dispose();
                    return d.Grades[0];
                }
                else
                {

                    var d = ENV.engine.gradients(() => f(x), new Tensor[] { x });
                    d.value.dispose();
                    return d.Grades[0];
                }
            };
        }
        public static Func<Tensor[], Tensor, Tensor[]> grads(Func<Tensor[], Tensor> f)
        {

            return (Tensor[] args, Tensor dy) =>
            {
                if (dy != null)
                {

                    var d = ENV.engine.gradients(() => f(args), args, dy);
                    d.value.dispose();
                    return d.Grades.ToArray();
                }
                else
                {

                    var d = ENV.engine.gradients(() => f(args), args);
                    d.value.dispose();
                    return d.Grades.ToArray();
                }
            };
        }


        public static Tuple<Tensor, Dictionary<string, Tensor>> variableGrads(Func<Tensor> f, List<Variable> varList = null)
        {
            if (varList == null)
            {
                // Get all of the trainable variables.
                varList = new List<Variable>();
                foreach (var varName in ENV.engine.registeredVariables)
                {
                    varList.Add(varName.Value);
                }
            }
            // Prune non-trainable variables.
            var originalVarCount = varList.Count;
            varList = varList.Where(variable => variable.trainable).ToList();
            var allowNoGradients = true;
            var gv =
                ENV.engine.gradients(f, varList.ToArray(), null, allowNoGradients);
            var value = gv.value;
            var grads = gv.Grades;
            var namedGrads =new  Dictionary<string, Tensor>();
            int i = 0;
            foreach (var item in varList)
            {
                namedGrads.Add(item.Name, grads[i]);
                i++;
            }
            return new Tuple<Tensor, Dictionary<string, Tensor>>(value, namedGrads);

        }


        public static Func<Tensor[], Tensor> customGrad(CustomGradientFunc f)
        {
            return ENV.engine.customGrad(f, f.ToString());
        }
    }
}
