using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML.Optimizers
{
    public abstract class Optimizer
    {
        public Tensor minimize(Func<Tensor> f, bool returnCost=false, List<Variable> varList=null)
        {
            var vg = this.computeGradients(f, varList);
            var value = vg.Item1;
            var grads = vg.Item2;

            this.applyGradients(grads);
            foreach (var item in grads)
            {
                item.Value.dispose();
            }
            if (returnCost)
            {
                return value ;
            }
            else
            {
                value.dispose();
                return null;
            }
        }

        private Tuple<Tensor, Dictionary<string, Tensor>> computeGradients(Func<Tensor> f, List<Variable> varList = null)
        {
            return Ops.variableGrads(f, varList);
        }

        public abstract void applyGradients(Dictionary<string, Tensor> variableGradients);
    }
}
