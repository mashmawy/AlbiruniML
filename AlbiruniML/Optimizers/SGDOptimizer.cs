using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML.Optimizers
{
    public class SGDOptimizer:Optimizer
    {
        protected Tensor c;
        protected float learningRate;


        public SGDOptimizer(float learningRate)
            : base()
        {

            this.setLearningRate(learningRate);
        }

        public void setLearningRate(float learningRate)
        {
            this.learningRate = learningRate;
            if (this.c != null)
            {
                this.c.dispose();
            }
            this.c = Ops.keep(Ops.scalar(-learningRate));
        }
        public override void applyGradients(Dictionary<string, Tensor> variableGradients)
        {

            foreach (var item in variableGradients)
            {
                var gradient = item.Value;
                var value = ENV.engine.registeredVariables[item.Key];
                Ops.tidy( () =>
                {
                    var newValue = this.c.mul(gradient).add(value);
                    value.assign(newValue);
                    return value;
                });
            }
        }

        public void dispose()
        {
            this.c.dispose();
        }
    }
}
