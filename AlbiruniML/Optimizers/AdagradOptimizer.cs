using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML.Optimizers
{
    public class AdagradOptimizer : Optimizer
    {
        private Tensor c;
        private Tensor epsilon;
        private float initialAccumulatorValue;

        private Dictionary<string, Variable> accumulatedGrads= new Dictionary<string,Variable>();
        private float learningRate;
        public AdagradOptimizer(float learningRate, float initialAccumulatorValue = 0.1f)
            : base()
        {
            this.initialAccumulatorValue = initialAccumulatorValue;
            this.learningRate = learningRate;
            this.c = Ops.keep(Ops.scalar(-learningRate));
            this.epsilon = Ops.keep(Ops.scalar(1e-8f));
        }


        public override void applyGradients(Dictionary<string, Tensor> variableGradients)
        {

            foreach (var item in variableGradients)
            {
                var value = ENV.engine.registeredVariables[item.Key];


                if (!accumulatedGrads.ContainsKey(item.Key))
                {
                    Ops.tidy( () =>
                    {
                        this.accumulatedGrads.Add(item.Key,
                            Ops.fill(value.Shape, this.initialAccumulatorValue)
                  .variable(false));
                    });
                }
                var accumulatedGrad = this.accumulatedGrads[item.Key];
                var gradient = item.Value;
                Ops.tidy( () =>
                {
                    var newAccumulatedGrad = accumulatedGrad.add(gradient.square());
                    this.accumulatedGrads[item.Key].assign(newAccumulatedGrad);

                    var newValue =
                        this.c
                            .mul(gradient.div(newAccumulatedGrad.add(this.epsilon).sqrt()))
                            .add(value);
                    value.assign(newValue);
                });
            }
        }


        public void dispose()
        {
            this.epsilon.dispose();
            this.c.dispose();
            if (this.accumulatedGrads != null)
            {

                foreach (var item in accumulatedGrads)
                {
                    item.Value.dispose();
                }
            }
        }
    }
}
