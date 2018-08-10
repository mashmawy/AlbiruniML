using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML.Optimizers
{
    public class MomentumOptimizer : SGDOptimizer
    {
        protected new float  learningRate;
        private float momentum;
        private bool useNesterov;

        private Tensor m;
        private Dictionary<string, Variable> accumulations;

        public MomentumOptimizer(float learningRate, float momentum, bool useNesterov)
            : base(learningRate)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
            this.useNesterov = useNesterov;

            this.accumulations = new Dictionary<string, Variable>();

        }


        public override void applyGradients(Dictionary<string, Tensor> variableGradients)
        {

            this.m = Ops.scalar(momentum);
            foreach (var item in variableGradients)
            {
                var value = ENV.engine.registeredVariables[item.Key];


                if (!accumulations.ContainsKey(item.Key))
                { 
                    Ops.tidy(  () =>
                    {
                        this.accumulations.Add(item.Key, Ops.zerosLike(value).variable(false));
                    });
                }
                var accumulation = this.accumulations[item.Key];
                var gradient = item.Value;
                Ops.tidy(  () =>
                {
                    Tensor newValue = null;
                    var md = this.m.dataSync();
                    var accd = accumulation.dataSync();
                    var gd = gradient.dataSync();
                    var newAccumulation = this.m.mul(accumulation).add(gradient);
                    if (this.useNesterov)
                    {
                        newValue =
                            this.c.mul(gradient.add(newAccumulation.mul(this.m))).add(value);
                    }
                    else
                    {
                        newValue = this.c.mul(newAccumulation).add(value);
                    }
                    this.accumulations[item.Key].assign(newAccumulation);
                    value.assign(newValue);
                });
            }
        }

        public new void  dispose()
        {
            base.dispose();
            this.m.dispose();
            if (this.accumulations != null)
            {
                foreach (var variableName in this.accumulations)
                {
                    variableName.Value.dispose();
                }
            }
        }
        public void setMomentum(float momentum)
        {
            this.momentum = momentum;
        }
    }
}
