using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML.Optimizers
{
    public class AdamaxOptimizer : Optimizer
    {

        private Tensor c;
        private Tensor eps;
        private Variable accBeta1;
        private Tensor beta1;
        private Tensor beta2;
        private Tensor decay;
        private Tensor oneMinusBeta1;
        private Tensor one;
        private Variable iteration;

        private Dictionary<string, Variable> accumulatedFirstMoment = new Dictionary<string, Variable>();
        protected float learningRate; 

        private Dictionary<string, Variable> accumulatedWeightedInfNorm = new Dictionary<string, Variable>();



        public AdamaxOptimizer(float learningRate, float beta1, float beta2,
   float epsilon = 1e-8f, float decay = 0.0f)
            : base()
        {
            this.c = Ops.keep(Ops.scalar(-learningRate));
            this.eps = Ops.keep(Ops.scalar(epsilon));
            // b1, b2 keep initial value of beta* hyperparameters.
            this.beta1 = Ops.keep(Ops.scalar(beta1));
            this.beta2 = Ops.keep(Ops.scalar(beta2));

            this.decay = Ops.keep(Ops.scalar(decay));

            Ops.tidy( () =>
            {
                this.iteration = Ops.scalar(0).variable(false);
                this.accBeta1 = Ops.scalar(beta1).variable(false);
            });

            this.oneMinusBeta1 = Ops.keep(Ops.scalar(1 - beta1));
            this.one = Ops.keep(Ops.scalar(1));
        }

        public override void applyGradients(Dictionary<string, Tensor> variableGradients)
        {
            Ops.tidy( () =>
            {
                var oneMinusAccBeta1 = this.one.sub(this.accBeta1);
                var lr = this.c.div(this.one.add(this.decay.mul(this.iteration)));

                foreach (var item in variableGradients)
                {
                    var value = ENV.engine.registeredVariables[item.Key];
                    if (!this.accumulatedFirstMoment.ContainsKey(item.Key) )
                    { 
                        this.accumulatedFirstMoment.Add(item.Key,
                            Ops.zerosLike(value).variable(false));
                    }
                    if (!this.accumulatedWeightedInfNorm.ContainsKey(item.Key))
                    { 
                        this.accumulatedWeightedInfNorm.Add(item.Key,
                            Ops.zerosLike(value).variable(false));
                    }

                    var gradient = variableGradients[item.Key];
                    var firstMoment = this.accumulatedFirstMoment[item.Key];
                    var weightedInfNorm = this.accumulatedWeightedInfNorm[item.Key];

                    var newFirstMoment =
                        this.beta1.mul(firstMoment).add(this.oneMinusBeta1.mul(gradient));

                    var ut0 = this.beta2.mul(weightedInfNorm);
                    var ut1 = gradient.abs();

                    var newWeightedInfNorm = ut0.maximum(ut1);

                    this.accumulatedFirstMoment[item.Key].assign(newFirstMoment);
                    this.accumulatedWeightedInfNorm[item.Key].assign(
                        newWeightedInfNorm);

                    var newValue =
                        lr.div(oneMinusAccBeta1)
                            .mul(newFirstMoment.div(this.eps.add(newWeightedInfNorm)))
                            .add(value);

                    value.assign(newValue);
                }

                this.iteration.assign(this.iteration.add(this.one));
                this.accBeta1.assign(this.accBeta1.mul(this.beta1));
            });
        }


        public void dispose()
        {
            this.c.dispose();
            this.eps.dispose();
            this.accBeta1.dispose();
            this.beta1.dispose();
            this.beta2.dispose();
            this.oneMinusBeta1.dispose();

            this.decay.dispose();
            this.iteration.dispose();

            this.one.dispose();
            if (this.accumulatedFirstMoment != null)
            {
                foreach (var variableName in this.accumulatedFirstMoment)
                {
                    variableName.Value.dispose();
                }
            }
            if (this.accumulatedWeightedInfNorm != null)
            {
                foreach (var variableName in this.accumulatedWeightedInfNorm)
                {
                    variableName.Value.dispose();
                }
            }
        }


    }
}
