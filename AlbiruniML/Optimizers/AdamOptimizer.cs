using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML.Optimizers
{
    public class AdamOptimizer : Optimizer
    {

        private Tensor c;
        private Tensor eps;
        private Tensor beta1;
        private Tensor beta2;

        private Variable accBeta1;
        private Variable accBeta2;
        private Tensor oneMinusBeta1;
        private Tensor oneMinusBeta2;
        private Tensor one;


        private Dictionary<string, Variable> accumulatedFirstMoment = new Dictionary<string, Variable>();
        protected float learningRate;

        private Dictionary<string, Variable> accumulatedSecondMoment = new Dictionary<string, Variable>();




        public AdamOptimizer(float learningRate, float beta1, float beta2,
   float epsilon = 1e-8f)
            : base()
        {
            this.c = Ops.keep(Ops.scalar(-learningRate));
            this.eps = Ops.keep(Ops.scalar(epsilon));
            // b1, b2 keep initial value of beta* hyperparameters.
            this.beta1 = Ops.keep(Ops.scalar(beta1));
            this.beta2 = Ops.keep(Ops.scalar(beta2));

            Ops.tidy( () =>
            {
                this.accBeta2 = Ops.scalar(beta2).variable(false);
                this.accBeta1 = Ops.scalar(beta1).variable(false);
            });

            this.oneMinusBeta1 = Ops.keep(Ops.scalar(1 - beta1));
            this.oneMinusBeta2 = Ops.keep(Ops.scalar(1 - beta2));
            this.one = Ops.keep(Ops.scalar(1));
        }


        public override void applyGradients(Dictionary<string, Tensor> variableGradients)
        {
            Ops.tidy( () =>
            {
                var oneMinusAccBeta1 = this.one.sub(this.accBeta1);
                var oneMinusAccBeta2 = this.one.sub(this.accBeta2);

                foreach (var item in variableGradients)
                {
                    var value = ENV.engine.registeredVariables[item.Key];
                    if (this.accumulatedFirstMoment.ContainsKey(item.Key) == false)
                    {
                        var trainable = false;
                        this.accumulatedFirstMoment.Add(item.Key,
                            Ops.zerosLike(value).variable(trainable));
                    }
                    if (this.accumulatedSecondMoment.ContainsKey(item.Key) == false)
                    {
                        var trainable = false;
                        this.accumulatedSecondMoment.Add(item.Key,
                            Ops.zerosLike(value).variable(trainable));
                    }

                    var gradient = variableGradients[item.Key];
                    var firstMoment = this.accumulatedFirstMoment[item.Key];
                    var secondMoment = this.accumulatedSecondMoment[item.Key];

                    var newFirstMoment =
                        this.beta1.mul(firstMoment).add(this.oneMinusBeta1.mul(gradient));
                    var newSecondMoment =
                        this.beta2.mul(secondMoment)
                            .add(this.oneMinusBeta2.mul(gradient.square()));

                    var biasCorrectedFirstMoment = newFirstMoment.div(oneMinusAccBeta1);
                    var biasCorrectedSecondMoment = newSecondMoment.div(oneMinusAccBeta2);

                    this.accumulatedFirstMoment[item.Key].assign(newFirstMoment);
                    this.accumulatedSecondMoment[item.Key].assign(newSecondMoment);

                    var newValue = this.c
                                         .mul(biasCorrectedFirstMoment.div(this.eps.add(
                                             biasCorrectedSecondMoment.sqrt())))
                                         .add(value);
                    value.assign(newValue);
                }

                this.accBeta1.assign(this.accBeta1.mul(this.beta1));
                this.accBeta2.assign(this.accBeta2.mul(this.beta2));
            });
        }


        public void dispose()
        {
            this.c.dispose();
            this.eps.dispose();
            this.beta1.dispose();
            this.beta2.dispose();
            this.accBeta1.dispose();
            this.accBeta2.dispose();
            this.oneMinusBeta1.dispose();
            this.oneMinusBeta2.dispose();
            this.one.dispose();
            this.one.dispose();
            if (this.accumulatedFirstMoment != null)
            {
                foreach (var variableName in this.accumulatedFirstMoment)
                {
                    variableName.Value.dispose();
                }
            }
            if (this.accumulatedSecondMoment != null)
            {
                foreach (var variableName in this.accumulatedSecondMoment)
                {
                    variableName.Value.dispose();
                }
            }
        }

    }
}
