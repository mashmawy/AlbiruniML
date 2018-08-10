using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML.Optimizers
{
    public class AdadeltaOptimizer : Optimizer
    {
        private Tensor c;
        private Tensor epsilon;
        private Tensor rho;
        private Tensor oneMinusRho;

        private Dictionary<string, Variable> accumulatedGrads = new Dictionary<string, Variable>();

        private Dictionary<string, Variable> accumulatedUpdates = new Dictionary<string, Variable>();
        protected float learningRate;
        public AdadeltaOptimizer(float learningRate, float rho, float epsilon = 1e-8f)
            : base()
        {
            this.learningRate = learningRate;
            this.c = Ops.keep(Ops.scalar(-learningRate));
            this.epsilon = Ops.keep(Ops.scalar(epsilon));
            this.rho = Ops.keep(Ops.scalar(rho));
            this.oneMinusRho = Ops.keep(Ops.scalar(1 - rho));


        }




        public override void applyGradients(Dictionary<string, Tensor> variableGradients)
        {
            foreach (var item in variableGradients)
            {
                var value = ENV.engine.registeredVariables[item.Key];
                if (this.accumulatedGrads.ContainsKey(item.Key) == false)
                {

                    Ops.tidy( () =>
                    {
                        this.accumulatedGrads.Add(item.Key,
                            Ops.zerosLike(value).variable(false));
                    });
                }
                if (this.accumulatedUpdates.ContainsKey(item.Key) == false)
                {
                    Ops.tidy( () =>
                    {
                        this.accumulatedUpdates.Add(item.Key,
                            Ops.zerosLike(value).variable(false));
                    });
                }

                var gradient = variableGradients[item.Key];
                var accumulatedGrad = this.accumulatedGrads[item.Key];
                var accumulatedUpdate = this.accumulatedUpdates[item.Key];

                Ops.tidy( () =>
      {
          var newAccumulatedGrad =
              this.rho.mul(accumulatedGrad)
                  .add(this.oneMinusRho.mul(gradient.square()));

          var updates = accumulatedUpdate.add(this.epsilon)
                              .sqrt()
                              .div(accumulatedGrad.add(this.epsilon).sqrt())
                              .mul(gradient);

          var newAccumulatedUpdate =
              this.rho.mul(accumulatedUpdate)
                  .add(this.oneMinusRho.mul(updates.square()));

          this.accumulatedGrads[item.Key].assign(newAccumulatedGrad);
          this.accumulatedUpdates[item.Key].assign(newAccumulatedUpdate);

          var newValue = this.c.mul(updates).add(value);
          value.assign(newValue);
      });
            }
        }


        public void dispose()
        {
            this.c.dispose();
            this.epsilon.dispose();
            this.rho.dispose();
            this.oneMinusRho.dispose();
            if (this.accumulatedUpdates != null)
            {

                foreach (var item in accumulatedUpdates)
                {
                    item.Value.dispose();
                }
            }
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
