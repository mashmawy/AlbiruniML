using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML.Optimizers
{
    public class RMSPropOptimizer : Optimizer
    {
        private Tensor c;
        private Tensor epsilonScalar;
        private Tensor decayScalar;
        private Tensor momentumScalar;
        private Tensor oneMinusDecay;
        private bool centered;
        private Dictionary<string, Variable> accumulatedMeanSquares = new Dictionary<string, Variable>();
        private Dictionary<string, Variable> accumulatedMeanGrads = new Dictionary<string, Variable>();
        private Dictionary<string, Variable> accumulatedMoments = new Dictionary<string, Variable>();
        protected float learningRate;

        public RMSPropOptimizer(float learningRate, float decay = 0.9f, float momentum = 0.0f,
     float epsilon = 1e-8f, bool centered = false)
            : base()
        {
            this.c = Ops.keep(Ops.scalar(learningRate));
            this.epsilonScalar = Ops.keep(Ops.scalar(epsilon));
            this.decayScalar = Ops.keep(Ops.scalar(decay));
            this.momentumScalar = Ops.keep(Ops.scalar(momentum));
            this.oneMinusDecay = Ops.keep(Ops.scalar(1 - decay));
            this.centered = centered;
        }
        public override void applyGradients(Dictionary<string, Tensor> variableGradients)
        {

            foreach (var item in variableGradients)
            {
                var value = ENV.engine.registeredVariables[item.Key];

                if (!accumulatedMeanSquares.ContainsKey(item.Key))
                {
                    Ops.tidy( () =>
                    {
                        this.accumulatedMeanSquares.Add(item.Key, Ops.zerosLike(value).variable(false));
                    });
                }
                if (!accumulatedMeanGrads.ContainsKey(item.Key))
                {
                    Ops.tidy( () =>
                    {
                        this.accumulatedMeanGrads.Add(item.Key, Ops.zerosLike(value).variable(false));
                    });
                }
                if (!accumulatedMoments.ContainsKey(item.Key))
                {
                    Ops.tidy( () =>
                    {
                        this.accumulatedMoments.Add(item.Key, Ops.zerosLike(value).variable(false));
                    });
                }


                var accumulatedMeanSquare = this.accumulatedMeanSquares[item.Key];
                var accumulatedMeanGrad = this.accumulatedMeanGrads[item.Key];
                var accumulatedMoment = this.accumulatedMoments[item.Key];
                var gradient = variableGradients[item.Key];

                Ops.tidy( () =>
                {

                    var newAccumulatedMeanSquare =
             this.decayScalar.mul(accumulatedMeanSquare)
 .add(this.oneMinusDecay.mul(gradient.square()));
                    if (this.centered)
                    {
                        // Centered gradient
                        var newAccumulatedMeanGrad =
                            this.decayScalar.mul(accumulatedMeanGrad)
                                .add(this.oneMinusDecay.mul(gradient));

                        var newAccumulatedMoments =
                            this.momentumScalar.mul(accumulatedMoment)
                                .add(this.c.mul(gradient).div(
                                    newAccumulatedMeanSquare
                                        .sub(newAccumulatedMeanGrad.square().add(
                                            this.epsilonScalar))
                                        .sqrt()));

                        this.accumulatedMeanSquares[item.Key].assign(
                            newAccumulatedMeanSquare);
                        this.accumulatedMeanGrads[item.Key].assign(
                            newAccumulatedMeanGrad);
                        this.accumulatedMoments[item.Key].assign(newAccumulatedMoments);

                        var newValue = value.sub(newAccumulatedMoments);
                        value.assign(newValue);
                    }
                    else
                    {
                        var newAccumulatedMeanGrad =
                this.decayScalar.mul(accumulatedMeanGrad)
                    .add(this.oneMinusDecay.mul(gradient));

                        var newAccumulatedMoments =
                            this.momentumScalar.mul(accumulatedMoment)
                                .add(this.c.mul(gradient).div(
                                    newAccumulatedMeanSquare
                                        .sub(newAccumulatedMeanGrad.square().add(
                                            this.epsilonScalar))
                                        .sqrt()));

                        this.accumulatedMeanSquares[item.Key].assign(
                            newAccumulatedMeanSquare);
                        this.accumulatedMeanGrads[item.Key].assign(
                            newAccumulatedMeanGrad);
                        this.accumulatedMoments[item.Key].assign(newAccumulatedMoments);

                        var newValue = value.sub(newAccumulatedMoments);
                        value.assign(newValue);
                    }

                });


            }
        }

        public void dispose()
        {
            this.c.dispose();
            this.epsilonScalar.dispose();
            this.decayScalar.dispose();
            this.momentumScalar.dispose();
            this.oneMinusDecay.dispose();
            if (this.accumulatedMeanSquares != null)
            {
                foreach (var variableName in this.accumulatedMeanSquares)
                {
                    variableName.Value.dispose();
                }
            } if (this.accumulatedMeanGrads != null)
            {
                foreach (var variableName in this.accumulatedMeanGrads)
                {
                    variableName.Value.dispose();
                }
            }
            if (this.accumulatedMoments != null)
            {
                foreach (var variableName in this.accumulatedMoments)
                {
                    variableName.Value.dispose();
                }
            }
        }
    }
}
