using AlbiruniML.Optimizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public static partial class Ops
    {

        public static class train
        {
            /// <summary>
            ///  Constructs a `SGDOptimizer` that uses stochastic gradient descent.
            /// </summary>
            /// <param name="learningRate">The learning rate to use for the SGD algorithm.</param>
            /// <returns></returns>
            public static SGDOptimizer sgd(float learningRate)
            {
                return new SGDOptimizer(learningRate);
            }

            /// <summary>
            ///  Constructs a `MomentumOptimizer` that uses momentum gradient
            ///descent.
            /// See
            ///[http://proceedings.mlr.press/v28/sutskever13.pdf](
            ///http://proceedings.mlr.press/v28/sutskever13.pdf)
            /// </summary>
            /// <param name="learningRate">The learning rate to use for the Momentum gradient
            ///descent algorithm.</param>
            /// <param name="momentum">The momentum to use for the momentum gradient descent
            /// algorithm.</param>
            /// <param name="useNesterov"></param>
            /// <returns></returns>
            public static MomentumOptimizer momentum(float learningRate, float momentum,
                bool useNesterov = false)
            {
                return new MomentumOptimizer(learningRate, momentum, useNesterov);
            }

            /// <summary>
            ///  Constructs a `RMSPropOptimizer` that uses RMSProp gradient
            /// descent. This implementation uses plain momentum and is not centered
            /// version of RMSProp.
            /// 
            ///  See
            ///[http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf](
            ///http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
            /// </summary>
            /// <param name="learningRate">The learning rate to use for the RMSProp gradient
            ///descent algorithm.</param>
            /// <param name="decay">The discounting factor for the history/coming gradient</param>
            /// <param name="momentum">The momentum to use for the RMSProp gradient descent algorithm.</param>
            /// <param name="epsilon">Small value to avoid zero denominator.</param>
            /// <param name="centered ">If true, gradients are normalized by the estimated
            /// variance of the gradient..</param>
            /// <returns></returns>
            public static RMSPropOptimizer rmsprop(float learningRate, float decay = .9f,
                float momentum = 0.0f, float epsilon = 1e-8f, bool centered = false)
            {
                return new RMSPropOptimizer(learningRate, decay, momentum, epsilon, centered);
            }

            /// <summary>
            ///  Constructs a `AdamOptimizer` that uses the Adam algorithm.
            /// See [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
            /// </summary>
            /// <param name="learningRate">The learning rate to use for the Adam gradient
            ///descent algorithm.</param>
            /// <param name="beta1">The exponential decay rate for the 1st moment estimates.</param>
            /// <param name="beta2"> The exponential decay rate for the 2nd moment estimates.</param>
            /// <param name="epsilon">A small constant for numerical stability.</param>
            /// <returns></returns>
            public static AdamOptimizer adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
            {
                return new AdamOptimizer(learningRate, beta1, beta2, epsilon);
            }

            /// <summary>
            /// Constructs a `AdadeltaOptimizer` that uses the Adadelta algorithm.
            /// See [https://arxiv.org/abs/1212.5701](https://arxiv.org/abs/1212.5701
            /// </summary>
            /// <param name="learningRate">The learning rate to use for the Adadelta gradient
            /// descent algorithm.</param>
            /// <param name="rho">The learning rate decay over each update.</param>
            /// <param name="epsilon"> A constant epsilon used to better condition the grad update.</param>
            /// <returns></returns>
            public static AdadeltaOptimizer adadelta(float learningRate = .001f, float rho = .95f, float
                epsilon = 1e-8f)
            {
                return new AdadeltaOptimizer(learningRate, rho, epsilon);
            }

            /// <summary>
            /// Constructs a `AdamaxOptimizer` that uses the Adamax algorithm.
            /// See [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
            /// </summary>
            /// <param name="learningRate">The learning rate to use for the Adamax gradient
            /// descent algorithm.</param>
            /// <param name="beta1">The exponential decay rate for the 1st moment estimates.</param>
            /// <param name="beta2">The exponential decay rate for the 2nd moment estimates.</param>
            /// <param name="epsilon">A small constant for numerical stability.</param>
            /// <param name="decay">The learning rate decay over each update.</param>
            /// <returns></returns>
            public static AdamaxOptimizer adamax(float learningRate = 0.002f, float beta1 = 0.9f,
                float beta2 = 0.999f, float epsilon = 1e-8f,
      float decay = 0.0f)
            {
                return new AdamaxOptimizer(learningRate, beta1, beta2, epsilon, decay);
            }

            /// <summary>
            /// Constructs a `AdagradOptimizer` that uses the Adagrad algorithm.
            /// See
            /// [http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf](
            /// http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
            /// or
            /// [http://ruder.io/optimizing-gradient-descent/index.html#adagrad](
            /// http://ruder.io/optimizing-gradient-descent/index.html#adagrad)
            /// </summary>
            /// <param name="learningRate">The learning rate to use for the Adagrad gradient
            ///descent algorithm.</param>
            /// <param name="initialAccumulatorValue">Starting value for the accumulators, must be
            /// positive.</param>
            /// <returns></returns>
            public static AdagradOptimizer adagrad(float learningRate, float initialAccumulatorValue = 0.1f)
            {
                return new AdagradOptimizer(learningRate, initialAccumulatorValue);
            }
        }

    }
}
