using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public enum Reduction
    {
        NONE,
        MEAN,
        SUM,
        SUM_BY_NONZERO_WEIGHTS
    }
    public static partial class Ops
    {

        public static class loss
        {
            /// <summary>
            /// Computes the weighted loss between two tensors.
            /// </summary>
            /// <param name="losses">Tensor of shape `[batch_size, d1, ... dN]`.</param>
            /// <param name="weights"> Tensor whose rank is either 0, or the same rank as
            ///  `losses`, and must be broadcastable to `losses` (i.e., all
            ///  dimensions must be either `1`, or the same as the corresponding
            ///  `losses` dimension).</param>
            /// <param name="reduction">Type of reduction to apply to loss. Should be of type
            /// `Reduction`</param>
            /// <returns></returns>
            public static Tensor computeWeightedLoss(Tensor losses, Tensor weights = null,
             Reduction reduction = Reduction.SUM_BY_NONZERO_WEIGHTS)
            {
                var weightedLoss = (weights == null) ? losses : losses.mul(weights);

                if (reduction == Reduction.SUM)
                {
                    return weightedLoss.sum();
                }
                else
                    if (reduction == Reduction.MEAN)
                    {
                        return (weights == null) ? weightedLoss.mean() :
                                                   weightedLoss.sum().div(weights.sum());
                    }
                    else

                        if (reduction == Reduction.SUM_BY_NONZERO_WEIGHTS)
                        {
                            if (weights == null)
                            {
                                return weightedLoss.sum().div(Ops.scalar(losses.Size));
                            }
                            else
                            {
                                var numNonZeros = weights.notEqual(Ops.scalar(0)).sum();
                                return weightedLoss.sum().div(numNonZeros);
                            }
                        }
                        else //Reduction.NONE
                        {
                            return weightedLoss;
                        }

            }


            /// <summary>
            ///  Computes the absolute difference loss between two tensors.
            /// </summary>
            /// <param name="labels">The ground truth output tensor, same dimensions as
            ///'predictions'.</param>
            /// <param name="predictions"> The predicted outputs.</param>
            /// <param name="weights">Tensor whose rank is either 0, or the same rank as
            ///   `labels`, and must be broadcastable to `labels` (i.e., all dimensions
            ///   must be either `1`, or the same as the corresponding `losses`
            ///   dimension).</param>
            /// <param name="reduction">Type of reduction to apply to loss. Should be of type
            /// `Reduction`</param>
            /// <returns></returns>
            public static Tensor absoluteDifference(Tensor labels, Tensor predictions,
                Tensor weights = null, Reduction reduction = Reduction.SUM_BY_NONZERO_WEIGHTS)
            {

                var losses = labels.sub(predictions).abs();
                return computeWeightedLoss(losses, weights, reduction);
            }

            /// <summary>
            /// Computes the mean squared error between two tensors.
            /// </summary>
            /// <param name="labels">The ground truth output tensor, same dimensions as
            /// 'predictions'.</param>
            /// <param name="predictions"> The predicted outputs.</param>
            /// <param name="weights">Tensor whose rank is either 0, or the same rank as
            /// `labels`, and must be broadcastable to `labels` (i.e., all dimensions
            /// must be either `1`, or the same as the corresponding `losses`
            /// dimension).</param>
            /// <param name="reduction">Type of reduction to apply to loss. Should be of type
            /// `Reduction`</param>
            /// <returns></returns>
            public static Tensor meanSquaredError(Tensor labels, Tensor predictions,
             Tensor weights = null, Reduction reduction = Reduction.SUM_BY_NONZERO_WEIGHTS)
            {

                var losses = labels.squaredDifference(predictions);
                return computeWeightedLoss(losses, weights, reduction);
            }


            /// <summary>
            /// Computes the cosine distance loss between two tensors.
            /// </summary>
            /// <param name="labels">The ground truth output tensor, same dimensions as
            /// 'predictions'.</param>
            /// <param name="predictions"> The predicted outputs.</param>
            /// <param name="axis">The dimension along which the cosine distance is computed.</param>
            /// <param name="weights"> Tensor whose rank is either 0, or the same rank as
            /// `labels`, and must be broadcastable to `labels` (i.e., all dimensions
            /// must be either `1`, or the same as the corresponding `losses`
            /// dimension).</param>
            /// <param name="reduction">Type of reduction to apply to loss. Should be of type
            /// `Reduction`</param>
            /// <returns></returns>
            public static Tensor cosineDistance(Tensor labels, Tensor predictions, int axis,
           Tensor weights = null, Reduction reduction = Reduction.SUM_BY_NONZERO_WEIGHTS)
            {
                var one = Ops.scalar(1);
                var losses = one.sub(labels.mul(predictions).sum(new int[] { axis }, true));
                return computeWeightedLoss(losses, weights, reduction);
            }

            /// <summary>
            /// Computes the Hinge loss between two tensors.
            /// </summary>
            /// <param name="labels">The ground truth output tensor, same dimensions as
            /// 'predictions'.</param>
            /// <param name="predictions"> The predicted outputs.</param>
            /// <param name="weights">Tensor whose rank is either 0, or the same rank as
            /// `labels`, and must be broadcastable to `labels` (i.e., all dimensions
            /// must be either `1`, or the same as the corresponding `losses`
            /// dimension).</param>
            /// <param name="reduction">Type of reduction to apply to loss. Should be of type
            /// `Reduction`</param>
            /// <returns></returns>
            public static Tensor hingeLoss(Tensor labels, Tensor predictions,
          Tensor weights = null, Reduction reduction = Reduction.SUM_BY_NONZERO_WEIGHTS)
            {
                var one = Ops.scalar(1);
                // Convert binary labels to (-1, 1)
                labels = Ops.scalar(2).mul(labels).sub(one);
                var losses = one.sub(labels.mul(predictions)).relu();
                return computeWeightedLoss(losses, weights, reduction);
            }

            /// <summary>
            /// Computes the log loss between two tensors.
            /// </summary>
            /// <param name="labels">The ground truth output tensor, same dimensions as
            ///    'predictions'.</param>
            /// <param name="predictions">The predicted outputs.</param>
            /// <param name="weights">Tensor whose rank is either 0, or the same rank as
            ///  `labels`, and must be broadcastable to `labels` (i.e., all dimensions
            ///  must be either `1`, or the same as the corresponding `losses`
            ///  dimension).</param>
            /// <param name="epsilon">A small increment to avoid taking log of zero</param>
            /// <param name="reduction">Type of reduction to apply to loss. Should be of type
            ///   `Reduction`</param>
            /// <returns></returns>
            public static Tensor logLoss(Tensor labels, Tensor predictions,
        Tensor weights = null, float epsilon = 1e-7f, Reduction reduction = Reduction.SUM_BY_NONZERO_WEIGHTS)
            {
                var one = Ops.scalar(1);
                var epsilonScalar = Ops.scalar(epsilon);
                var losses = labels.mul(predictions.add(epsilonScalar).log())
                                   .neg()
                                   .sub(one.sub(labels).mul(
                                       one.sub(predictions).add(epsilonScalar).log()));
                return computeWeightedLoss(losses, weights, reduction);
            }



            private static Tensor sigmoidCrossEntropyWithLogits(Tensor labels, Tensor logits)
            {
                //
                // Implementation Details:
                //
                // For brevity, let `x = logits`, `z = labels`.  The logistic loss is
                //     z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
                //   = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
                //   = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
                //   = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
                //   = (1 - z) * x + log(1 + exp(-x))
                //   = x - x * z + log(1 + exp(-x))
                //
                //   For x < 0, to avoid overflow in exp(-x), we reformulate the above
                //     x - x * z + log(1 + exp(-x))
                //   = log(exp(x)) - x * z + log(1 + exp(-x))
                //   = - x * z + log(1 + exp(x))
                //
                // Hence, to ensure stability and avoid overflow, the implementation uses
                // this equivalent formulation:
                //     max(x, 0) - x * z + log(1 + exp(-abs(x)))
                //
                var maxOutput = logits.relu();
                var outputXTarget = logits.mul(labels);
                var sigmoidOutput = logits.abs().neg().exp().log1p();

                return maxOutput.sub(outputXTarget).add(sigmoidOutput);
            }

            /// <summary>
            /// Computes the sigmoid cross entropy loss between two tensors.
            ///
            /// If labelSmoothing is nonzero, smooth the labels towards 1/2:
            ///
            ///   newMulticlassLabels = multiclassLabels * (1 - labelSmoothing)
            ///                        + 0.5 * labelSmoothing
            /// </summary>
            /// <param name="multiClassLabels">The ground truth output tensor of shape
            /// [batch_size, num_classes], same dimensions as 'predictions'.</param>
            /// <param name="logits">The predicted outputs.</param>
            /// <param name="weights">Tensor whose rank is either 0, or the same rank as
            ///   `labels`, and must be broadcastable to `labels` (i.e., all dimensions
            ///   must be either `1`, or the same as the corresponding `losses`
            ///   dimension).</param>
            /// <param name="labelSmoothing">If greater than 0, then smooth the labels.</param>
            /// <param name="reduction">Type of reduction to apply to loss. Should be of type
            ///  `Reduction`</param>
            /// <returns></returns>
            public static Tensor sigmoidCrossEntropy(Tensor multiClassLabels, Tensor logits, Tensor weights = null, float labelSmoothing = 0,
                Reduction reduction = Reduction.SUM_BY_NONZERO_WEIGHTS)
            {




                if (labelSmoothing > 0)
                {
                    var labelSmoothingScalar = scalar(labelSmoothing);
                    var one = scalar(1);
                    var half = scalar(0.5f);

                    multiClassLabels = multiClassLabels.mul(one.sub(labelSmoothingScalar))
                                           .add(half.mul(labelSmoothingScalar));
                }
                var losses = sigmoidCrossEntropyWithLogits(multiClassLabels, logits);

                return computeWeightedLoss(losses, weights, reduction);
            }


            /// <summary>
            ///Computes softmax cross entropy between logits and labels.
            ///
            /// Measures the probability error in discrete classification tasks in which
            /// the classes are mutually exclusive (each entry is in exactly one class).
            /// For example, each CIFAR-10 image is labeled with one and only one label: an
            /// image can be a dog or a truck, but not both.
            ///
            /// `NOTE`: While the classes are mutually exclusive, their probabilities need
            /// not be. All that is required is that each row of labels is a valid
            /// probability distribution. If they are not, the computation of the gradient
            /// will be incorrect.
            ///
            /// `WARNING`: This op expects unscaled logits, since it performs a softmax on
            /// logits internally for efficiency. Do not call this op with the output of
            /// softmax, as it will produce incorrect results.
            ///
            /// logits and labels must have the same shape, e.g. [batch_size, num_classes]
            /// and the same dtype.
            /// </summary>
            /// <param name="labelst">The labels array.</param>
            /// <param name="logitst">The logits array.</param>
            /// <param name="dim">The dimension softmax would be performed on. Defaults to `-1`
            ///     which indicates the last dimension.</param>
            /// <returns></returns>
            public static Tensor softmaxCrossEntropy(Tensor labelst, Tensor logitst, int dim = -1)
            {
                if (dim == -1)
                {
                    dim = logitst.Rank - 1;
                }
                var customOp = customGrad(
                    (Tensor[] x) =>
                    {
                        var labels = x[0];
                        var logits = x[1];

                        var predictedProbs = logits.softmax(dim);
                        var costVector =
                            scalar(1e-5f).add(predictedProbs).log().mul(labels).neg();
                        var value = costVector.sum(new int[] { dim });

                        CustomGradientResults res = new CustomGradientResults();
                        res.value = value;
                        res.gradFunc = (Tensor dy) =>
                        {
                            var dyShape = Util.expandShapeToKeepDim(dy.Shape, new int[] { dim });

                            return new List<Tensor>() { 
                            dy.reshape(dyShape).mul(labels.sub(predictedProbs)),
                        dy.reshape(dyShape).mul(predictedProbs.sub(labels))
                        };


                        };
                        return res;
                    }
                    );
                return customOp(new Tensor[] { labelst, logitst });
            }

        }
    }
}
