using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public static partial class Ops
    {
         
        /// <summary>
        /// Batch normalization, strictly for 2D. For the more relaxed version, see
        /// batchNormalization".
        /// </summary>
        /// <param name="x">The input Tensor.</param>
        /// <param name="mean">A mean Tensor.</param>
        /// <param name="variance">A variance Tensor.</param>
        /// <param name="varianceEpsilon">A small float number to avoid dividing by 0.</param>
        /// <param name="scale">A scale Tensor.</param>
        /// <param name="offset">An offset Tensor.</param>
        /// <returns></returns>
        public static Tensor batchNormalization2d(this Tensor x, Tensor mean, Tensor variance,
            float varianceEpsilon = 0.001f, Tensor scale = null, Tensor offset = null)
        {
            Util.assert(
        x.Rank == 2,
        "Error in batchNormalization3D: x must be Rank 2 but got Rank " +
            "${x.Rank}.");
            Util.assert(
                mean.Rank == 2 || mean.Rank == 1,
                "Error in batchNormalization3D: mean must be Rank 2 or Rank 1 but " +
                    "got Rank ${mean.Rank}.");
            Util.assert(
                variance.Rank == 2 || variance.Rank == 1,
                "Error in batchNormalization3D: variance must be Rank 2 or Rank 1 " +
                    "but got Rank ${variance.Rank}.");
            if (scale != null)
            {
                Util.assert(
                    scale.Rank == 2 || scale.Rank == 1,
                    "Error in batchNormalization3D: scale must be Rank 2 or Rank 1 " +
                        "but got Rank ${scale.Rank}.");
            }
            if (offset != null)
            {
                Util.assert(
                    offset.Rank == 2 || offset.Rank == 1,
                    "Error in batchNormalization3D: offset must be Rank 2 or Rank 1 " +
                        "but got Rank ${offset.Rank}.");
            }

            return batchNormalization(
        x, mean, variance, varianceEpsilon, scale, offset) as Tensor;
        }
         
        /// <summary>
        /// Batch normalization, strictly for 3D. For the more relaxed version, see
        /// batchNormalization".
        /// </summary>
        /// <param name="x">The input Tensor.</param>
        /// <param name="mean">A mean Tensor.</param>
        /// <param name="variance">A variance Tensor.</param>
        /// <param name="varianceEpsilon">A small float number to avoid dividing by 0.</param>
        /// <param name="scale">A scale Tensor.</param>
        /// <param name="offset">An offset Tensor.</param>
        /// <returns></returns>
        public static Tensor batchNormalization3d(this Tensor x, Tensor mean, Tensor variance,
            float varianceEpsilon = 0.001f, Tensor scale = null, Tensor offset = null)
        {
            Util.assert(
        x.Rank == 3,
        "Error in batchNormalization3D: x must be Rank 3 but got Rank " +
            "${x.Rank}.");
            Util.assert(
                mean.Rank == 3 || mean.Rank == 1,
                "Error in batchNormalization3D: mean must be Rank 3 or Rank 1 but " +
                    "got Rank ${mean.Rank}.");
            Util.assert(
                variance.Rank == 3 || variance.Rank == 1,
                "Error in batchNormalization3D: variance must be Rank 3 or Rank 1 " +
                    "but got Rank ${variance.Rank}.");
            if (scale != null)
            {
                Util.assert(
                    scale.Rank == 3 || scale.Rank == 1,
                    "Error in batchNormalization3D: scale must be Rank 3 or Rank 1 " +
                        "but got Rank ${scale.Rank}.");
            }
            if (offset != null)
            {
                Util.assert(
                    offset.Rank == 3 || offset.Rank == 1,
                    "Error in batchNormalization3D: offset must be Rank 3 or Rank 1 " +
                        "but got Rank ${offset.Rank}.");
            }

            return batchNormalization(
        x, mean, variance, varianceEpsilon, scale, offset) as Tensor;
        }
         
        /// <summary>
        /// Batch normalization, strictly for 4D. For the more relaxed version, see
        /// batchNormalization".
        /// </summary>
        /// <param name="x">The input Tensor.</param>
        /// <param name="mean">A mean Tensor.</param>
        /// <param name="variance">A variance Tensor.</param>
        /// <param name="varianceEpsilon">A small float number to avoid dividing by 0.</param>
        /// <param name="scale">A scale Tensor.</param>
        /// <param name="offset">An offset Tensor.</param>
        /// <returns></returns>
        public static Tensor batchNormalization4d(this Tensor x, Tensor mean, Tensor variance,
            float varianceEpsilon = 0.001f, Tensor scale = null, Tensor offset = null)
        {
            Util.assert(
                   x.Rank == 4,
                   "Error in batchNormalization4D: x must be Rank 4 but got Rank " +
                       "${x.Rank}.");
            Util.assert(
                mean.Rank == 4 || mean.Rank == 1,
                "Error in batchNormalization4D: mean must be Rank 4 or Rank 1 but " +
                    "got Rank ${mean.Rank}.");
            Util.assert(
                variance.Rank == 4 || variance.Rank == 1,
                "Error in batchNormalization4D: variance must be Rank 4 or Rank 1 " +
                    "but got Rank ${variance.Rank}.");
            if (scale != null)
            {
                Util.assert(
                    scale.Rank == 4 || scale.Rank == 1,
                    "Error in batchNormalization4D: scale must be Rank 4 or Rank 1 " +
                        "but got Rank ${scale.Rank}.");
            }
            if (offset != null)
            {
                Util.assert(
                    offset.Rank == 4 || offset.Rank == 1,
                    "Error in batchNormalization4D: offset must be Rank 4 or Rank 1 " +
                        "but got Rank ${offset.Rank}.");
            }
            return  batchNormalization(
        x, mean, variance, varianceEpsilon, scale, offset) as Tensor;
        }
         
        /// <summary>
        /// Batch normalization.
        /// 
        /// As described in
        /// [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167).
        /// 
        /// Mean, variance, scale, and offset can be of two
        /// shapes:
        ///   - The same shape as the input.
        ///   - In the common case, the depth dimension is the last dimension of x, so
        ///     the values would be an "Tensor1D" of shape [depth].
        /// </summary>
        /// <param name="x">The input Tensor.</param>
        /// <param name="mean">A mean Tensor.</param>
        /// <param name="variance">A variance Tensor.</param>
        /// <param name="varianceEpsilon">A small float number to avoid dividing by 0.</param>
        /// <param name="scale">A scale Tensor.</param>
        /// <param name="offset">An offset Tensor.</param>
        /// <returns></returns>
        public static Tensor batchNormalization(this Tensor x, Tensor mean, Tensor variance,
            float varianceEpsilon = 0.001f, Tensor scale = null, Tensor offset = null)
        {
            if (mean.Rank != variance.Rank)
            {
                throw new Exception("Batch normalization gradient requires mean and variance to have equal Ranks.");

            }
            if (offset != null)
            {
                if (mean.Rank != offset.Rank)
                {
                    throw new Exception("Batch normalization gradient requires mean and offset to have equal Ranks.");

                }

            }

            if (scale != null)
            {
                if (mean.Rank != scale.Rank)
                {
                    throw new Exception("Batch normalization gradient requires mean and scale to have equal Ranks.");

                }
            }

            Tensor x4D = null;
            if (x.Rank == 0 || x.Rank == 1)
            {
                x4D = x.as4D(1, 1, 1, x.Size);
            }
            else if (x.Rank == 2)
            {
                x4D = x.as4D(1, 1, x.Shape[0], x.Shape[1]);
            }
            else if (x.Rank == 3)
            {
                x4D = x.as4D(1, x.Shape[0], x.Shape[1], x.Shape[2]) as Tensor;
            }
            else
            {
                x4D = x as Tensor;
            }

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                var scaleValue = scale == null ? scalar(1) : scale;
                var reductionAxes = Util.getReductionAxes(mean.Shape, x4D.Shape);
                List<int> tileShape = new List<int>();
                if (mean.Rank == 1)
                {
                    for (var i = 0; i < x4D.Shape.Length - 1; ++i)
                    {
                        tileShape.Add(x4D.Shape[i]);
                    }
                    tileShape.Add(1);
                }

                var xMinusMean = x.sub(mean);
                var dyTimesScaleValue = dy.mul(scaleValue);
                var oneOverSqrtVariance =
                    rsqrt(variance.add(scalar(varianceEpsilon)));
                var minusHalfRCube = oneOverSqrtVariance.mul(oneOverSqrtVariance)
                                           .mul(oneOverSqrtVariance)
                                           .mul(scalar(-0.5f));

                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    if (mean.Rank == 1)
                    {
                        return dy
                            .mul(tile(
                                oneOverSqrtVariance.as4D(1, 1, 1, mean.Shape[0]), tileShape.ToArray()))
                            .mul(scaleValue)
                            .reshape(x.Shape);
                    }
                    else
                    {
                        return dy.mul(oneOverSqrtVariance).mul(scaleValue).reshape(x.Shape);
                    }
                });
                g.gradient.Add("mean", () =>
                {
                    var meanDer =
                oneOverSqrtVariance.mul(scalar(-1)).mul(dyTimesScaleValue);
                    if (mean.Rank == 1)
                    {
                        meanDer = meanDer.sum(reductionAxes);
                    }
                    return meanDer.reshape(mean.Shape);
                });
                g.gradient.Add("variance", () =>
                {
                    var varianceDer = minusHalfRCube.mul(xMinusMean).mul(dyTimesScaleValue);
                    if (mean.Rank == 1)
                    {
                        varianceDer = varianceDer.sum(reductionAxes);
                    }
                    return varianceDer.reshape(mean.Shape);
                });
                g.gradient.Add("scale", () =>
                {
                    var xMinusMean2TimesRsqrt = xMinusMean.mul(oneOverSqrtVariance);
                    var scaleDer = dy.mul(xMinusMean2TimesRsqrt);
                    if (mean.Rank == 1)
                    {
                        scaleDer = scaleDer.sum(reductionAxes);
                    }
                    return scaleDer.reshape(mean.Shape);
                });
                g.gradient.Add("offset", () =>
                {
                    var offsetDer = dy;
                    if (mean.Rank == 1)
                    {
                        offsetDer = offsetDer.sum(reductionAxes);
                    }
                    return offsetDer.reshape(mean.Shape);
                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.batchNormalization(
            x4D, batchnormReshape4D(mean), batchnormReshape4D(variance),
            varianceEpsilon, batchnormReshape4D(scale),
            batchnormReshape4D(offset));
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            inputs.Add("mean", mean);
            inputs.Add("variance", variance);
            inputs.Add("scale", scale);
            inputs.Add("offset", offset);
            var res= e.runKernel(f, inputs, grad) ;
            return res.reshape(x.Shape);
        }
        
        public static Tensor batchnormReshape4D(this Tensor x)
        {
            if (x == null)
            {
                return null;
            }
            if (x.Rank == 0)
            {
                return x.as1D();
            }
            else if (x.Rank == 1)
            {
                return x as Tensor;
            }
            else if (x.Rank == 2)
            {
                return x.as4D(1, 1, x.Shape[0], x.Shape[1]);
            }
            else if (x.Rank == 3)
            {
                return x.as4D(1, x.Shape[0], x.Shape[1], x.Shape[2]);
            }
            return x as Tensor;
        }
    }
}
