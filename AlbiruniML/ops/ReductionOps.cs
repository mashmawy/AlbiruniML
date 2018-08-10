using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public static partial class Ops
    {
 
        /// <summary>
        /// Computes the log(sum(exp(elements across the reduction dimensions)).
        /// Reduces the input along the dimensions given in `axis`. Unless `keepDims`
        /// is true, the rank of the array is reduced by 1 for each entry in `axis`.
        /// If `keepDims` is true, the reduced dimensions are retained with length 1.
        /// If `axis` has no entries, all dimensions are reduced, and an array with a
        /// single element is returned.
        /// </summary> 
        /// <param name="x">The input tensor</param>
        /// <param name="axis">axis The dimension(s) to reduce. If null (the default),
        /// reduces all dimensions.</param>
        /// <param name="keepDims">If true, retains reduced dimensions with length  of 1. Defaults to false.</param>
        /// <returns></returns>
        public static Tensor logSumExp(this Tensor x, int[] axis = null, bool keepDims = false) 
        {
            var axes = Util.parseAxisParam(axis, x.Shape);
            var xMax = x.max(axes, true);
            var a = x.sub(xMax);
            var b = a.exp();
            var c = b.sum(axes);
            var d = c.log();
            var res = xMax.reshape(d.Shape).add(d);
            if (keepDims)
            {
                var newShape = Util.expandShapeToKeepDim(res.Shape, axes);
                return res.reshape(newShape) ;
            }
            return res ;
        }


        /// <summary>
        ///  Computes the sum of elements across dimensions of a `Tensor`.
        ///  Reduces the input along the dimensions given in `axes`. Unless `keepDims`
        ///  is true, the rank of the `Tensor` is reduced by 1 for each entry in `axes`.
        ///  If `keepDims` is true, the reduced dimensions are retained with length 1.
        ///  If axes has no entries, all dimensions are reduced, and a `Tensor` with a
        ///  single element is returned.
        /// </summary> 
        /// <param name="x">The input tensor to compute the sum over. If the dtype is `bool`
        /// it will be converted to `int32` and the output dtype will be `int32`.</param>
        /// <param name="axis">The dimension(s) to reduce. By default it reduces all dimensions.</param>
        /// <param name="keepDims">If true, retains reduced dimensions with size 1.</param>
        /// <returns></returns>
        public static Tensor sum(this Tensor x, int[] axis = null, bool keepDims = false) 
        {
            
            var axes = Util.parseAxisParam(axis, x.Shape);
            var customOp = customGrad(
                (Tensor[] opInputs) =>
                {
                    var xi = opInputs[0];
                    var permutation = Util.getAxesPermutation(axes, xi.Rank);
                    var reductionAxes = axes;
                    var permutedX = xi;
                    if (permutation != null)
                    {
                        permutedX = xi.transpose(permutation);
                        reductionAxes =
                            Util.getInnerMostAxes(reductionAxes.Length, xi.Rank);
                    }
                    ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
                    {

                        return bk.Sum(permutedX, reductionAxes);
                    };

                    var inputs = new Dictionary<string, Tensor>();
                    inputs.Add("x", xi);
                    var value = ENV.engine.runKernel(f, inputs);

                    if (keepDims)
                    {
                        var newShape = Util.expandShapeToKeepDim(value.Shape, axes);
                        value = value.reshape(newShape);
                    }
                    CustomGradientResults res = new CustomGradientResults();
                    res.value = value;
                    res.gradFunc = (Tensor dy) =>
                    {
                        var expandedDyShape =new List<int>( xi.Shape).ToArray();

                        foreach (var axis2 in axes)
                        {
                            expandedDyShape[axis2] = 1;

                        }
                        var expandedDy = dy.reshape(expandedDyShape);
                        var derX = expandedDy.mul(Ops.ones(xi.Shape));
                        return new List<Tensor>() { derX };


                    };
                    return res;
                }
                );
            return customOp(new Tensor[] { x }) ;
        }


        /// <summary>
        /// Computes the mean of elements across dimensions of a `Tensor`.
        /// Reduces `x` along the dimensions given in `axis`. Unless `keepDims` is
        /// true, the rank of the `Tensor` is reduced by 1 for each entry in `axis`.
        /// If `keepDims` is true, the reduced dimensions are retained with length 1.
        /// If `axis` has no entries, all dimensions are reduced, and a `Tensor` with
        /// a single element is returned.
        /// </summary> 
        /// <param name="xtensor">The input tensor.</param>
        /// <param name="axis">The dimension(s) to reduce. By default it reduces all dimensions.</param>
        /// <param name="keepDims">If true, retains reduced dimensions with size 1.</param>
        /// <returns></returns>
        public static Tensor mean(this Tensor xtensor, int[] axis = null, bool keepDims = false) 
        {
            var axes = Util.parseAxisParam(axis, xtensor.Shape);
            var shapes = Util.computeOutAndReduceShapes(xtensor.Shape, axes);
            var reduceShape = shapes.Item2;
            var reduceSize = Util.SizeFromShape(reduceShape);
            var customOp = customGrad(
                (Tensor[] x) =>
                {
                    var reduceSizeScalar = scalar(reduceSize);
                    var xReduce = x[0];
                    var ress = xReduce.div(reduceSizeScalar);

                    var value = ress.sum(axis, keepDims);
                  
                    CustomGradientResults res = new CustomGradientResults();
                    res.value = value;
                    res.gradFunc = (Tensor dy) =>
                    {
                        var expandedDyShape = new List<int>(xReduce.Shape).ToArray();

                        foreach (var axis2 in axes)
                        {
                            expandedDyShape[axis2] = 1;

                        }
                        var expandedDy = dy.reshape(expandedDyShape);
                        var derX = expandedDy.mul(Ops.ones(xReduce.Shape));
                        return new List<Tensor>() { derX };


                    };
                    return res;
                }
                );
            return customOp(new Tensor[] { xtensor }) ;
        }


        /// <summary>
        /// Computes the minimum value from the input.
        /// Reduces the input along the dimensions given in `axes`. Unless `keepDims`
        /// is true, the rank of the array is reduced by 1 for each entry in `axes`.
        /// If `keepDims` is true, the reduced dimensions are retained with length 1.
        /// If `axes` has no entries, all dimensions are reduced, and an array with a
        /// single element is returned.
        /// </summary> 
        /// <param name="x">The input Tensor.</param>
        /// <param name="axis">The dimension(s) to reduce. By default it reduces all dimensions.</param>
        /// <param name="keepDims">If true, retains reduced dimensions with size 1.</param>
        /// <returns></returns>
        public static Tensor min(this Tensor x, int[] axis = null, bool keepDims = false) 
        {

            var origAxes = Util.parseAxisParam(axis, x.Shape);
            var axes = origAxes;
            var permutedAxes = Util.getAxesPermutation(axes, x.Rank);
            if (permutedAxes != null)
            {
                x = x.transpose(permutedAxes);
                axes = Util.getInnerMostAxes(axes.Length, x.Rank);
            }

            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return
                    bk.min(x, axes);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            var res = e.runKernel(f, inputs);
            if (keepDims)
            {
                var newShape = Util.expandShapeToKeepDim(res.Shape, origAxes);
                return res.reshape(newShape) ;
            }
            return res ;
        }
        
        /// <summary>
        /// Computes the maximum of elements across dimensions of a `Tensor`.
        ///
        /// Reduces the input along the dimensions given in `axes`. Unless `keepDims`
        /// is true, the rank of the `Tensor` is reduced by 1 for each entry in `axes`.
        /// If `keepDims` is true, the reduced dimensions are retained with length 1.
        /// If `axes` has no entries, all dimensions are reduced, and an `Tensor` with
        /// a single element is returned.
        /// </summary> 
        /// <param name="x">The input tensor.</param>
        /// <param name="axis">The dimension(s) to reduce. By default it reduces all dimensions.</param>
        /// <param name="keepDims">If true, retains reduced dimensions with size 1.</param>
        /// <returns></returns>
        public static Tensor max(this Tensor x, int[] axis = null, bool keepDims = false) 
        {

            var origAxes = Util.parseAxisParam(axis, x.Shape);
            var axes = origAxes;
            var permutedAxes = Util.getAxesPermutation(axes, x.Rank);
            if (permutedAxes != null)
            {
                x = x.transpose(permutedAxes);
                axes = Util.getInnerMostAxes(axes.Length, x.Rank);
            }

            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return
                    bk.max(x, axes);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            var res = e.runKernel(f, inputs);
            if (keepDims)
            {
                var newShape = Util.expandShapeToKeepDim(res.Shape, origAxes);
                return res.reshape(newShape) ;
            }
            return res ;
        }
       
        /// <summary>
        /// Returns the indices of the minimum values along an `axis`.
        /// 
        /// The result has the same shape as `input` with the dimension along `axis`
        /// removed.
        /// </summary> 
        /// <param name="x">The input tensor.</param>
        /// <param name="axis">The dimension to reduce. Defaults to 0 (outer-most dimension).</param>
        /// <returns></returns>
        public static Tensor argMin(this Tensor x, int[] axis = null) 
        {
            var axes = Util.parseAxisParam(axis, x.Shape);
            var permutedAxes = Util.getAxesPermutation(axes, x.Rank);
            if (permutedAxes != null)
            {
                x = x.transpose(permutedAxes);
                axes = Util.getInnerMostAxes(axes.Length, x.Rank);
            }
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return
                    bk.ArgMin(x, axes);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs) ;
        }
      
        /// <summary>
        /// Returns the indices of the maximum values along an `axis`.
        /// 
        /// The result has the same shape as `input` with the dimension along `axis`
        /// removed.
        /// </summary> 
        /// <param name="x">The input tensor.</param>
        /// <param name="axis">The dimension to reduce. Defaults to 0 (outer-most dimension).</param>
        /// <returns></returns>
        public static Tensor argMax(this Tensor x, int[] axis = null) 
        {
            var axes = Util.parseAxisParam(axis, x.Shape);
            var permutedAxes = Util.getAxesPermutation(axes, x.Rank);
            if (permutedAxes != null)
            {
                x = x.transpose(permutedAxes);
                axes = Util.getInnerMostAxes(axes.Length, x.Rank);
            }
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return
                    bk.ArgMax(x, axes);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs) ;
        }

  
        /// <summary>
        /// Calculates the mean and variance of `x`. The mean and variance are
        /// calculated by aggregating the contents of `x` across `axes`. If `x` is
        /// 1-D and `axes = [0]` this is just the mean and variance of a vector.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="axis">The dimension(s) along with to compute mean and
        /// variance. By default it reduces all dimensions.</param>
        /// <param name="keepDims">If true, the moments have the same dimensionality as the
        /// input.</param>
        /// <returns>An WeakReference with two keys: `mean` and `variance`.</returns>
        public static momentsResult moments(this Tensor x, int[] axis = null, bool keepDims = false)
        {
            var axes = Util.parseAxisParam(axis, x.Shape);
            var mean = x.mean(axes, keepDims);
            var keepDimsShape = mean.Shape;
            if (!keepDims)
            {
                keepDimsShape = Util.expandShapeToKeepDim(mean.Shape, axes);
            }
            var devSquared = x.sub(mean.reshape(keepDimsShape)).square();
            var variance = devSquared.mean(axes, keepDims);
            return new momentsResult() { mean = mean, variance = variance };
        }


        /// <summary>
        /// Computes the logical or of elements across dimensions of a `Tensor`.
        /// Reduces the input along the dimensions given in `axes`. Unless `keepDims`
        /// is true, the rank of the `Tensor` is reduced by 1 for each entry in `axes`.
        /// If `keepDims` is true, the reduced dimensions are retained with length 1.
        /// If `axes` has no entries, all dimensions are reduced, and an `Tensor` with
        /// a single element is returned.
        /// </summary>
        /// <param name="x">The input tensor. Must be of dtype bool.</param>
        /// <param name="axis">The dimension(s) to reduce. By default it reduces
        ///     all dimensions.</param>
        /// <param name="keepDims">If true, retains reduced dimensions with size 1.</param>
        /// <returns></returns>
        public static Tensor any(this Tensor x, int[] axis=null, bool keepDims=false)
        {
            var origAxes = Util.parseAxisParam(axis, x.Shape);
            var axes = origAxes;
            var permutedAxes = Util.getAxesPermutation(axes, x.Rank);
            if (permutedAxes != null)
            {
                x = x.transpose(permutedAxes);
                axes = Util.getInnerMostAxes(axes.Length, x.Rank);
            }
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return
                    bk.any(x, axes);
            };
            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            var res = e.runKernel(f, inputs);
            if (keepDims)
            {
                var newShape = Util.expandShapeToKeepDim(res.Shape, origAxes);
                return res.reshape(newShape);
            }
            return res;
        }



        /// <summary>
        /// Computes the logical and of elements across dimensions of a `Tensor`.
        /// 
        /// Reduces the input along the dimensions given in `axes`. Unless `keepDims`
        /// is true, the rank of the `Tensor` is reduced by 1 for each entry in `axes`.
        /// If `keepDims` is true, the reduced dimensions are retained with length 1.
        /// If `axes` has no entries, all dimensions are reduced, and an `Tensor` with
        /// a single element is returned.
        /// </summary>
        /// <param name="x">The input tensor. Must be of dtype bool.</param>
        /// <param name="axis">The dimension(s) to reduce. By default it reduces
        ///   all dimensions.</param>
        /// <param name="keepDims">If true, retains reduced dimensions with size 1.</param>
        /// <returns></returns>
        public static Tensor all(this Tensor x, int[] axis = null, bool keepDims = false)
        {
            var origAxes = Util.parseAxisParam(axis, x.Shape);
            var axes = origAxes;
            var permutedAxes = Util.getAxesPermutation(axes, x.Rank);
            if (permutedAxes != null)
            {
                x = x.transpose(permutedAxes);
                axes = Util.getInnerMostAxes(axes.Length, x.Rank);
            }
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return
                   bk.all(x, axes);
            };
            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            var res = ENV.engine.runKernel(f, inputs);
            if (keepDims)
            {
                var newShape = Util.expandShapeToKeepDim(res.Shape, origAxes);
                return res.reshape(newShape);
            }
            return res;
        }
    }


    public class momentsResult
    {
        public Tensor mean;
        public Tensor variance;
    }
}
