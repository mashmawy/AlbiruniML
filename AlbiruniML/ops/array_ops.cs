using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public static partial class Ops
    {


        public static float[] ToRowMajor(this float[] data,int[] Shape)
        { 
            float[] res = new float[data.Length];
            var N1 = Shape[0];
            var N2 = Shape[1];
            var N3 = Shape[2];
            var N4 = Shape[3];
            for (int n1 = 0; n1 < Shape[0]; n1++)
            {
                for (int n2 = 0; n2 < Shape[1]; n2++)
                {
                    for (int n3 = 0; n3 < Shape[2]; n3++)
                    {
                        for (int n4 = 0; n4 < Shape[3]; n4++)
                        {

                            var ind1 = n1 + N1
                                   * (n2 + N2 *
                                   (n3 + N3 * (n4)));

                            var ind2 = n4 + N4
                                   * (n3 + N3 *
                                   (n2 + N2 * (n1)));
                            res[ind2] = data[ind1];
                        }
                    }
                }
            }


            return res;
        }
        public static float[] ToColumnMajor(this Tensor x)
        {
            float[] data = x.dataSync();
            float[] res = new float[data.Length];
            var N1 = x.Shape[0];
            var N2 = x.Shape[1];
            var N3 = x.Shape[2];
            var N4 = x.Shape[3];
            for (int n1 = 0; n1 < N1; n1++)
            {
                for (int n2 = 0; n2 < N2; n2++)
                {
                    for (int n3 = 0; n3 < N3; n3++)
                    {
                        for (int n4 = 0; n4 < N4; n4++)
                        {

                            var ind1 = n1 + N1
                                   * (n2 + N2 *
                                   (n3 + N3 * (n4)));
                            var ind2 = n4 + N4
                                   * (n3 + N3 *
                                   (n2 + N2 * (n1)));
                            res[ind1] = data[ind2];
                            
                        }
                    }
                }
            }


            return res;
        }
        public static float[] data(params float[] arr)
        {
            return arr;
        }

        public static Shape shape(params int[] shape)
        {
            return new Shape(shape);
        }

        public static Shape shape(this Tensor t)
        {
            return (Shape)t.Shape;
        }

        public static Tensor flatten(this Tensor t)
        {
            return t.as1D();
        }

        /// <summary> 
        ///Creates a `Tensor` with the provided values, shape. 
        /// </summary>
        /// <param name="values">The values of the tensor</param>
        /// <param name="shape">The shape of the tensor. Optional. If not provided it is inferred from `values`.</param>
        /// <returns></returns>
        public static Tensor tensor(float[] values, int[] shape = null)
        {
            if (shape == null)
            {
                shape = new int[values.Length];
            }
            var res = Tensor.Make(shape, new TensorData(values));
            return res;
        }


        /// <summary>
        /// Creates rank-0 `Tensor` (scalar) with the provided value .
        /// The same functionality can be achieved with `tensor`, but in general
        /// we recommend using `scalar` as it makes the code more readable.
        /// </summary>
        /// <param name="value"> The value of the scalar</param> 
        public static Tensor scalar(float value)
        {

            return tensor(new float[] { value }, new int[] { 1 })
             ;
        }

        /// <summary>
        ///  Creates rank-1 `Tensor` with the provided values, shape.
        ///  The same functionality can be achieved with `tensor`, but in general
        ///  we recommend using `tensor1d` as it makes the code more readable.
        /// </summary>
        /// <param name="value">The values of the tensor. Can be array of numbers or boolean</param>
        /// <returns></returns>
        public static Tensor tensor1d(float[] value)
        {
            return tensor(value, new int[] { value.Length });

        }

        /// <summary>
        /// Creates rank-2 `Tensor` with the provided values, shape .
        /// The same functionality can be achieved with `tensor`, but in general
        /// we recommend using `tensor2d` as it makes the code more readable.
        /// </summary>
        /// <param name="value">The values of the tensor. Can be array of numbers or boolean</param>
        /// <param name="d1">first dimension</param>
        /// <param name="d2">second dimension</param>
        /// <returns></returns>
        public static Tensor tensor2d(float[] value, int d1, int d2)
        {
            return tensor(value, new int[] { d1, d2 })
             ;
        }
        /// <summary>
        /// Creates rank-3 `Tensor` with the provided values, shape.
        /// The same functionality can be achieved with `tensor`, but in general
        /// we recommend using `tensor3d` as it makes the code more readable.
        /// </summary>
        /// <param name="value">The values of the tensor. Can be array of numbers or boolean</param>
        /// <param name="d1">first dimension</param>
        /// <param name="d2">second dimension</param>
        /// <param name="d3">third dimension</param>
        /// <returns></returns>
        public static Tensor tensor3d(float[] value, int d1, int d2, int d3)
        {

            var res = tensor(value, new int[] { d1, d2, d3 });

            return res;
        }

        /// <summary>
        /// Creates rank-4 `Tensor` with the provided values, shape.
        /// The same functionality can be achieved with `tensor`, but in general
        /// we recommend using `tensor4d` as it makes the code more readable.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="d1">first dimension</param>
        /// <param name="d2">second dimension</param>
        /// <param name="d3">third dimension</param>
        /// <param name="d4">fourth dimension</param>
        /// <returns></returns>
        public static Tensor tensor4d(float[] value, int d1, int d2, int d3, int d4)
        {
            return tensor(value, new int[] { d1, d2, d3, d4 })
             ;
        }
        /// <summary>
        /// Creates a `Tensor` with all elements set to 1.
        /// </summary>
        /// <param name="shape">An array of integers defining the output tensor shape.</param>
        /// <returns></returns>
        public static Tensor ones(int[] shape)
        {
            float[] values = null;
            values = new float[Util.SizeFromShape(shape)];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = 1;
            }

            return Tensor.Make(shape, new TensorData(values));
        }
        /// <summary>
        /// Creates a `Tensor` with all elements set to 0.
        /// </summary>
        /// <param name="shape">An array of integers defining the output tensor shape. Can
        ///  be 'float32', 'int32' or 'bool'. Defaults to 'float'.
        /// </param>
        /// <returns></returns>
        public static Tensor zeros(int[] shape)
        {
            float[] values = null;

            values = new float[Util.SizeFromShape(shape)];

            return Tensor.Make(shape, new TensorData(values));
        }

        /// <summary>
        /// Creates a `Tensor` filled with a scalar value.
        /// </summary>
        /// <param name="shape">An array of integers defining the output tensor shape.</param>
        /// <param name="value">The scalar value to fill the tensor with.</param>
        /// <returns></returns>
        public static Tensor fill(int[] shape, float value)
        {
            float[] values = null;
            values = new float[Util.SizeFromShape(shape)];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = value;
            }

            return Tensor.Make(shape, new TensorData(values));
        }


        /// <summary>
        ///  Creates a `Tensor` with all elements set to 1 with the same shape as the
        ///  given tensor.
        /// </summary>
        /// <param name="x"> A tensor.</param>
        /// <returns></returns>
        public static Tensor onesLike(this Tensor x)
        {
            return ones(x.Shape);
        }
        /// <summary>
        ///  Creates a `Tensor` with all elements set to 0 with the same shape as the
        ///  given tensor.
        /// </summary>
        /// <param name="x"> A tensor.</param>
        /// <returns></returns>
        public static Tensor zerosLike(this Tensor x)
        {

            return zeros(x.Shape);
        }

        /// <summary>
        /// Create an identity matrix.
        /// </summary>
        /// <param name="numRows">Number of rows.</param>
        /// <param name="numColumns"> Number of columns. Defaults to `numRows`.</param>
        /// <param name="batchShape"> If provided, will add the batch shape to the beginning
        ///   of the shape of the returned `Tensor` by repeating the identity
        ///   matrix.</param>
        /// <returns>Identity matrix of the specified size and data type, possibly
        /// with batch repetition if `batchShape` is specified.</returns>
        public static Tensor eye(int numRows, Nullable<int> numColumns = null, int[] batchShape = null)
        {
            if (numColumns == null)
            {
                numColumns = numRows;
            }
            var buffer = Ops.buffer(new int[] { numRows, numColumns.Value });
            var n = numRows <= numColumns ? numRows : numColumns;
            for (var i = 0; i < n; ++i)
            {
                buffer.Set(1, i, i);
            }

            var res = buffer.toTensor().as2D(numRows, numColumns.Value);
            if (batchShape == null)
            {
                return res;
            }
            else
            {
                if (batchShape.Length == 1)
                {
                    return Ops.tile(
                        Ops.expandDims(res, 0), new int[] { batchShape[0], 1, 1 });
                }
                else if (batchShape.Length == 2)
                {
                    return Ops.tile(
                        Ops.expandDims(Ops.expandDims(res, 0), 0),
                        new int[] { batchShape[0], batchShape[1], 1, 1 });
                }
                else
                {
                    // TODO(cais): Add support for length-3 once Tensor5D is available.
                    throw new Exception(
                        "eye() currently supports only 1D and 2D ");
                }
            }
        }

        /// <summary>
        ///  Creates a new tensor with the same values and shape as the specified
        ///  tensor.
        /// </summary>
        /// <param name="x">The tensor to clone.</param>
        /// <returns></returns>
        public static Tensor clone(this Tensor x)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () => { return dy; });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return Tensor.Make(x.Shape, new TensorData(x.dataId));
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);

        }


        private static Random randomSeed = new Random();

        /// <summary>
        ///  Creates a `Tensor` with values sampled from a normal distribution.
        /// </summary>
        /// <example>
        /// <code>
        ///  alb.randomNormal(alb.data(2, 2)).print(); 
        /// </code>
        /// </example>
        /// <param name="shape"> An array of integers defining the output tensor shape.</param>
        /// <param name="mean">The mean of the normal distribution.</param>
        /// <param name="stdDev">The standard deviation of the normal distribution.</param>
        /// <param name="seed">The seed for the random number generator.</param>
        /// <returns></returns>
        public static Tensor randomNormal(int[] shape, double mean = 0,
            double stdDev = 1, Nullable<double> seed = null)
        {

            if (seed == null)
            {
                seed = randomSeed.NextDouble();
            }
            var randGauss = new MPRandGauss(mean, stdDev, false, seed);
            var size = Util.SizeFromShape(shape);
            var res = buffer(shape);
            for (int i = 0; i < res.values.Length; i++)
            {

                res.values[i] = (float)randGauss.nextValue();
            }
            return res.toTensor();
        }

        /// <summary>
        ///  Creates a `Tensor` with values sampled from a truncated normal
        ///  distribution.
        ///  The generated values follow a normal distribution with specified mean and
        ///standard deviation, except that values whose magnitude is more than 2
        ///standard deviations from the mean are dropped and re-picked.
        /// </summary>
        /// <example>
        /// <code>
        ///  alb.truncatedNormal(alb.data(2, 2)).print(); 
        /// </code>
        /// </example>
        /// <param name="shape">An array of integers defining the output tensor shape.</param>
        /// <param name="mean">The mean of the normal distribution.</param>
        /// <param name="stdDev">The standard deviation of the normal distribution.</param>
        /// <param name="seed">The seed for the random number generator.</param>
        /// <returns></returns>
        public static Tensor truncatedNormal(int[] shape, double mean = 0,
            double stdDev = 1, Nullable<double> seed = null)
        {
            var randGauss = new MPRandGauss(mean, stdDev, true, seed);
            var size = Util.SizeFromShape(shape);
            var values = new float[size];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (float)randGauss.nextValue();
            }
            return tensor(values, shape);
        }


        /// <summary>
        /// Creates a `Tensor` with values sampled from a uniform distribution.
        /// The generated values follow a uniform distribution in the range [minval,
        ///  maxval). The lower bound minval is included in the range, while the upper
        ///   bound maxval is excluded.
        /// </summary>
        /// <example>
        /// <code>
        ///  alb.randomUniform(alb.data(2, 2)).print(); 
        /// </code>
        /// </example>
        /// <param name="shape">An array of integers defining the output tensor shape.</param>
        /// <param name="minval">The lower bound on the range of random values to generate.
        /// Defaults to 0.</param>
        /// <param name="maxval">The upper bound on the range of random values to generate.
        ///  Defaults to 1.</param>
        /// <returns></returns>
        public static Tensor randomUniform(int[] shape, double minval = 0, double maxval = 1)
        {
            var res = buffer(shape);
            for (var i = 0; i < res.values.Length; i++)
            {
                res.values[i] = (float)Util.randUniform(minval, maxval);
            }
            return res.toTensor();
        }

        /// <summary>
        /// Creates a `Tensor` with values sampled from a random number generator
        /// function defined by the user.
        /// </summary>
        /// <param name="shape">An array of integers defining the output tensor shape.</param>
        /// <param name="randFunction">A random number generator function which is called for
        /// each element in the output tensor.</param>
        /// <returns></returns>
        public static Tensor rand(int[] shape, Func<double> randFunction)
        {
            var size = Util.SizeFromShape(shape);

            var values = new float[size];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (float)randFunction();
            }
            return tensor(values, shape);
        }

        /// <summary>
        /// Creates a `Tensor` with values drawn from a multinomial distribution.
        /// </summary>
        /// <example>
        /// <code>
        ///  var probs = alb.tensor(alb.data(.75f, .25f));
        ///  alb.multinomial(probs, 3).print();
        /// </code>
        /// </example>
        /// <param name="logits">1D array with unnormalized log-probabilities, or
        /// 2D array of shape `[batchSize, numOutcomes]`. See the `normalized`
        ///  parameter.</param>
        /// <param name="normalized"> Whether the provided `logits` are normalized true
        ///  probabilities (sum to 1).</param>
        /// <param name="numSamples">Number of samples to draw for each row slice.</param>
        /// <param name="seed">The seed number.</param>
        /// <returns></returns>
        public static Tensor multinomial(this Tensor logits, bool normalized, int numSamples, Nullable<double>
            seed = null)
        {
            var numOutcomes = logits.Size;
            var origRank = logits.Rank;

            if (numOutcomes < 2)
            {
                throw new Exception(
                    "Error in multinomial: you need at least 2 outcomes, but got " +
                    numOutcomes.ToString());
            }
            if (origRank > 2)
            {
                throw new Exception(
                    "Rank of probabilities must be 1 or 2, but is " + origRank.ToString());
            }
            seed = seed.HasValue ? seed : randomSeed.NextDouble();
            var logits2D = origRank == 1 ? logits.as2D(1, -1) : logits;
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.multinomial(logits2D, normalized, numSamples, seed.Value);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("logits2D", logits2D);
            var res = e.runKernel(f, inputs);
            return res;
        }

        /// <summary>
        /// Creates a one-hot `Tensor`. The locations represented by `indices` take
        /// value `onValue` (defaults to 1), while all other locations take value
        ///  `offValue` (defaults to 0).
        /// </summary>
        /// <param name="indices">`Tensor1D` of indices.</param>
        /// <param name="depth">The depth of the one hot dimension.</param>
        /// <param name="onValue">A number used to fill in output when the index matches
        /// the location.</param>
        /// <param name="offValue">A number used to fill in the output when the index does
        /// not match the location.
        /// </param>
        /// <returns></returns>
        public static Tensor oneHot(this Tensor indices, int depth, float onValue = 1,
            float offValue = 0)
        {

            if (depth < 2)
            {
                throw new Exception("Error in oneHot: depth must be >=2, but it is " + depth.ToString());
            }
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.oneHot(indices, depth, onValue, offValue);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("indices", indices);
            return e.runKernel(f, inputs);

        }

        /// <summary>
        /// Reshapes a `Tensor` to a given shape.
        /// Given a input tensor, returns a new tensor with the same values as the
        /// input tensor with shape `shape`.
        /// If one component of shape is the special value -1, the size of that
        /// dimension is computed so that the total size remains constant. In
        /// particular, a shape of [-1] flattens into 1-D. At most one component of
        /// shape can be -1.
        ///
        /// If shape is 1-D or higher, then the operation returns a tensor with shape
        /// shape filled with the values of tensor. In this case, the number of
        /// elements implied by shape must be the same as the number of elements in
        /// tensor.
        /// </summary>
        /// <param name="x">The input tensor to be reshaped.</param>
        /// <param name="shape">An array of integers defining the output tensor shape.</param>
        /// <returns></returns>
        public static Tensor reshape(this Tensor x, int[] shape)
        {
            var shape2 = Util.InferFromImplicitShape(shape, x.Size);
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () => { return dy.reshape(x.Shape); });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.reshape(x, shape2);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        /// Removes dimensions of size 1 from the shape of a `Tensor`.
        /// </summary>
        /// <param name="x">The input tensor to be squeezed.</param>
        /// <param name="axis">
        /// An optional list of numbers. If specified, only
        /// squeezes the dimensions listed. The dimension index starts at 0. It is
        /// an error to squeeze a dimension that is not 1.
        /// </param>
        /// <returns></returns>
        public static Tensor squeeze(this Tensor x, int[] axis = null)
        {

            return reshape(x, Util.SqueezeShape(x.Shape, axis).NewShape);
        }




        /// <summary>
        ///Construct an tensor by repeating it the number of times given by reps. 
        ///This operation creates a new tensor by replicating `input` `reps`
        ///times. The output tensor's i'th dimension has `input.shape[i] *
        ///reps[i]` elements, and the values of `input` are replicated
        ///`reps[i]` times along the i'th dimension. For example, tiling
        ///`[a, b, c, d]` by `[2]` produces `[a, b, c, d, a, b, c, d]`.
        /// </summary>
        /// <param name="x">The tensor to tile.</param>
        /// <param name="reps">Determines the number of replications per dimension.</param>
        /// <returns></returns>
        public static Tensor tile(this Tensor x, int[] reps)
        {
            if (x.Rank != reps.Length)
            {
                throw new Exception("Error in tile: rank of input " + x.Rank.ToString() + " must match length of reps " + reps.Length);
            }
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {

                    var xGrad = zerosLike(x);
                    if (x.Rank == 1)
                    {
                        for (var i = 0; i < reps[0]; ++i)
                        {
                            xGrad = xGrad.add(dy.slice(new int[] { i * x.Shape[0] },
                                new int[] { x.Shape[0] }));
                        }
                    }
                    else if (x.Rank == 2)
                    {
                        for (var i = 0; i < reps[0]; ++i)
                        {
                            for (var j = 0; j < reps[1]; ++j)
                            {
                                xGrad = xGrad.add(dy.slice(
                                    new int[] { i * x.Shape[0], j * x.Shape[1] },
                                    new int[] { x.Shape[0], x.Shape[1] }));
                            }
                        }
                    }
                    else if (x.Rank == 3)
                    {
                        for (var i = 0; i < reps[0]; ++i)
                        {
                            for (var j = 0; j < reps[1]; ++j)
                            {
                                for (var k = 0; k < reps[2]; ++k)
                                {
                                    xGrad = xGrad.add(dy.slice(
                                        new int[] { i * x.Shape[0], j * x.Shape[1], k * x.Shape[2] },
                                        new int[] { x.Shape[0], x.Shape[1], x.Shape[2] }));
                                }
                            }
                        }
                    }
                    else if (x.Rank == 4)
                    {
                        for (var i = 0; i < reps[0]; ++i)
                        {
                            for (var j = 0; j < reps[1]; ++j)
                            {
                                for (var k = 0; k < reps[2]; ++k)
                                {
                                    for (var l = 0; l < reps[3]; ++l)
                                    {
                                        xGrad = xGrad.add(dy.slice(
                                            new int[] {
                        i * x.Shape[0], j * x.Shape[1], k * x.Shape[2],
                        l * x.Shape[3]
                      },
                                           new int[] { x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3] }));
                                    }
                                }
                            }
                        }
                    }
                    return xGrad;

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) => { return bk.tile(x, reps); };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        /// Computes the sum along segments of a `Tensor`.
        /// </summary>
        /// <param name="x">The `Tensor` that will be summed along its segments</param>
        /// <param name="segmentIds">A `Tensor1D` whose rank is equal to the rank of `x`'s 
        /// dimension along the `axis`.  Maps each element of `x` to a segment.</param>
        /// <param name="numSegments">The number of distinct `segmentIds`</param>
        /// <returns></returns>
        public static Tensor unsortedSegmentSum(this Tensor x, Tensor segmentIds, int numSegments)
        {
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {


                    return gatherDropNegatives(dy, segmentIds);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.unsortedSegmentSum(x, segmentIds, numSegments);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        /// Gather slices from tensor `x`'s axis `axis` according to `indices`.
        /// </summary>
        /// <param name="x">The input tensor whose slices to be gathered.</param>
        /// <param name="indices">The indices of the values to extract.</param>
        /// <param name="axis">The axis over which to select values. Defaults to 0.</param>
        /// <returns></returns>
        public static Tensor gather(this Tensor x, Tensor indices, int axis = 0)
        {

            var axes = Util.parseAxisParam(new int[] { axis }, x.Shape);
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {

                    if (axis == 0)
                    {
                        return unsortedSegmentSum(dy, indices, x.Shape[axis]);
                    }
                    var paramsShape = x.Shape;
                    var indicesSize = indices.Size;

                    var outerShape = paramsShape.Slice(0, axis);
                    var outerDims = outerShape.Length;
                    var innerShape = paramsShape.Slice(axis, paramsShape.Length).Slice(1);
                    var innerDims = innerShape.Length;

                    var outerAxesIndices = arrayRange(0, outerDims);
                    var innerAxesIndices =
                        arrayRange(outerDims + 1, outerDims + 1 + innerDims);

                    var valuesShape =
                        arrayConcat(new int[][] { outerShape, new int[] { indicesSize }, innerShape });

                    var values = dy.reshape(valuesShape);
                    var reshapedIndices = indices.reshape(new int[] { indicesSize });

                    var transposeDims =
                        arrayConcat(new int[][] { new int[] { outerDims }, outerAxesIndices, innerAxesIndices });
                    var valuesTranspose = values.transpose(transposeDims);

                    var paramsGrad = Ops.unsortedSegmentSum(
                        valuesTranspose, reshapedIndices, x.Shape[axis]);

                    var invertTransposeDims = Util.getUndoAxesPermutation(transposeDims);
                    paramsGrad = paramsGrad.transpose(invertTransposeDims);

                    return paramsGrad;

                });
                return g;
            };
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            { return bk.gather(x, indices, axis); };

            Engine e = ENV.engine;
            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            inputs.Add("indices", indices);
            return e.runKernel(f, inputs, grad);
        }
        public static int[] arrayConcat(int[][] arrays)
        {
            List<int> result = new List<int>();
            for (var i = 0; i < arrays.Length; ++i)
            {
                for (var j = 0; j < arrays[i].Length; ++j)
                {
                    result.Add(arrays[i][j]);
                }
            }
            return result.ToArray();
        }
        public static int[] arrayRange(int start, int stop)
        {
            List<int> result = new List<int>();
            for (var i = start; i < stop; ++i)
            {
                result.Add(i);
            }
            return result.ToArray();
        }
        public static Tensor gatherDropNegatives(this Tensor x, Tensor indices)
        {
            // Helper function for unsorted segment ops. Gathers params for
            // positive segment ids and gathers 0 for inputs with negative segment id.
            // Mirrors _GatherDropNegatives from tensorflow/python/ops/math_grad.py
            var zeroClippedIndices =
               maximum(indices, zerosLike(indices));
            var gathered = gather(x, zeroClippedIndices);
            var isPositive = greaterEqual(indices, scalar(0));
            var numIters = gathered.Rank - isPositive.Rank;
            for (var i = 0; i < numIters; ++i)
            {
                isPositive = expandDims(isPositive, i + 1);
            }
            isPositive =
                 logicalAnd(isPositive, ones(gathered.Shape));
            var zeroSlice = zerosLike(gathered);
            return where(isPositive, gathered, zeroSlice);
        }

        /// <summary>
        /// Pads a `Tensor1D` with a given value and paddings. See `pad` for details.
        /// </summary>
        public static Tensor pad1d(this Tensor x, int[][] paddings, float constantValue = 0f)
        {
            if (paddings.Length != 2)
            {
                throw new Exception("Invalid number of paddings. Must be length of 2.");
            }
            return pad(x, paddings, constantValue);
        }

        /// <summary>
        /// Pads a `Tensor2D` with a given value and paddings. See `pad` for details.
        /// </summary> 
        public static Tensor pad2d(this Tensor x, int[][] paddings, float constantValue = 0f)
        {
            if (!(paddings.Length == 2 && paddings[0].Length == 2 &&
            paddings[1].Length == 2))
            {
                throw new Exception("Invalid number of paddings. Must be length of 2 each.");
            }
            return pad(x, paddings, constantValue);
        }

        /// <summary>
        /// Pads a `Tensor3D` with a given value and paddings. See `pad` for details.
        /// </summary> 
        public static Tensor pad3d(this Tensor x, int[][] paddings, float constantValue = 0f)
        {
            if (!(paddings.Length == 3 && paddings[0].Length == 2 &&
            paddings[1].Length == 2 && paddings[2].Length == 2))
            {
                throw new Exception("Invalid number of paddings. Must be length of 2 each.");
            }
            return pad(x, paddings, constantValue);
        }

        /// <summary>
        /// Pads a `Tensor4D` with a given value and paddings. See `pad` for details.
        /// </summary> 
        public static Tensor pad4d(this Tensor x, int[][] paddings, float constantValue = 0f)
        {
            if (!(paddings.Length == 4 && paddings[0].Length == 2 &&
            paddings[1].Length == 2 && paddings[2].Length == 2 &&
            paddings[3].Length == 2))
            {
                throw new Exception("Invalid number of paddings. Must be length of 2 each.");
            }
            return pad(x, paddings, constantValue);
        }

        /// <summary>
        /// Pads a `Tensor` with a given value and paddings.
        /// This operation currently only implements the `CONSTANT` mode.
        /// </summary>
        /// <param name="x">The tensor to pad.</param>
        /// <param name="paddings">
        ///  An array of length `R` (the rank of the tensor), where each
        ///element is a length-2 tuple of ints `[padBefore, padAfter]`, specifying
        ///how much to pad along each dimension of the tensor.
        /// </param>
        /// <param name="constantValue">The pad value to use. Defaults to 0.</param>
        /// <returns></returns>
        public static Tensor pad(this Tensor x, int[][] paddings, float constantValue = 0f)
        {
            if (x.Rank == 0)
            {
                throw new Exception("pad(scalar) is not defined. Pass non-scalar to pad");
            }
            // Pad introduces values around the original tensor, so the gradient slices
            // the original shape out of the gradient.
            var begin = paddings.Select(p => p[0]).ToArray();
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () => { return dy.slice(begin, x.Shape); });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.pad(x, paddings, constantValue);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad);
        }

        /// <summary>
        /// Unstacks a `Tensor` of rank-`R` into a list of rank-`(R-1)` `Tensor`s.
        /// </summary>
        /// <param name="x">A tensor object.</param>
        /// <param name="axis">The axis to unstack along. Defaults to 0 (the first dim).</param>
        /// <returns></returns>
        public static Tensor[] unstack(this Tensor x, int axis=0)
        {
            var num = x.Shape[axis];
            var outputShape = new int[x.Rank - 1];
            var outIndex = 0;
            for (var i = 0; i < x.Rank; i++)
            {
                if (i != axis)
                {
                    outputShape[outIndex] = x.Shape[i];
                    outIndex++;
                }
            }
              var splitSizes= new int[num];
              splitSizes.Fill(1);
              var begin = new int[x.Rank];
              var size = x.Shape.Slice();
              return splitSizes.Select(s =>
              {
                  size[axis] = s;
                  var slice = x.slice(begin, size);
                  begin[axis] += s;
                  return slice.reshape(outputShape);
              }).ToArray();
        }

        /// <summary>
        /// Stacks a list of rank-`R` `Tensor`s into one rank-`(R+1)` `Tensor`.
        /// </summary>
        /// <param name="tensors"> A list of tensor objects with the same shape.</param>
        /// <param name="axis">The axis to stack along. Defaults to 0 (the first dim).</param>
        /// <returns></returns>
        public static Tensor stack(this Tensor[] tensors, int axis = 0)
        {
            var rank = tensors[0].Rank;
            var shape = tensors[0].Shape;
            var expandedTensors = tensors.Select(t => t.expandDims(axis)).ToArray();
            return concat(expandedTensors, axis);
        }

        /// <summary>
        ///Splits a `Tensor` into sub tensors.
        ///
        /// If `numOrSizeSplits` is a number, splits `x` along dimension `axis`
        /// into `numOrSizeSplits` smaller tensors.
        /// Requires that `numOrSizeSplits` evenly divides `x.shape[axis]`.
        ///
        /// If `numOrSizeSplits` is a number array, splits `x` into
        /// `(numOrSizeSplits.length` pieces. The shape of the `i`-th piece has the
        /// same size as `x` except along dimension `axis` where the size is
        /// `numOrSizeSplits[i]`.
        /// </summary>
        /// <param name="x">The input tensor to split.</param>
        /// <param name="numOrSizeSplits">
        /// Either an integer indicating the number of
        /// splits along the axis or an array of integers containing the sizes of each
        /// output tensor along the axis. If a number then it must evenly divide
        /// `x.shape[axis]`; otherwise the sum of sizes must match `x.shape[axis]`.
        /// </param>
        /// <param name="axis"> The dimension along which to split. Defaults to 0 (the first dim).</param>
        /// <returns></returns>
        public static Tensor[] split(this Tensor x, int[] numOrSizeSplits, int axis = 0)
        {
            axis = Util.parseAxisParam(new int[] { axis }, x.Shape)[0];

            int[] splitSizes = null;

            if (numOrSizeSplits.Length == 1)
            {
                splitSizes = new int[numOrSizeSplits[0]];
                for (int i = 0; i < splitSizes.Length; i++)
                {
                    splitSizes[i] = x.Shape[axis] / numOrSizeSplits[0];
                }
            }
            else
            {
                splitSizes = numOrSizeSplits;

            }


            var begin = new int[x.Rank];
            var size = x.Shape.Slice(0);
            return splitSizes.Select(s =>
            {
                size[axis] = s;
                var slice = x.slice(begin, size);
                begin[axis] += s;
                return slice;
            }).ToArray();
        }


        /// <summary>
        ///Splits a `Tensor` into sub tensors.
        ///
        /// If `numOrSizeSplits` is a number, splits `x` along dimension `axis`
        /// into `numOrSizeSplits` smaller tensors.
        /// Requires that `numOrSizeSplits` evenly divides `x.shape[axis]`.
        ///
        /// If `numOrSizeSplits` is a number array, splits `x` into
        /// `(numOrSizeSplits.length` pieces. The shape of the `i`-th piece has the
        /// same size as `x` except along dimension `axis` where the size is
        /// `numOrSizeSplits[i]`.
        /// </summary>
        /// <param name="x">The input tensor to split.</param>
        /// <param name="numOrSizeSplits">
        /// Either an integer indicating the number of
        /// splits along the axis or an array of integers containing the sizes of each
        /// output tensor along the axis. If a number then it must evenly divide
        /// `x.shape[axis]`; otherwise the sum of sizes must match `x.shape[axis]`.
        /// </param>
        /// <param name="axis"> The dimension along which to split. Defaults to 0 (the first dim).</param>
        /// <returns></returns>
        public static Tensor[] split(this Tensor x, int numOrSizeSplits, int axis = 0)
        {
            axis = Util.parseAxisParam(new int[] { axis }, x.Shape)[0];


            var splitSizes = new int[numOrSizeSplits]; ;
            for (int i = 0; i < splitSizes.Length; i++)
            {
                splitSizes[i] = x.Shape[axis] / numOrSizeSplits;
            }
            var begin = new int[x.Rank];
            var size = x.Shape.ToArray();
            return splitSizes.Select(s =>
            {
                size[axis] = s;
                var slice = x.slice(begin, size);
                begin[axis] += s;
                return slice;
            }).ToArray();
        }


        /// <summary>
        /// Computes the cumulative sum of a `Tensor` along `axis`.
        /// </summary>
        /// <param name="x">The input tensor to be summed.</param>
        /// <param name="axis">The axis along which to sum. Optional. Defaults to 0.</param>
        /// <param name="exclusive">Whether to perform exclusive cumulative sum. Optional.
        ///   Defaults to false. If set to true then the sum of each tensor entry
        ///   does not include its own value, but only the values previous to it
        ///     along the specified axis.</param>
        /// <param name="reverse">Whether to sum in the opposite direction. Optional.
        ///   Defaults to false.</param>
        /// <returns></returns>
        public static Tensor cumsum(this Tensor x, int axis = 0, bool exclusive = false,
            bool reverse = false)
        {

            axis = axis | 0;
            var permutation = Util.getAxesPermutation(new int[] { axis }, x.Rank);
            var permutedX = x;
            if (permutation != null)
            {
                permutedX = x.transpose(permutation);
            }
            var permutedAxis = Util.getInnerMostAxes(1, x.Rank)[0];


            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("permutedX", () => { return dy.cumsum(axis, exclusive, !reverse); });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.cumsum(
permutedX, permutedAxis, exclusive, reverse);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("permutedX", permutedX);
            var value = e.runKernel(f, inputs, grad);
            if (permutation != null)
            {
                value = value.transpose(permutation);
            }
            return value;

        }
        /// <summary>
        /// Returns a `Tensor` that has expanded rank, by inserting a dimension
        /// into the tensor's shape.
        /// </summary>
        /// <param name="x">The input tensor whose dimensions to be expanded.</param>
        /// <param name="axis">The dimension index at which to insert shape of `1`. Defaults
        /// to 0 (the first dimension).</param>
        /// <returns></returns>
        public static Tensor expandDims(this Tensor x, int axis)
        {
            if (!(axis <= x.Rank))
            {
                throw new Exception("Axis must be <= rank of the tensor");
            }
            if (axis < 0)
            {
                // Negative value is counted from the tail of rank.

                axis = x.Rank + axis + 1;
            }
            List<int> i2 = new List<int>(x.Shape);
            i2.Insert(axis, 1);

            return reshape(x, i2.ToArray());
        }


        /// <summary>
        /// Return an evenly spaced sequence of numbers over the given interval.
        /// </summary>
        /// <param name="start">The start value of the sequence.</param>
        /// <param name="stop">The end value of the sequence.</param>
        /// <param name="num">The number of values to generate.</param>
        /// <returns></returns>
        public static Tensor linspace(int start, int stop, int num)
        {
            if (num == 0)
            {
                throw new Exception("Cannot request zero samples");
            }
            var step = (stop - start) / (num - 1);
            var values = new float[num];
            values[0] = start;
            for (var i = 1; i < values.Length; i++)
            {
                values[i] = values[i - 1] + step;
            }

            return tensor1d(values);
        }

        /// <summary>
        ///   Creates a new `Tensor1D` filled with the numbers in the range provided.
        ///
        ///The tensor is a is half-open interval meaning it includes start, but
        ///excludes stop. Decrementing ranges and negative step values are also
        ///supported.
        /// </summary>
        /// <param name="start">An integer start value</param>
        /// <param name="stop"> An integer stop value</param>
        /// <param name="step">An integer increment (will default to 1 or -1)</param>
        /// <returns></returns>
        public static Tensor range(int start, int stop, int step = 1)
        {
            if (step == 0)
            {
                throw new Exception("Cannot have a step of zero");
            }
            var sameStartStop = start == stop;
            var increasingRangeNegativeStep = start < stop && step < 0;
            var decreasingRangePositiveStep = stop < start && step > 1;
            if (sameStartStop || increasingRangeNegativeStep ||
       decreasingRangePositiveStep)
            {
                return zeros(new int[] { 0 });
            }

            var numElements = (int)Math.Abs(Math.Ceiling((stop - start) / (float)step));

            if (stop < start && step == 1)
            {
                // Auto adjust the step's sign if it hasn't been set
                // (or was set to 1)
                step = -1;
            }
            var values = new float[numElements];// Util.makeZerosTypedArray(numElements);
            values[0] = start;
            for (int i = 1; i < values.Length; i++)
            {
                values[i] = values[i - 1] + step;
            }

            return tensor1d(values);

        }

        /// <summary>
        ///Creates an empty `TensorBuffer` with the specified `shape`. 
        ///The values are stored in cpu as `TypedArray`. Fill the buffer using
        ///`buffer.set()`, or by modifying directly `buffer.values`. When done,
        ///call `buffer.toTensor()` to get an immutable `Tensor` with those values.
        ///
        ///When done, call `buffer.toTensor()` to get an immutable `Tensor` with
        ///those values.
        /// </summary>
        /// <param name="shape"> An array of integers defining the output tensor shape.</param>
        /// <param name="values">The values of the buffer</param>
        /// <returns></returns>
        public static TensorBuffer buffer(int[] shape
            , float[] values = null)
        {
            return new TensorBuffer(shape, values);
        }


        /// <summary>
        /// This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of
        /// shape `blockShape + [batch]`, interleaves these blocks back into the grid
        /// defined by the spatial dimensions `[1, ..., M]`, to obtain a result with
        /// the same rank as the input. The spatial dimensions of this intermediate
        /// result are then optionally cropped according to `crops` to produce the
        /// output. This is the reverse of `spaceToBatchND`. See below for a precise
        /// description.
        /// 
        /// 
        /// This operation is equivalent to the following steps:
        ///
        ///1. Reshape `x` to `reshaped` of shape: `[blockShape[0], ...,
        ///blockShape[M-1], batch / prod(blockShape), x.shape[1], ...,
        ///x.shape[N-1]]`
        ///
        ///2. Permute dimensions of `reshaped`to produce `permuted` of shape `[batch /
        ///prod(blockShape),x.shape[1], blockShape[0], ..., x.shape[M],
        ///blockShape[M-1],x.shape[M+1], ..., x.shape[N-1]]`
        ///
        ///3. Reshape `permuted` to produce `reshapedPermuted` of shape `[batch /
        ///prod(blockShape),x.shape[1] * blockShape[0], ..., x.shape[M] *
        ///blockShape[M-1],x.shape[M+1], ..., x.shape[N-1]]`
        ///
        ///4. Crop the start and end of dimensions `[1, ..., M]` of `reshapedPermuted`
        ///according to `crops` to produce the output of shape: `[batch /
        ///prod(blockShape),x.shape[1] * blockShape[0] - crops[0,0] - crops[0,1],
        ///..., x.shape[M] * blockShape[M-1] - crops[M-1,0] -
        ///crops[M-1,1],x.shape[M+1], ..., x.shape[N-1]]`
        /// </summary>
        /// <param name="x">A `Tensor`. N-D with `x.shape` = `[batch] + spatialShape +
        /// remainingShape`, where spatialShape has `M` dimensions.</param>
        /// <param name="blockShape">A 1-D array. Must be one of the following types: `int32`,
        /// `int64`. Must have shape `[M]`, all values must be >= 1.</param>
        /// <param name="crops">A 2-D array.  Must be one of the following types: `int32`,
        /// `int64`. Must have shape `[M, 2]`, all values must be >= 0. `crops[i] =
        /// [cropStart, cropEnd]` specifies the amount to crop from input dimension `i
        /// + 1`, which corresponds to spatial dimension `i`. It is required that
        /// `cropStart[i] + cropEnd[i] less than or = blockShape[i] * inputShape[i + 1]`</param>
        /// <returns></returns>
        public static Tensor batchToSpaceND(this Tensor x, int[] blockShape, int[][] crops)
        {
            var prod = blockShape.Aggregate((a, b) => a * b);

            Util.assert(
      x.Rank >= 1 + blockShape.Length,
     "input rank should be > than [blockShape]");

            Util.assert(
                crops.Length == blockShape.Length,
                "crops.shape[0] must be equal to [blockShape]");

            Util.assert(
                x.Shape[0] % prod == 0,
          "input tensor batch must be divisible by prod( blockShape )");
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.batchToSpaceND(x, blockShape, crops);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs);
        }


         
    }
}
