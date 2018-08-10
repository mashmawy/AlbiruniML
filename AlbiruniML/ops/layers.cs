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
        /// Repeats a 2D tensor.
        /// If `x` has shape `[samples, dim]` and `n` is 2, for example, the output
        /// will have shape `[samples, 2, dim]`.
        /// </summary>
        /// <param name="x">Input tensor.</param>
        /// <param name="n">Integer, number of times to repeat.</param>
        /// <returns>The result of the repeat operation.</returns>
        public static Tensor repeat(this Tensor x, int n)
        {
            if (x.Shape.Length != 2)
            {
                throw new Exception(
                    "repeat() expects a Rank-2 tensor, but received a " +
                    "Rank-" + x.Shape.Length + " tensor.");
            }
            var y = expandDims(x, 1);
            return tile(y, shape(1, n, 1));
        }

        /// <summary>
        /// Turn a nD tensor into a 2D tensor with same 0th dimension.
        /// In other words, it flattens each data samples of a batch
        /// </summary>
        /// <param name="x">The tensor to flatten. The Rank of this tensor is required to be 2 or higher.</param>
        /// <returns>The result of the flattening.</returns>
        public static Tensor batchFlatten(this Tensor x)
        {
            if (x.Shape.Length <= 1)
            {
                throw new Exception(
                    "batchFlatten requires a minimum Rank of 2. Got Rank: " + x.Rank + ".");
            }
            var newShape = shape(x.Shape[0], Util.arrayProd(x.Shape, 1));
            return x.reshape(newShape);
        }


        /// <summary>
        ///  Do slicing along the first axis.
        /// </summary>
        /// <param name="array">input `Tensor`.</param>
        /// <param name="start">starting index, inclusive.</param>
        /// <param name="size">size of the slice along the first axis.</param>
        /// <returns>result of the slicing.</returns>
        public static Tensor sliceAlongFirstAxis(this Tensor array, int start, int size)
        {
            switch (array.Rank)
            {
                case 1:
                    return slice1d(array, start, size);
                case 2:
                    return slice2d(array, new int[] { start, 0 }, new int[] { size, array.Shape[1] });
                case 3:
                    return slice3d(
                        array, new int[] { start, 0, 0 },
                        new int[] { size, array.Shape[1], array.Shape[2] });
                case 4:
                    return slice4d(
                        array, new int[] { start, 0, 0, 0 },
                        new int[] { size, array.Shape[1], array.Shape[2], array.Shape[3] });
                default:
                    throw new Exception(
                        "sliceAlongFirstAxis() received an unsupported tensor Rank: " +
                        array.Rank.ToString());
            }
        }


        /// <summary>
        ///  Do slicing along the last axis.
        /// </summary>
        /// <param name="array">input `Tensor`.</param>
        /// <param name="start">starting index, inclusive.</param>
        /// <param name="size">size of the slice along the last axis.</param>
        /// <returns>result of the slicing.</returns>
        public static Tensor sliceAlongLastAxis(this Tensor array, int start, int size)
        {
            switch (array.Rank)
            {
                case 1:
                    return slice1d(array, start, size);
                case 2:
                    return slice2d(array, new int[] { 0, start }, new int[] { array.Shape[0], size });
                case 3:
                    return slice3d(
                        array, new int[] { 0, 0, start },
                        new int[] { array.Shape[0], array.Shape[1], size });
                case 4:
                    return slice4d(
                        array, new int[] { 0, 0, 0, start },
                        new int[] { array.Shape[0], array.Shape[1], array.Shape[2], size });
                default:
                    throw new Exception(
                        "sliceAlongLastAxis() received an unsupported tensor Rank:" +
                        "${array.Rank}");
            }
        }


        /// <summary>
        /// Do slicing along the sepcified axis.
        /// </summary>
        /// <param name="array">input `Tensor`.</param>
        /// <param name="start">starting index, inclusive.</param>
        /// <param name="size">of the slice along the chosen axis.</param>
        /// <param name="axis">choose an axis</param>
        /// <returns>result of the slicing.</returns>
        public static Tensor sliceAlongLastAxis(this Tensor array, int start, int size, int axis)
        {
            switch (array.Rank)
            {
                case 1:
                    return slice1d(array, start, size);
                case 2:
                    switch (axis)
                    {
                        case 1:
                            return sliceAlongFirstAxis(array, start, size);
                        case 2:
                            return sliceAlongLastAxis(array, start, size);
                        default:
                            throw new Exception(
                               "The axis is not within the Rank of the tensor " +
                                axis.ToString());
                    }
                case 3:
                    switch (axis)
                    {
                        case 1:
                            return sliceAlongFirstAxis(array, start, size);
                        case 2:
                            return slice3d(
                                array, new int[] { 0, start, 0 },
                                new int[] { array.Shape[0], size, array.Shape[2] });
                        case 3:
                            return sliceAlongLastAxis(array, start, size);
                        default:
                            throw new Exception(
                                "The axis is not within the Rank of the tensor " +
                                 axis.ToString());
                    }
                case 4:
                    switch (axis)
                    {
                        case 1:
                            return sliceAlongFirstAxis(array, start, size);
                        case 2:
                            return slice4d(
                                array, new int[] { 0, start, 0, 0 },
                               new int[] { array.Shape[0], size, array.Shape[2], array.Shape[3] });
                        case 3:
                            return slice4d(
                                array, new int[] { 0, 0, start, 0 },
                                new int[] { array.Shape[0], array.Shape[1], size, array.Shape[3] });
                        case 4:
                            return sliceAlongLastAxis(array, start, size);
                        default:
                            throw new Exception(
                                "The axis is not within the Rank of the tensor " +
                                axis.ToString());
                    }
                default:
                    throw new Exception(
                        "sliceAlongLastAxis() received an unsupported tensor Rank: " +
                         array.Rank.ToString());
            }
        }

        /// <summary>
        ///  Concatenates a list of tensors alongside the specified axis.
        /// </summary>
        /// <param name="tensors">`Array` of tensors to concatenate.</param>
        /// <param name="axis"> Concatenation axis.</param>
        /// <returns>The result of the concatenation.</returns>
        public static Tensor concatenate(Tensor[] tensors, int axis = -1)
        {
            var Rank = 0;
            if (axis < 0)
            {
                Rank = tensors[0].Shape.Length;
                if (Rank != 0)
                {
                    axis = Rank;
                }
                else
                {
                    axis = 0;
                }
            }
            if (axis == tensors[0].Shape.Length)
            {
                // Porting Note: This is necessary because concat() requires axis to be
                //   in the interval [-Rank, Rank).
                axis = -1;
            }
            // Porting Note: Sparse concat is not supported yet.
            return concat(tensors, axis);
        }

        /// <summary>
        /// Concatenate two arrays along the first dimension.
        /// </summary>
        /// <param name="a">The 1st `Tensor` to concatenate.</param>
        /// <param name="b">The 2nd `Tensor` to concatenate.</param>
        /// <returns>Result of the concatenation.</returns>
        public static Tensor concatAlongFirstAxis(this Tensor a, Tensor b)
        {
            switch (a.Shape.Length)
            {
                case 1:
                    return concat1d(new Tensor[] { a, b });
                case 2:
                    return concat2d(new Tensor[] { a, b }, 0);
                case 3:
                    return concat3d(new Tensor[] { a, b }, 0);
                case 4:
                    return concat4d(new Tensor[] { a, b }, 0);
                default:
                    throw new Exception(
          "concatAlongFirstAxis() received an unsupported tensor Rank: " +
          a.Rank.ToString());
}
        }

      
        /// <summary>
        /// Add a bias to a tensor.
        /// </summary>
        /// <param name="x">The tensor to add the bias to.</param>
        /// <param name="bias">The bias to add to `x`. Must be 1D or the same rank as `x`.</param>
        /// <param name="dataFormat"></param>
        /// <returns> Result of the bias adding.</returns>
        public static Tensor biasAdd(this Tensor x, Tensor bias, ConvDataFormat dataFormat = ConvDataFormat.channelsLast)
        {
            if (bias.Rank != 1 && bias.Rank != x.Rank)
            {
                throw new Exception(
                    "Unexpected bias dimensions: " + bias.Rank +
                    "; expected it to be 1 or " + x.Rank);
            }

            var biasShape = bias.Shape;
            Tensor y = null ;
            if (x.Rank == 5)
            {
                if (dataFormat == ConvDataFormat.channelsFirst)
                {
                    if (biasShape.Length == 1)
                    {
                        y = x.add(bias.reshape(shape(1, biasShape[0], 1, 1, 1)));
                    }
                    else
                    {
                        y = x.add(bias.reshape(
                            shape(1, biasShape[3], biasShape[0], biasShape[1], biasShape[2])));
                    }
                }
                else if (dataFormat == ConvDataFormat.channelsLast)
                {
                    if (biasShape.Length == 1)
                    {
                        y = x.add(bias.reshape(shape(1, 1, 1, 1, biasShape[0])));
                    }
                    else
                    {
                        var nb = new List<int>();
                        nb.Add(1);
                        nb.AddRange(biasShape);
                        y = x.add(bias.reshape(nb.ToArray()));
                    }
                }
            }
            else if (x.Rank == 4)
            {
                if (dataFormat == ConvDataFormat.channelsFirst)
                {
                    if (biasShape.Length == 1)
                    {
                        y = x.add(bias.reshape(shape(1, biasShape[0], 1, 1)));
                    }
                    else
                    {
                        y = x.add(
                            bias.reshape(shape(1, biasShape[2], biasShape[0], biasShape[1])));
                    }
                }
                else if (dataFormat == ConvDataFormat.channelsLast)
                {
                    if (biasShape.Length == 1)
                    {
                        y = x.add(bias.reshape(shape(1, 1, 1, biasShape[0])));
                    }
                    else
                    {
                        var nb = new List<int>();
                        nb.Add(1);
                        nb.AddRange(biasShape);
                        y = x.add(bias.reshape(nb.ToArray()));
                    }
                }
            }
            else if (x.Rank == 3)
            {
                if (dataFormat == ConvDataFormat.channelsFirst)
                {
                    if (biasShape.Length == 1)
                    {
                        y = x.add(bias.reshape(shape(1, biasShape[0], 1)));
                    }
                    else
                    {
                        y = x.add(bias.reshape(shape(1, biasShape[1], biasShape[0])));
                    }
                }
                else if (dataFormat == ConvDataFormat.channelsLast)
                {
                    if (biasShape.Length == 1)
                    {
                        y = x.add(bias.reshape(shape(1, 1, biasShape[0])));
                    }
                    else
                    {
                        var nb = new List<int>();
                        nb.Add(1);
                        nb.AddRange(biasShape);
                        y = x.add(bias.reshape(nb.ToArray()));
                    }
                }
            }
            else if (x.Rank < 3)
            {
                y = x.add(bias);
            }
            else
            {
                throw new Exception("Unsupported input Rank by biasAdd: " + x.Rank.ToString());
            }
            return y;
        }

        /// <summary>
        /// Softsign of a tensor.
        /// Defined as x / (abs(x) + 1), element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static Tensor softsign(this Tensor x)
        {
            return  div(x,  add(scalar(1), abs(x)));

        }


        /// <summary>
        /// Sets entries in `x` to zero at random, while scaling the entire tensor.
        /// </summary>
        /// <param name="x">input tensor.</param>
        /// <param name="level">fraction of the entries in the tensor that will be set to 0.</param>
        /// <param name="noiseShape">shape of randomly generated keep/drop flags, must be broadcastable to the shape of `x`.</param>
        /// <param name="seed">random seed to ensure determinism.</param>
        /// <returns>Result of the dropout operation.</returns>
        public static Tensor dropout(this Tensor x, Tensor level, int[] noiseShape = null, Nullable<int> seed=null)
        {
            if (noiseShape != null && !Util.ArrayIsEqual(x.Shape, noiseShape))
            {
                throw new Exception(
                    "Non-default noise shape is not implemented yet  ");
            }
            if (seed != null)
            {
                throw new Exception("seed is not implemented for dropout yet.");
            }
            var multiplier = step(add(
                neg(level), randomUniform(x.Shape, 0, 1)));
            // Scale the kept elements, so the expected sum is unchanged.
            multiplier = mul(
                div(scalar(1), sub(scalar(1), level)),
                multiplier);
            return mul(x, multiplier);
        }

        /// <summary>
        /// Element-wise, segment-wise linear approximation of sigmoid.
        ///  Returns `0.` if `x less than -2.5`, `1.` if `x > 2.5`.
        ///  In `-2.5 less than = x less than = 2.5`, returns `0.2 * x + 0.5`.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static Tensor hardSigmoid(this Tensor x)
        {
            var y = add(scalar(0.5f), mul(scalar(0.2f), x));
            return clipByValue(y, 0, 1);
        }
    
         
    }
}
