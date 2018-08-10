using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{

    public static class Util
    {

        public static readonly double ERF_P = 0.3275911;
        public static readonly double ERF_A1 = 0.254829592;
        public static readonly double ERF_A2 = -0.284496736;
        public static readonly double ERF_A3 = 1.421413741;
        public static readonly double ERF_A4 = -1.453152027;
        public static readonly double ERF_A5 = 1.061405429;


        /// <summary>
        /// Gets the new shape of the input Tensor after it's been reshaped
        /// to :
        /// [blockShape[0], ..., blockShape[M-1], batch / prod(blockShape),
        /// inputShape[1], ..., inputShape[N-1]]
        /// 
        /// See step 1: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
        /// </summary>
        /// <param name="inputShape"></param>
        /// <param name="blockShape"></param>
        /// <param name="prod"></param>
        /// <returns></returns>
        public static int[] getReshaped(int[] inputShape, int[] blockShape, int prod)
        {
            var reshaped = blockShape.Slice(0).ToList();
            reshaped.Add(inputShape[0] / prod);
             reshaped.AddRange(inputShape.Slice(1));
            return reshaped.ToArray();
        }

        /// <summary>
        /// Gets the permutation that will transpose the dimensions of the
        /// reshaped tensor to shape:
        ///
        /// [batch / prod(block_shape),inputShape[1], blockShape[0], ...,
        /// inputShape[M], blockShape[M-1],inputShape[M+1], ..., inputShape[N-1]]
        ///
        /// see step 2: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
        /// </summary>
        /// <param name="reshapedRank"></param>
        /// <param name="blockShapeRank"></param>
        /// <returns></returns>
        public static int[] getPermuted(int reshapedRank, int blockShapeRank)
        {
            var permuted = new List<int>() { blockShapeRank };
            for (var i = blockShapeRank + 1; i < reshapedRank; ++i)
            {
                if (i <= 2 * blockShapeRank)
                {
                    permuted.Add(i);
                    permuted.Add(i - (blockShapeRank + 1));
                }
                else
                {
                    permuted.Add(i);
                }
            }
            return permuted.ToArray();
        }
        /// <summary>
        ///  Gets the shape of the reshaped and permuted input Tensor before any cropping
        /// is applied.  The new shape will be:
        /// 
        /// [batch / prod(blockShape),inputShape[1] * blockShape[0], ...,
        /// inputShape[M] * blockShape[M-1],inputShape[M+1], ..., inputShape[N-1]]
        /// 
        /// See step 3: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
        /// </summary>
        /// <param name="inputShape"></param>
        /// <param name="blockShape"></param>
        /// <param name="prod"></param>
        /// <returns></returns>
        public static int[] getReshapedPermuted(int[] inputShape, int[] blockShape, int prod)
        {
            var reshapedPermuted = new List<int>() { inputShape[0] / prod };
            for (var i = 1; i < inputShape.Length; ++i)
            {
                if (i <= blockShape.Length)
                {
                    reshapedPermuted.Add(blockShape[i - 1] * inputShape[i]);
                }
                else
                {
                    reshapedPermuted.Add(inputShape[i]);
                }
            }
            return reshapedPermuted.ToArray();
        }
        /// <summary>
        /// Converts the crops argument into the beginning coordinates of a slice
        /// operation
        /// </summary>
        /// <param name="crops"></param>
        /// <param name="blockShape"></param>
        /// <returns></returns>
        public static int[] getSliceBeginCoords(int[][] crops, int blockShape)
        {
            var sliceBeginCoords = new List<int>() { 0 };
            for (var i = 0; i < blockShape; ++i)
            {
                sliceBeginCoords.Add(crops[i][0]);
            }
            return sliceBeginCoords.ToArray();
        }

        /// <summary>
        /// converts the crops argument into the size of a slice operation.  When
        /// combined with getSliceBeginCoords this function allows the reshaped and
        /// permuted Tensor to be cropped to its final output shape of:
        /// 
        /// inputShape[1] * blockShape[0] - crops[0,0] - crops[0,1], ...,
        /// inputShape[M] * blockShape[M-1] -crops[M-1,0] -
        /// crops[M-1,1],inputShape[M+1], ..., inputShape[N-1]]
        /// 
        /// See step 4: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
        /// </summary>
        /// <param name="uncroppedShape"></param>
        /// <param name="crops"></param>
        /// <param name="blockShape"></param>
        /// <returns></returns>
        public static int[] getSliceSize(int[] uncroppedShape, int[][] crops, int blockShape)
        {
            var sliceSize = new List<int> (uncroppedShape.Slice(0, 1));
            for (var i = 0; i < blockShape; ++i)
            {
                sliceSize.Add(uncroppedShape[i + 1] - crops[i][0] - crops[i][1]);
            }

            return sliceSize.ToArray();
        }


        public static bool ArrayIsEqual(int[] arr1, int[] arr2)
        {
            if (arr1 == null && arr2 != null)
            {
                return false;
            }
            if (arr1 != null && arr2 == null)
            {
                return false;
            }

            if (arr1 == null && arr2 == null)
            {
                return true;
            }
            if (arr1.Length != arr2.Length)
            {
                return false;
            }
            for (int i = 0; i < arr1.Length; i++)
            {
                if (arr1[i] != arr2[i])
                {
                    return false;
                } 
            }
            return true;
        }
        /// <summary>
        ///  Calculate the product of an array of numbers.
        /// </summary>
        /// <param name="array"></param>
        /// <param name="begin"></param>
        /// <param name="end"></param>
        /// <returns></returns>
        public static int arrayProd(int[] array, int begin = 0, Nullable<int> end = null)
        {
            if (end == null)
            {
                end = array.Length;
            }

            var prod = 1;
            for (var i = begin; i < end; ++i)
            {
                prod *= array[i];
            }
            return prod;
        }
        public static void assert(bool expr, string msg)
        {
            if (!expr)
            {
                throw new Exception(msg);
            }
        }

        public static int[][] getStridedSlicedInfo(int[] shape, int[] begin, int[] end, int[] strides,
            int beginMask = 0, int endMask = 0)
        {
            // Note that the axis orders are reversed for runtime ops, so the indices,
            // strides and masks must be as well too.
            var startIndex = new int[shape.Length];
            var endIndex = new int[shape.Length];
            for (var i = 0; i < shape.Length; i++)
            {
                startIndex[i] = startForAxis(beginMask, begin, strides, shape, i);
                endIndex[i] = stopForAxis(endMask, end, strides, shape, i);
            }
             
            var size = new int[shape.Length];
            size = size.Select((d, i) =>
            {
                var count = 0;
                for (var start = startIndex[i];
                     !(strides[i] > 0 ? start >= endIndex[i] : start <= endIndex[i]);
                     start += strides[i])
                {
                    count += 1;
                }
                return count;
            }).ToArray();
            return new int[][] { startIndex, size }; 
        }

        public static int startForAxis(int beginMask, int[] startIndices, int[] strides, int[] inputShape, int axis)
        {
            // Begin with the specified index
            var start = startIndices[axis];

            var asd = beginMask & 1 << axis;
            // Check the axis bit from right of beginMask
            if ((beginMask & 1 << axis) == 1)
            {
                if (strides[axis] > 0)
                {
                    // Forward iteration - use the first element. These values will get
                    // clamped below (Note: We could have set them to 0 and axis_size-1, but
                    // use lowest() and max() to maintain symmetry with StopForAxis())
                    start = int.MinValue;
                }
                else
                {
                    // Backward iteration - use the last element.
                    start = int.MaxValue;
                }
            }

            // Handle negative indices
            var axisSize = inputShape[axis];
            if (start < 0)
            {
                start += axisSize;
            }

            // Clamping
            start = Util.clamp(0, start, axisSize - 1);

            return start;
        }

        public static int stopForAxis(int endMask, int[] stopIndices, int[] strides, int[] inputShape, int axis)
        {
            // Begin with the specified index
            var stop = stopIndices[axis];

            // Check the axis bit from right of endMask
            if ((endMask & (1 << axis)) == 1)
            {
                if (strides[axis] > 0)
                {
                    // Forward iteration - use the last element. These values will get
                    // clamped below
                    stop = int.MaxValue;
                }
                else
                {
                    // Backward iteration - use the first element.
                    stop = int.MinValue;
                }
            }

            // Handle negative indices
            var axisSize = inputShape[axis];
            if (stop < 0)
            {
                stop += axisSize;
            }

            // Clamping
            // Because the end index points one past the last element, we need slightly
            // different clamping ranges depending on the direction.
            if (strides[axis] > 0)
            {
                // Forward iteration
                stop = Util.clamp(0, stop, axisSize);
            }
            else
            {
                // Backward iteration
                stop = Util.clamp(-1, stop, axisSize - 1);
            }

            return stop;
        }

        public static int clamp(int min, int x, int max)
        {
            return (int)Math.Max((float)min, Math.Min((float)x, max));
        }
        public static float clamp(float min, float x, float max)
        {
            return (float)Math.Max((float)min, Math.Min((float)x, max));
        }
        //public static int[] getBroadcastDims(int[] inShape, int[] outShape)
        //{
        //    var inRank = inShape.Length;
        //    List<int> dims = new List<int>();

        //    for (var i = 0; i < inRank; i++)
        //    {
        //        var dim = inRank - 1 - i;
        //        var a = (dim) > -1 && (dim) < inShape.Length ? inShape[dim] : 1;

        //        var b = (outShape.Length - i - 1) > -1 && (outShape.Length - 1 - i)
        //            < outShape.Length ? outShape[outShape.Length - 1 - i] : 1;
        //        if (b > 1 && a == 1)
        //        {
        //            dims.Insert(0, dim);
        //        }
        //    }

        //    return dims.ToArray();
        //}
        public static int[] getBroadcastDims(int[] inShape, int[] outShape)
        {
            var inRank = inShape.Length;
            List<int> dims = new List<int>();

            for (var i = 0; i < inRank; i++)
            {
                var dim = inRank - 1 - i;
                var a = (dim) > -1 && (dim) < inShape.Length ? inShape[dim] : 1;

                var b = (outShape.Length - i - 1) > -1 && (outShape.Length - 1 - i) 
                    < outShape.Length ? outShape[outShape.Length - 1 - i] : 1;
                if (b > 1 && a == 1)
                {
                    dims.Insert(0, dim);
                }
            }
            
            return dims.ToArray();
        }

        public static int[] getBroadcastDims2(int[] inShape, int[] outShape)
        {
            var inRank = inShape.Length; 
            int[] dims2 = new int[inShape.Length];

            for (var i = 0; i < inRank; i++)
            {
                var dim = inRank - 1 - i;
                var a = (dim) > -1 && (dim) < inShape.Length ? inShape[dim] : 1;

                var b = (outShape.Length - i - 1) > -1 && (outShape.Length - 1 - i)
                    < outShape.Length ? outShape[outShape.Length - 1 - i] : 1;
                if (b > 1 && a == 1)
                {
                    dims2[(dims2.Length-1) - i] = 0; 
                }
                else
                {
                    dims2[(dims2.Length - 1) - i] = -1;
                }
            } 
            return dims2;
        }




        public static double SELU_SCALEALPHA = 1.7580993408473768599402175208123;
        public static double SELU_SCALE = 1.0507009873554804934193349852946;

        public static Conv2DInfo computePool2DInfo(int[] inShape, int[] filterSize,
            int[] strides, PadType pad, roundingMode roundingMode,
            ConvDataFormat dataFormat = ConvDataFormat.channelsLast, Nullable<int> padValue = null)
        {
            var filterHeight = filterSize[0];
            var filterWidth = filterSize[1];
            int[] filterShape;
            if (dataFormat == ConvDataFormat.channelsLast)
            {
                filterShape = new int[] { filterHeight, filterWidth, inShape[3], inShape[3] };
            }
            else if (dataFormat == ConvDataFormat.channelsFirst)
            {
                filterShape = new int[] { filterHeight, filterWidth, inShape[1], inShape[1] };
            }
            else
            {
                throw new Exception("Unknown dataFormat");
            }
            var dilations = 1;

            return computeConv2DInfo(
                inShape, filterShape, strides, new int[] { dilations }, pad, roundingMode, false,
                dataFormat, padValue);
        }

        /// <summary>
        /// apply floor, round or ceil rounding on value
        /// </summary>
        /// <param name="value"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static float conditionalRound(float value, roundingMode mode)
        {
            if (mode == roundingMode.none)
            {
                return value;
            }
            switch (mode)
            {
                case roundingMode.floor:
                    return (float)Math.Floor(value);
                    break;
                case roundingMode.round:
                    return (float)Math.Round(value);
                    break;
                case roundingMode.ceil:
                    return (float)Math.Ceiling(value);
                    break;
                default:
                    throw new Exception("Unknown roundingMode");
                    break;
            }
        }


        public static Conv2DInfo computeConv2DInfo(int[] inShape, int[] filterShape,
            int[] strides, int[] dilations, PadType pad,
            roundingMode roundingMode = roundingMode.none, bool depthwise = false,
            ConvDataFormat dataFormat = ConvDataFormat.channelsLast, Nullable<int> padValue = null)
        {
            var batchSize = -1;
            var inHeight = -1;
            var inWidth = -1;
            var inChannels = -1;
            if (dataFormat == ConvDataFormat.channelsLast)
            {
                batchSize = inShape[0];
                inHeight = inShape[1];
                inWidth = inShape[2];
                inChannels = inShape[3];
            }
            else
            {
                batchSize = inShape[0];
                inChannels = inShape[1];
                inHeight = inShape[2];
                inWidth = inShape[3];
            }

            var filterHeight = filterShape[0];
            var filterWidth = filterShape[1];
            var filterChannels = filterShape[3];

            var strideHeight = strides[0];
            var strideWidth = strides[1];

            var dilationHeight = dilations[0];
            int dilationWidth = 0;
            if (dilations.Length > 1)
            {

                  dilationWidth = dilations[1];
            }
            else
            {

                dilationWidth = dilations[0];
            }

            var effectiveFilterHeight = getEffectiveFilterSize(filterHeight, dilationHeight);
            var effectiveFilterWidth =
     getEffectiveFilterSize(filterWidth, dilationWidth);



            var d = getPadAndOutInfo(
      pad, inHeight, inWidth, strideHeight, strideWidth, effectiveFilterHeight,
      effectiveFilterWidth, roundingMode, padValue);
            var padInfo = d.Item1;
            var outHeight = d.Item2;
            var outWidth = d.Item3;
            var outChannels = depthwise ? filterChannels * inChannels : filterChannels;
            int[] outShape = null;
            if (dataFormat == ConvDataFormat.channelsFirst)
            {
                outShape = new int[] { batchSize, outChannels, outHeight, outWidth };
            }
            else if (dataFormat == ConvDataFormat.channelsLast)
            {
                outShape = new int[] { batchSize, outHeight, outWidth, outChannels };
            }


            return new Conv2DInfo()
            {
                batchSize = batchSize,
                dataFormat = dataFormat,
                inHeight = inHeight,
                inWidth = inWidth,
                inChannels = inChannels,
                outHeight = outHeight,
                outWidth = outWidth,
                outChannels = outChannels,
                padInfo = padInfo,
                strideHeight = strideHeight,
                strideWidth = strideWidth,
                filterHeight = filterHeight,
                filterWidth = filterWidth,
                dilationHeight = dilationHeight,
                dilationWidth = dilationWidth,
                inShape = inShape,
                outShape = outShape,
                filterShape = filterShape
            };
        }

        public static Tuple<PadInfo, int, int> getPadAndOutInfo(PadType pad, int inHeight, int inWidth,
            int strideHeight, int strideWidth, int filterHeight, int filterWidth,
            roundingMode roundingMode = roundingMode.none, Nullable<int> padValue = null)
        {
            var padInfo = new PadInfo();
            var outHeight = 0;
            var outWidth = 0;

            if (pad == PadType.number)
            {
                padInfo.bottom = padValue.Value;
                padInfo.left = padValue.Value;
                padInfo.right = padValue.Value;
                padInfo.top = padValue.Value;

                var outShape = computeOutputShape3D(
     new int[] { inHeight, inWidth, 1 }, filterHeight, 1, strideHeight, padValue,
      roundingMode);
                outHeight = outShape[0];
                outWidth = outShape[1];
                padInfo.alongh = padValue.Value;
                padInfo.alongw = padValue.Value;
            }
            else if (pad == PadType.valid)
            {
                padInfo.bottom = 0;
                padInfo.left = 0;
                padInfo.right = 0;
                padInfo.top = 0;
                outHeight = (int)Math.Ceiling((inHeight - filterHeight + 1d) / strideHeight);
                outWidth = (int)Math.Ceiling((inWidth - filterWidth + 1d) / strideWidth);
                padInfo.alongh = 0;
                padInfo.alongw = 0;
            }
            else if (pad == PadType.same)
            {
                outHeight = (int)Math.Ceiling(inHeight / (float)strideHeight);
                outWidth = (int)Math.Ceiling(inWidth / (float)strideWidth);

                var padAlongHeight =
        (outHeight - 1) * strideHeight + filterHeight - inHeight;
                var padAlongWidth = (outWidth - 1) * strideWidth +
                    filterWidth - inWidth;

                var top = (int)Math.Floor(padAlongHeight / 2f);
                var bottom = (int)padAlongHeight - top;
                var left = (int)Math.Floor(padAlongWidth / 2f);
                var right = (int)padAlongWidth - left;

                padInfo.bottom = bottom;
                padInfo.left = left;
                padInfo.right = right;
                padInfo.top = top;

                padInfo.alongh = padAlongHeight;
                padInfo.alongw = padAlongWidth;
            }
            else
            {
                throw new Exception("Unknown padding parameter");
            }


            return new Tuple<PadInfo, int, int>(padInfo, outHeight, outWidth);
        }

        public static int[] computeOutputShape3D(int[] inShape,
            int fieldSize, int outDepth, int stride, Nullable<int> zeroPad, roundingMode roundingMode)
        {
            if (zeroPad == null)
            {
                zeroPad = computeDefaultPad(inShape, fieldSize, stride);
            }

            var inputRows = inShape[0];
            var inputCols = inShape[1];

            var outputRows = (int)conditionalRound(
                (inputRows - fieldSize + 2 * zeroPad.Value) / stride + 1, roundingMode);

            var outputCols = (int)conditionalRound(
      (inputCols - fieldSize + 2 * zeroPad.Value) / stride + 1, roundingMode);

            return new int[] { outputRows, outputCols, outDepth };
        }

        public static int computeDefaultPad(int[] inputShape, int fieldSize, int stride, int dilation = 1)
        {
            var effectiveFieldSize = getEffectiveFilterSize(fieldSize, dilation);
            return (int)Math.Floor(
              ((float)inputShape[0] * (stride - 1) - stride + effectiveFieldSize) / 2);
        }
        public static int getEffectiveFilterSize(int filterSize, int dilation)
        {
            if (dilation <= 1)
            {
                return filterSize;
            }

            return filterSize + (filterSize - 1) * (dilation - 1);
        }
        public static int[] getReductionAxes(int[] inShape, int[] outShape)
        {
            List<int> result = new List<int>();
            for (var i = 0; i < outShape.Length; i++)
            {
                var inAxis = inShape.Length - i - 1;

                int inDim2 = -1;
                if (inAxis < inShape.Length && inAxis >= 0)
                {
                    inDim2 = inShape[inAxis];
                }

                var outAxis = outShape.Length - i - 1;
                var outDim = outShape[outAxis];
                if (inDim2 == -1 || (inDim2 == 1 && outDim > 1))
                {
                    result.Insert(0, outAxis);
                }
            }
            return result.ToArray();
        }
        public static int[] assertAndGetBroadcastShape(int[] shapeA, int[] shapeB)
        {
            List<int> result = new List<int>();
            int l = Math.Max(shapeA.Length, shapeB.Length);


            for (var i = 0; i < l; i++)
            {
                int a = (shapeA.Length - i - 1) < shapeA.Length && (shapeA.Length - i - 1) > -1 ? shapeA[shapeA.Length - i - 1] : 1;
                int b = (shapeB.Length - i - 1) < shapeB.Length && (shapeB.Length - i - 1) > -1 ? shapeB[shapeB.Length - i - 1] : 1;
                if (a > 1 && b > 1 && a != b)
                {
                    throw new Exception("Operands could not be broadcast together");
                }
                result.Insert(0, Math.Max(a, b));
            }

            return result.ToArray();
        }

        public static Tuple<int[], int[]> computeOutAndReduceShapes(int[] aShape, int[] axes)
        {
            List<int> outShape = new List<int>();
            var rank = aShape.Length;
            for (var dim = 0; dim < rank; dim++)
            {
                if (!axes.Contains(dim) )
                {
                    outShape.Add(aShape[dim]);
                }
            }
            var reduceShape = axes.Select(dim => aShape[dim]).ToArray();
            return new Tuple<int[], int[]>(outShape.ToArray(), reduceShape);

        }

        public static computeGradientSliceShapesResutl computeGradientSliceShapes(int[] aShape, int[] bShape)
        {
            return new computeGradientSliceShapesResutl()
            {
                aBegin = new int[] { 0, 0 },
                aSize = aShape,
                bBegin = new int[] { 0, aShape[1] },
                bSize = bShape
            };
        }

        public static int[] computeOutShape(int[] x1Shape, int[] x2Shape, int axis)
        {
            var outputShape = new List<int>( x1Shape).ToArray();
            outputShape[axis] += x2Shape[axis];
            return outputShape;
        }

        public static bool axesAreInnerMostDims(int[] axes, int rank)
        {
            for (var i = 0; i < axes.Length; ++i)
            {
                if (axes[axes.Length - i - 1] != rank - 1 - i)
                {
                    return false;
                }
            }
            return true;
        }

        public static int[] getAxesPermutation(int[] axes, int rank)
        {
            if (axesAreInnerMostDims(axes, rank))
            {
                return null;
            }
            List<int> result = new List<int>();
            for (var i = 0; i < rank; ++i)
            {
                if (axes.ToList().IndexOf(i) == -1)
                {
                    result.Add(i);
                }
            }
            result.AddRange(axes);
            return result.ToArray();
        }
        public static int[] parseAxisParam(int[] axis, int[] shape)
        {
            var rank = shape.Length;
            axis = axis == null ? shape.Select((s, i) => i).ToArray() : axis;
            return axis.Select(a => a < 0 ? rank + a : a).ToArray();
        }

        public static int[] getUndoAxesPermutation(int[] axes)
        {
            return axes.Select((axis, i) => new int[] { i, axis })
      .OrderBy((a) => a[1])
      .Select(x => x[0]).ToArray();
        }
        static Random r1 = new Random(23);
        public static double randUniform(double a, double b)
        {
            return r1.NextDouble() * (b - a) + a;
        }
        public static int SizeFromShape(int[] shape)
        {
            if (shape.Length == 0)
            {
                return 1;
            }
            var size = shape[0];
            for (int i = 1; i < shape.Length; i++)
            {
                size *= shape[i];

            }
            return size;
        }
        public static int[] InferFromImplicitShape(int[] shape, int size)
        {
            int shapeProd = 1;
            int implicitIdx = -1;

            for (int i = 0; i < shape.Length; ++i)
            {
                if (shape[i] > 0)
                {
                    shapeProd *= shape[i];
                }
                else if (shape[i] == -1)
                {
                    if (implicitIdx != -1)
                    {
                        throw new Exception("Shapes can only have 1 implicit size. " +
                        Environment.NewLine +
                        "Found -1 at dim " + implicitIdx.ToString()
                        + " and dim " + i.ToString());
                    }
                    implicitIdx = i;
                }
                else if (shape[i] <= 0)
                {
                    throw new Exception("Shapes can not be <= 0. Found " +
      shape[i].ToString() + "  at dim " + i.ToString());
                }
            }
            if (implicitIdx == -1)
            {
                if (size > 0 && size != shapeProd)
                {
                    throw new Exception("Size " + size.ToString() +
                    " must match the product of shape " + shape.ToString());
                }
                return shape;
            }
            if (size % shapeProd != 0)
            {
                throw new Exception("The implicit shape can't be a fractional number Got " + size.ToString() + " / " + shapeProd.ToString());
            }
            var newShap = shape.ToArray();
            newShap[implicitIdx] = size / shapeProd;
            return newShap;
        }

        /// <summary>
        /// Reduces the shape by removing all dimensions of shape 1
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static SqueezeResult SqueezeShape(int[] shape, int[] axis = null)
        {
            List<int> newShape = new List<int>();
            List<int> keptDims = new List<int>();

            int j = 0;
            for (int i = 0; i < shape.Length; i++)
            {
                if (axis != null)
                {
                    if (axis[j] == i && shape[i] > 1)
                    {
                        throw new Exception("Can't squeeze axis " + i.ToString() + " since its dim " + shape[i].ToString() + " is not 1");

                    }
                    if ((axis[j] == null || axis[j] > i) && shape[i] == 1)
                    {
                        newShape.Add(shape[i]);
                        keptDims.Add(i);
                    }
                    if (axis[j] <= i)
                    {
                        j++;
                    }
                }
                if (shape[i] > 1)
                {
                    newShape.Add(shape[i]);
                    keptDims.Add(i);
                }
            }
            return new SqueezeResult()
            {
                KeptDims = keptDims.ToArray(),
                NewShape = newShape.ToArray()
            };
        }

        public static int[] getInnerMostAxes(int numAxes, int rank)
        {
            List<int> res = new List<int>();
            for (var i = rank - numAxes; i < rank; ++i)
            {
                res.Add(i);
            }
            return res.ToArray();
        }

        public static int[] combineLocations(int[] outputLoc, int[] reduceLoc, int[] axes)
        {
            var rank = outputLoc.Length + reduceLoc.Length;
            var loc = new List<int>();
            var outIdx = 0;
            var reduceIdx = 0;
            for (var dim = 0; dim < rank; dim++)
            {
                if (axes.ToList().IndexOf(dim) == -1)
                {
                    loc.Add(outputLoc[outIdx++]);
                }
                else
                {
                    loc.Add(reduceLoc[reduceIdx++]);
                }
            }
            return loc.ToArray();
        }

        public static int[] expandShapeToKeepDim(int[] shape, int[] axes)
        {
            var reduceSubShape = axes.Select(x => 1).ToArray();
            return combineLocations(shape, reduceSubShape, axes);
        }
    }

    public class Shape
    {
        private int[] _shape;
        public Shape(params int[] shape)
        {
            this._shape = shape;
        }
        public static implicit operator int[](Shape m)
        {
            return m._shape;
        }
        public static explicit operator Shape(int[] m)
        {
            Shape money = new Shape(m);
            return money;
        }
        public override string ToString()
        {
            if (_shape.Length ==0)
            {
                return "[0]";
            }
            StringBuilder sb = new StringBuilder();
            sb.Append("[");
            for (int i = 0; i < _shape.Length-1; i++)
            {
                sb.Append(_shape[i].ToString()+", ");
            }
            sb.Append(_shape[_shape.Length - 1].ToString());
            sb.Append("]");
            return sb.ToString();
        }
    }

    public static class AlbiruniExtension
    {
        /// <summary>
        /// Get the array slice between the two indexes.
        /// ... Inclusive for start index, exclusive for end index.
        /// </summary>
        public static T[] Slice<T>(this T[] source )
        {

            var len = source.Length;
         
             
            // Return new array.
            T[] res = new T[len];
            for (int i = 0; i < len; i++)
            {
                res[i] = source[i ];
            }
            return res;
        }
        /// <summary>
        /// Get the array slice between the two indexes.
        /// ... Inclusive for start index, exclusive for end index.
        /// </summary>
        public static T[] Slice<T>(this T[] source, int start, int end)
        {
            // Handles negative ends.
            if (end < 0)
            {
                end = source.Length + end;
            }
            int len = end - start;

            // Return new array.
            T[] res = new T[len];
            for (int i = 0; i < len; i++)
            {
                res[i] = source[i + start];
            }
            return res;
        }
        /// <summary>
        /// Get the array slice between the two indexes.
        /// ... Inclusive for start index, exclusive for end index.
        /// </summary>
        public static T[] Slice<T>(this T[] source, int start )
        {
 
            int len = source.Length - start;

            // Return new array.
            T[] res = new T[len];
            for (int i = 0; i < len; i++)
            {
                res[i] = source[i + start];
            }
            return res;
        }

        public static void Fill<T>(this T[] source, T val)
        {
            for (int i = 0; i < source.Length; i++)
            {
                source[i] = val;
            }
             
        }
        public static Tensor ToTensor(this int i)
        {
            return Ops.scalar(i);
        }
        public static Tensor ToTensor(this float f)
        {
            return Ops.scalar(f);
        }
        public static Tensor ToTensor(this double d)
        {
            return Ops.scalar((float)d);
        }
        public static Tensor ToTensor(this bool b)
        {
            return Ops.scalar(Convert.ToSingle(b));
        }

        public static Tensor ToTensor(this float[] arr)
        {
            return Ops.tensor1d(arr);
        }
        public static Tensor ToTensor(this float[] arr,params int[] shape )
        {
            return Ops.tensor(arr,shape);
        }
        public static Tensor ToTensor(this int[] arr)
        {
            float[] val = new float[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                val[i] = arr[i];
            }


            return Ops.tensor1d(val);
        }
        public static Tensor ToTensor(this int[] arr, params int[] shape)
        {
            float[] val = new float[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                val[i] = arr[i];
            }

            return Ops.tensor(val, shape);
        }


        public static Tensor ToTensor(this double[] arr)
        {
            float[] val = new float[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                val[i] =(float) arr[i];
            }


            return Ops.tensor1d(val);
        }
        public static Tensor ToTensor(this double[] arr, params int[] shape)
        {
            float[] val = new float[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                val[i] = (float) arr[i];
            }

            return Ops.tensor(val, shape);
        }


        public static Tensor ToTensor(this bool[] arr)
        {
            float[] val = new float[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                val[i] = Convert.ToSingle( arr[i]);
            }


            return Ops.tensor1d(val);
        }
        public static Tensor ToTensor(this bool[] arr, params int[] shape)
        {
            float[] val = new float[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                val[i] =Convert.ToSingle( arr[i]);
            }

            return Ops.tensor(val, shape);
        }


        private static Random rng = new Random();
        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

    }
}
