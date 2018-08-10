
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public enum PadType
    {
        valid,
        same,
        number
    }
    public static partial class Ops
    {

        /// <summary>
        /// Bilinear resize a batch of 3D images to a new shape.
        /// </summary>
        /// <param name="images">The images, of rank 4 or rank 3, of shape
        /// `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.</param>
        /// <param name="size">The new shape `[newHeight, newWidth]` to resize the
        /// images to. Each channel is resized individually.</param>
        /// <param name="alignCorners">Defaults to False. If true, rescale
        /// input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4
        ///   corners of images and resized images. If false, rescale by
        ///   `new_height / height`. Treat similarly the width dimension.</param>
        /// <returns></returns>
        public static Tensor resizeBilinear(this Tensor images, int[] size, bool alignCorners = false)
        {
            Tensor batchImages = null;
            var reshapedTo4D = false;
            if (images.Rank == 3)
            {
                reshapedTo4D = true;
                batchImages =
                    images.as4D(1, images.Shape[0], images.Shape[1], images.Shape[2]);
            }
            else
            {
                batchImages = images as Tensor;
            }
            Engine e = ENV.engine;
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () => {
                    ForwardFunc fb = (IBackend bk, Func<Tensor, Tensor> saved) =>
                    {
                        return bk.resizeBilinearBackprop(dy, batchImages, alignCorners);
                    };
                    return e.runKernel(fb, new Dictionary<string, Tensor>());
                
                
                });
                return g;
            };
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.resizeBilinear(batchImages, size[0], size[1], alignCorners);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("batchImages", batchImages);
            var res = e.runKernel(f, inputs);

            if (reshapedTo4D)
            {
                return res.as3D(res.Shape[1], res.Shape[2], res.Shape[3]);
            }
            return res;
        }

        /// <summary>
        /// Computes a 1D convolution over the input x.
        /// </summary>
        /// <param name="x">The input tensor, of Rank 3 or Rank 2, of shape
        /// "[batch, width, inChannels]". If Rank 2, batch of 1 is assumed.</param>
        /// <param name="filter">The filter, Rank 3, of shape
        ///    "[filterWidth, inDepth, outDepth]".</param>
        /// <param name="stride">The number of entries by which the filter is moved right at
        ///    each step.</param>
        /// <param name="pad">The type of padding algorithm.
        ///   - "same" and stride 1: output will be of same size as input,
        ///      regardless of filter size.
        ///   - "valid": output will be smaller than input if filter is larger
        ///      than 1x1.
        ///  - For more info, see this guide:
        ///    [https://www.tensorflow.org/api_guides/python/nn#Convolution](
        ///         https://www.tensorflow.org/api_guides/python/nn#Convolution)
        /// </param>
        /// <param name="dilation">The dilation rate in which we sample input values in
        ///     atrous convolution. Defaults to "1". If it is greater than 1, then
        ///     stride must be "1".</param>
        /// <param name="dimRoundingMode">The rounding mode used when computing output
        ///    dimensions if pad is a number. If none is provided, it will not round
        ///    and error if the output is of fractional size.</param>
        /// <param name="padvalue">the value of pad if pad is number</param>
        /// <returns></returns>
        public static Tensor conv1d(this Tensor x, Tensor filter, int stride,
            PadType pad, int dilation = 1,
            roundingMode dimRoundingMode = roundingMode.none, Nullable<int> padvalue = null)
        {
            var input3D = x;
            var reshapedTo3D = false;
            if (x.Rank == 2)
            {
                reshapedTo3D = true;
                input3D = x.as3D(1, x.Shape[0], x.Shape[1]);
            }
            var filter4D =
        filter.as4D(1, filter.Shape[0], filter.Shape[1], filter.Shape[2]);
            var input4D =
                input3D.as4D(input3D.Shape[0], 1, input3D.Shape[1], input3D.Shape[2]);
            int[] strides = new int[] { 1, stride };
            int[] dilations = new int[] { 1, dilation };
            var res = conv2d(
        input4D, filter4D, strides, pad, dilations,
        dimRoundingMode, padvalue);
            if (reshapedTo3D)
            {
                return res.as2D(res.Shape[2], res.Shape[3]);
            }
            return res.as3D(res.Shape[0], res.Shape[2], res.Shape[3]);

        }


        /// <summary>
        /// Computes a 2D convolution over the input x.
        /// </summary>
        /// <param name="x">The input tensor, of Rank 4 or Rank 3, of shape
        ///   "[batch, height, width, inChannels]". If Rank 3, batch of 1 is</param>
        /// <param name="filter">The filter, Rank 4, of shape
        ///  "[filterHeight, filterWidth, inDepth, outDepth]".</param>
        /// <param name="strides">The strides of the convolution: "[strideHeight,
        /// strideWidth]".</param>
        /// <param name="pad">The type of padding algorithm.
        ///   - "same" and stride 1: output will be of same size as input,
        ///      regardless of filter size.
        ///   - "valid": output will be smaller than input if filter is larger
        ///      than 1x1.
        ///  - For more info, see this guide:
        ///    [https://www.tensorflow.org/api_guides/python/nn#Convolution](
        ///         https://www.tensorflow.org/api_guides/python/nn#Convolution)</param>
        /// <param name="dilations">The dilation rates: "[dilationHeight, dilationWidth]"
        ///     in which we sample input values across the height and width dimensions
        ///     in atrous convolution. Defaults to "[1, 1]". If "dilations" is a single
        ///     number, then "dilationHeight == dilationWidth". If it is greater than
        ///     1, then all values of "strides" must be 1.</param>
        /// <param name="dimRoundingMode">The rounding mode used when computing output
        ///     dimensions if pad is a number. If none is provided, it will not round
        ///     and error if the output is of fractional size.</param> 
        /// <param name="padvalue">the value of pad if pad is number</param>
        /// <returns></returns>
        public static Tensor conv2d(this Tensor x, Tensor filter, int[] strides,
            PadType pad, int[] dilations = null, roundingMode dimRoundingMode = roundingMode.none,
            Nullable<int> padValue = null)
        {
            if (dilations == null)
            {
                dilations = new int[] { 1, 1 };
            }

            Tensor x4D = null;
            var reshapedTo4D = false;

            if (x.Rank == 3)
            {
                reshapedTo4D = true;
                x4D = x.as4D(1, x.Shape[0], x.Shape[1], x.Shape[2]);
            }
            else
            {
                x4D = x as Tensor;
            }
            var convInfo = Util.computeConv2DInfo(
        x4D.Shape, filter.Shape, strides, dilations, pad,
        dimRoundingMode, false, ConvDataFormat.channelsLast, padValue);

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy,
                List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    var resg = conv2dDerInput(x4D.Shape, dy, filter, strides, pad, dimRoundingMode, padValue);

                    if (reshapedTo4D)
                    {
                        resg = resg.as3D(resg.Shape[1], resg.Shape[2], resg.Shape[3]);
                    }
                    return resg;
                });

                g.gradient.Add("filter", () =>
                {
                    return conv2dDerFilter(x4D, dy, filter.Shape, strides, pad, dimRoundingMode, padValue);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.conv2d(x4D, filter, convInfo);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            inputs.Add("filter", filter);
            var res = e.runKernel(f, inputs, grad);
            if (reshapedTo4D)
            {
                var newres = res.as3D(res.Shape[1], res.Shape[2], res.Shape[3]);
                return newres;
            }


            return res;

        }

        /// <summary>
        /// Computes the derivative of the filter of a 2D convolution.
        /// </summary>
        /// <param name="x">The input tensor, of Rank 4 or Rank 3 of shape
        ///  [batch, height, width, inChannels]. If Rank 3, batch of 1 is assumed.</param>
        /// <param name="dy">The dy image, of Rank 4 or Rank 3, of shape
        ///  [batch, height, width, outDepth]. If Rank 3, batch of 1 is assumed.</param>
        /// <param name="filterShape">The shape of the filter, length 4,
        ///   [filterHeight, filterWidth, inDepth, outDepth].</param>
        /// <param name="strides">The strides of the convolution: [strideHeight,
        /// strideWidth].</param>
        /// <param name="pad">A string from: 'same', 'valid'. The type of padding algorithm
        /// used in the forward prop of the op.</param>
        /// <param name="dimRoundingMode">A string from: 'ceil', 'round', 'floor'. The
        ///    rounding mode used when computing output dimensions if pad is a
        ///    number. If none is provided, it will not round and error if the output
        ///    is of fractional size.</param>
        /// <param name="padValue">the value of pad if pad is number</param>
        /// <returns></returns>
        public static Tensor conv2dDerFilter(this Tensor x, Tensor dy, int[] filterShape, int[] strides,
            PadType pad, roundingMode dimRoundingMode = roundingMode.none, Nullable<int> padValue = null)
        {
            Tensor x4D = null;
            if (x.Rank == 3)
            {
                x4D = x.as4D(1, x.Shape[0], x.Shape[1], x.Shape[2]);
            }
            else
            {
                x4D = x as Tensor;
            }
            Tensor dy4D = null;
            if (dy.Rank == 3)
            {
                dy4D = dy.as4D(1, dy.Shape[0], dy.Shape[1], dy.Shape[2]);
            }
            else
            {
                dy4D = dy as Tensor;
            }

            var dilations = 1;

            var convInfo = Util.computeConv2DInfo(
                x4D.Shape, filterShape, strides, new int[] { dilations, dilations }, pad, dimRoundingMode, false, ConvDataFormat.channelsLast, padValue);
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.conv2dDerFilter(x4D, dy4D, convInfo);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x4D", x4D);
            inputs.Add("dy4D", dy4D);
            return e.runKernel(f, inputs);
        }

        /// <summary>
        /// Computes the derivative of the input of a 2D convolution.
        /// </summary>
        /// <param name="xShape">The shape of the input: [batch, height, width, inDepth].
        ///If length of 3, batch of 1 is assumed.</param>
        /// <param name="dy">The derivative of the output, of Rank 4 or Rank 3 of shape
        ///   "[batch, outHeight, outWidth, outDepth]". If Rank 3, batch of 1 is
        /// assumed.</param>
        /// <param name="filter">The filter, Rank 4, of shape
        /// "[filterHeight, filterWidth, inDepth, outDepth]".</param>
        /// <param name="strides">The strides of the convolution: "[strideHeight,
        /// strideWidth]".</param>
        /// <param name="pad">The type of padding algorithm used:
        ///   - "same" and stride 1: output will be of same size as input,
        ///      regardless of filter size.
        ///   - "valid": output will be smaller than input if filter is larger
        ///      than 1x1.</param>
        /// <param name="dimRoundingMode">he rounding mode used when computing output
        ///   dimensions if pad is a number. If none is provided, it will not round
        ///   and error if the output is of fractional size.</param>
        /// <param name="padValue"></param>
        /// <returns>the value of pad if pad is number</returns>
        public static Tensor conv2dDerInput(int[] xShape, Tensor dy, Tensor filter,
            int[] strides, PadType pad, roundingMode dimRoundingMode, Nullable<int> padValue = null)
        {
            int[] xShape4D = xShape;
            Tensor dy4D = null;
            var reshapedTo4D = false;
            if (dy.Rank == 3)
            {
                reshapedTo4D = true;
                dy4D = dy.as4D(1, dy.Shape[0], dy.Shape[1], dy.Shape[2]);
                xShape4D = new int[] { 1, xShape[0], xShape[1], xShape[2] };
            }
            else
            {
                dy4D = dy as Tensor;
            }


            var inDepth = xShape4D[3];
            var outDepth = dy4D.Shape[3];

            var dilations = 1;

            var convInfo = Util.computeConv2DInfo(
                xShape4D, filter.Shape, strides, new int[] { dilations, dilations },
                pad, dimRoundingMode,
                false, ConvDataFormat.channelsLast, padValue);
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.conv2dDerInput(dy4D, filter, convInfo);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("dy4D", dy4D);
            var res = e.runKernel(f, inputs);
            if (reshapedTo4D)
            {
                return res.as3D(res.Shape[1], res.Shape[2], res.Shape[3]);
            }
            return res;
        }

        /// <summary>
        /// Computes the transposed 2D convolution of an image, also known as a
        /// deconvolution.
        /// </summary>
        /// <param name="x">The input image, of Rank 4 or Rank 3, of shape
        ///  "[batch, height, width, inDepth]". If Rank 3, batch of 1 is assumed.</param>
        /// <param name="filter">The filter, Rank 4, of shape
        ///  "[filterHeight, filterWidth, outDepth, inDepth]".
        ///   "inDepth" must match "inDepth" in "x".</param>
        /// <param name="outputShape">outputShape Output shape, of Rank 4 or Rank 3:
        ///     "[batch, height, width, outDepth]". If Rank 3, batch of 1 is assumed.</param>
        /// <param name="strides">The strides of the original convolution:
        ///     "[strideHeight, strideWidth]".</param>
        /// <param name="pad">The type of padding algorithm used in the non-transpose version
        ///   of the op.</param>
        /// <param name="dimRoundingMode">The rounding mode used when computing output
        ///    dimensions if pad is a number. If none is provided, it will not round
        ///    and error if the output is of fractional size.</param>
        /// <param name="padvalue">the value of pad if pad is number</param>
        /// <returns></returns>
        public static Tensor conv2dTranspose(this Tensor x, Tensor filter, int[] outputShape, int[] strides, PadType pad, roundingMode dimRoundingMode = roundingMode.none,
            Nullable<int> padvalue = null)
        {

            return conv2dDerInput(
           outputShape, x, filter, strides, pad, dimRoundingMode, padvalue);

        }


        /// <summary>
        /// Depthwise 2D convolution.
        ///    * Given a 4D "input" array and a "filter" array of shape
        /// "[filterHeight, filterWidth, inChannels, channelMultiplier]" containing
        /// "inChannels" convolutional filters of depth 1, this op applies a
        /// different filter to each input channel (expanding from 1 channel to
        /// "channelMultiplier" channels for each), then concatenates the results
        /// together. The output has "inChannels * channelMultiplier" channels.
        ///
        /// See
        /// [https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d](
        ///     https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)
        /// for more details.
        /// </summary>
        /// <param name="input">The input tensor, of Rank 4 or Rank 3, of shape
        ///    "[batch, height, width, inChannels]". If Rank 3, batch of 1 is assumed.</param>
        /// <param name="filter">The filter tensor, Rank 4, of shape
        ///  "[filterHeight, filterWidth, inChannels, channelMultiplier]"</param>
        /// <param name="strides">The strides of the convolution: "[strideHeight,
        /// strideWidth]". If strides is a single number, then "strideHeight ==
        /// strideWidth".</param>
        /// <param name="pad">The type of padding algorithm.
        ///  - "same" and stride 1: output will be of same size as input,
        ///      regardless of filter size.
        ///  - "valid": output will be smaller than input if filter is larger
        ///      than 1x1.
        ///  - For more info, see this guide:
        ///    [https://www.tensorflow.org/api_guides/python/nn#Convolution](
        ///         https://www.tensorflow.org/api_guides/python/nn#Convolution)</param>
        /// <param name="dilations">The dilation rates: "[dilationHeight, dilationWidth]"
        ///    in which we sample input values across the height and width dimensions
        ///    in atrous convolution. Defaults to "[1, 1]". If "rate" is a single
        ///    number, then "dilationHeight == dilationWidth". If it is greater than
        ///    1, then all values of "strides" must be 1.</param>
        /// <param name="dimRoundingMode">The rounding mode used when computing output
        ///     dimensions if pad is a number. If none is provided, it will not round
        ///     and error if the output is of fractional size.</param>
        /// <param name="padvalue">the value of pad if pad is number</param>
        /// <returns></returns>
        public static Tensor depthwiseConv2d(this Tensor input, Tensor filter,
            int[] strides, PadType pad, int[] dilations = null,
            roundingMode dimRoundingMode = roundingMode.none, Nullable<int> padvalue = null)
        {
            if (dilations == null)
            {
                dilations = new int[] { 1, 1 };
            }
            Tensor input4D = null;
            var reshapedTo4D = false;
            if (input.Rank == 3)
            {
                reshapedTo4D = true;
                input4D = input.as4D(1, input.Shape[0], input.Shape[1], input.Shape[2]);
            }
            else
            {
                input4D = input as Tensor;
            }

            var convInfo = Util.computeConv2DInfo(
     input4D.Shape, filter.Shape, strides, dilations, pad, dimRoundingMode,
     true /* depthwise */, ConvDataFormat.channelsLast, padvalue);


            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.depthwiseConv2D(input4D, filter, convInfo);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("input4D", input4D);
            inputs.Add("filter", filter);
            var res = e.runKernel(f, inputs);
            if (reshapedTo4D)
            {
                return res.as3D(res.Shape[1], res.Shape[2], res.Shape[3]);
            }


            return res;
        }


        /// <summary>
        /// 2-D convolution with separable filters.
        ///
        /// Performs a depthwise convolution that acts separately on channels followed
        /// by a pointwise convolution that mixes channels. Note that this is
        /// separability between dimensions [1, 2] and 3, not spatial separability
        /// between dimensions 1 and 2.
        ///
        /// See
        /// [https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d](
        ///     https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d)
        /// for more details.
        /// </summary>
        /// <param name="x">The input tensor, of rank 4 or rank 3, of shape
        ///     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
        /// assumed.</param>
        /// <param name="depthwiseFilter">The depthwise filter tensor, rank 4, of shape
        ///     `[filterHeight, filterWidth, inChannels, channelMultiplier]`. This is
        ///     the filter used in the first step.</param>
        /// <param name="pointwiseFilter">The pointwise filter tensor, rank 4, of shape
        ///     `[1, 1, inChannels * channelMultiplier, outChannels]`. This is
        ///     the filter used in the second step.</param>
        /// <param name="strides">The strides of the convolution: `[strideHeight,
        /// strideWidth]`. If strides is a single number, then `strideHeight ==
        /// strideWidth`.</param>
        /// <param name="pad"> The type of padding algorithm.
        ///  - `same` and stride 1: output will be of same size as input,
        ///      regardless of filter size.
        ///  - `valid`: output will be smaller than input if filter is larger
        ///      than 1x1.
        ///  - For more info, see this guide:
        ///    [https://www.tensorflow.org/api_guides/python/nn#Convolution](
        ///         https://www.tensorflow.org/api_guides/python/nn#Convolution)</param>
        /// <param name="dilation">The dilation rates: `[dilationHeight, dilationWidth]`
        ///     in which we sample input values across the height and width dimensions
        ///     in atrous convolution. Defaults to `[1, 1]`. If `rate` is a single
        ///     number, then `dilationHeight == dilationWidth`. If it is greater than
        ///     1, then all values of `strides` must be 1.</param>
        /// <param name="padvalue">the value of pad if pad is number</param>
        /// <returns></returns>
        public static Tensor separableConv2d(this Tensor x, Tensor depthwiseFilter, Tensor pointwiseFilter, int[] strides, PadType pad
            , int[] dilation, Nullable<int> padvalue = null)
        {
            var x4D = x;
            var reshapedTo4D = false;
            if (x.Rank == 3)
            {
                reshapedTo4D = true;
                x4D = x.as4D(1, x.Shape[0], x.Shape[1], x.Shape[2]);
            }

            Util.assert(
                x4D.Rank == 4,
                "Error in separableConv2d: input must be Rank 4, but got " +
                    "Rank ${x4D.Rank}.");
            Util.assert(
                depthwiseFilter.Rank == 4,
                "Error in separableConv2d: depthwise filter must be Rank 4, but got " +
                    "Rank ${depthwiseFilter.Rank}.");
            Util.assert(
                pointwiseFilter.Rank == 4,
                "Error in separableConv2d: pointwise filter must be Rank 4, but got " +
                    "Rank ${depthwiseFilter.Rank}.");
            Util.assert(
                pointwiseFilter.Shape[0] == 1,
                "Error in separableConv2d: the first dimension of pointwise filter " +
                    " must be 1, but got ${pointwiseFilter.shape[0]}.");
            Util.assert(
                pointwiseFilter.Shape[1] == 1,
                "Error in separableConv2d: the second dimension of pointwise filter " +
                    " must be 1, but got ${pointwiseFilter.shape[1]}.");

            var inChannels = depthwiseFilter.Shape[2];
            var channelMultiplier = depthwiseFilter.Shape[3];
            Util.assert(
        pointwiseFilter.Shape[2] == inChannels * channelMultiplier,
        "Error in separableConv2d: the third dimension of pointwise filter " +
            "must be ${inChannels * channelMultiplier}, " +
            "but got ${pointwiseFilter.shape[2]}.");



            var depthwise = depthwiseConv2d(
                x4D, depthwiseFilter, strides, pad, dilation, roundingMode.none, padvalue);
            var pointwiseStride = 1;
            var res = conv2d(
                depthwise, pointwiseFilter,
                new int[] { pointwiseStride, pointwiseStride }, PadType.valid);
            if (reshapedTo4D)
            {
                return res.as3D(res.Shape[1], res.Shape[2], res.Shape[3]);
            }
            return res;
        }

      
    }
}
