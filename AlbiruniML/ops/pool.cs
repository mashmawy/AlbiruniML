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
        /// Computes the 2D max pooling of an image.
        /// </summary>
        /// <param name="x">The input tensor, of rank 4 or rank 3 of shape
        ///    `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed</param>
        /// <param name="filterSize">The filter size, a tuple `[filterHeight, filterWidth]`.</param>
        /// <param name="strides">The strides of the pooling: `[strideHeight, strideWidth]`.</param>
        /// <param name="pad"> The type of padding algorithm.     
        /// - `same` and stride 1: output will be of same size as input,
        ///    regardless of filter size.
        /// - `valid`: output will be smaller than input if filter is larger
        ///    than 1x1.
        /// - For more info, see this guide:
        ///  [https://www.tensorflow.org/api_guides/python/nn#Convolution](
        ///       https://www.tensorflow.org/api_guides/python/nn#Convolution)</param>
        /// <param name="dimRoundingMode">The rounding mode used when computing output
        ///  dimensions if pad is a number. If none is provided, it will not round
        ///  and error if the output is of fractional size.</param>
        /// <param name="padvalue"></param>
        /// <returns></returns>
        public static Tensor maxPool(this Tensor x, int[] filterSize, int[] strides, PadType pad,
            roundingMode dimRoundingMode = roundingMode.none, Nullable<int> padvalue = null)
        {
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

            var convInfo = Util.computePool2DInfo(
        x4D.Shape, filterSize, strides, pad, dimRoundingMode, ConvDataFormat.channelsLast, padvalue);


            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    var y4D = s[0];
                    return maxPoolBackprop(dy, x4D, y4D, filterSize, strides, pad, dimRoundingMode, padvalue);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return saved( bk.maxPool(x4D, convInfo));
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x4D);
            var res = e.runKernel(f, inputs, grad);
            if (reshapedTo4D)
            {
                return res.as3D(res.Shape[1], res.Shape[2], res.Shape[3]);
            }
            return res;
        }

        /// <summary>
        /// Computes the backprop of a max pool.
        /// </summary>
        /// <param name="dy">The dy error, of rank 4 or rank 3 of shape
        /// [batchSize, height, width, channels]. If rank 3, batch of 1 is assumed.</param>
        /// <param name="input">The original input image, of rank 4, of shape
        /// [batchSize, height, width, channels].</param>
        /// <param name="output ">The original output image, of rank 4, of shape
        /// [batchSize, outHeight, outWidth, channels].</param>
        /// <param name="filterSize">The filter size, a tuple [filterHeight, filterWidth].</param>
        /// <param name="strides">The strides of the pooling: [strideHeight, strideWidth].</param>
        /// <param name="pad">A string from: 'same', 'valid'. The type of padding algorithm used in the forward prop of the op.</param>
        /// <param name="dimRoundingMode">A string from: 'ceil', 'round', 'floor'. The
        /// rounding mode used when computing output dimensions if pad is a
        /// number. If none is provided, it will not round and error if the output
        /// is of fractional size.</param>
        /// <param name="padvalue"></param>
        /// <returns></returns>
        private static Tensor maxPoolBackprop(Tensor dy, Tensor input, Tensor output, int[] filterSize,
            int[] strides, PadType pad, roundingMode dimRoundingMode, int? padvalue)
        {


            Tensor input4D = null;
            Tensor dy4D = null;


            var convInfo = Util.computePool2DInfo(
                input4D.Shape, filterSize, strides, pad, dimRoundingMode, ConvDataFormat.channelsLast, padvalue);

            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.maxPoolBackprop(dy, input, output, convInfo);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("dy4D", dy4D);
            inputs.Add("input4D", input4D);
            var res = e.runKernel(f, inputs);

            return res;
        }


        /// <summary>
        /// Computes the 2D average pooling of an image.
        /// </summary>
        /// <param name="x">The input tensor, of rank 4 or rank 3 of shape
        ///  `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.</param>
        /// <param name="filterSize">The filter size, a tuple `[filterHeight, filterWidth]`.</param>
        /// <param name="strides"> The strides of the pooling: `[strideHeight, strideWidth]`.</param>
        /// <param name="pad">The type of padding algorithm:
        /// - `same` and stride 1: output will be of same size as input,
        ///    regardless of filter size.
        /// - `valid`: output will be smaller than input if filter is larger
        ///    than 1x1.
        /// - For more info, see this guide:
        ///  [https://www.tensorflow.org/api_guides/python/nn#Convolution](
        ///      https://www.tensorflow.org/api_guides/python/nn#Convolution)</param>
        /// <param name="dimRoundingMode">The rounding mode used when computing output
        /// dimensions if pad is a number. If none is provided, it will not round
        /// and error if the output is of fractional size.</param>
        /// <param name="padvalue"></param>
        /// <returns></returns>
        public static Tensor avgPool(this Tensor x, int[] filterSize, int[] strides, PadType pad,
            roundingMode dimRoundingMode = roundingMode.none, Nullable<int> padvalue = null)
        {
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

            var convInfo = Util.computePool2DInfo(
        x4D.Shape, filterSize, strides, pad, dimRoundingMode, ConvDataFormat.channelsLast, padvalue);


            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () =>
                {
                    return avgPoolBackprop(dy, x4D, filterSize, strides, pad, padvalue);

                });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.avgPool(x4D, convInfo);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x4D);
            var res = e.runKernel(f, inputs, grad);
            if (reshapedTo4D)
            {
                return res.as3D(res.Shape[1], res.Shape[2], res.Shape[3]);
            }
            return res as Tensor;
        }

        /// <summary>
        ///  Computes the backprop of an avg pool.
        /// </summary>
        /// <param name="dy">The dy error, of rank 4 or rank 3 of shape
        ///  [batchSize, height, width, channels]. If rank 3, batch of 1 is assumed</param>
        /// <param name="input">The input image, of rank 4 or rank 3 of shape
        /// [batchSize, height, width, channels]. If rank 3, batch of 1 is assumed</param>
        /// <param name="filterSize">The filter size, a tuple [filterHeight, filterWidth].</param>
        /// <param name="strides">The strides of the pooling: [strideHeight, strideWidth].</param>
        /// <param name="pad"> A string from: 'same', 'valid'. The type of padding algorithm used in the forward prop of the op.</param>
        /// <param name="padvalue"></param>
        /// <returns></returns>
        private static Tensor avgPoolBackprop(Tensor dy, Tensor input, int[] filterSize,
            int[] strides, PadType pad, int? padvalue)
        {


            Tensor input4D = null;
            Tensor dy4D = null;
            var reshapedTo4D = false;
            if (input.Rank == 3)
            {
                reshapedTo4D = true;
                input4D = input.as4D(1, input.Shape[0], input.Shape[1], input.Shape[2]);
                dy4D = dy.as4D(1, dy.Shape[0], dy.Shape[1], dy.Shape[2]);
            }
            else
            {
                input4D = input as Tensor;
                dy4D = dy as Tensor;
            }


            var convInfo = Util.computePool2DInfo(
                input4D.Shape, filterSize, strides, pad, roundingMode.none, ConvDataFormat.channelsLast, padvalue);

            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.avgPoolBackprop(dy4D, input4D, convInfo);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("dy4D", dy4D);
            inputs.Add("input4D", input4D);
            var res = e.runKernel(f, inputs);
            if (reshapedTo4D)
            {
                return res.as3D(res.Shape[1], res.Shape[2], res.Shape[3]);
            }
            return res;
        }


    }
}
