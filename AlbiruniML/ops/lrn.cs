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
        /// Normalizes the activation of a local neighborhood across or within
        /// channels.
        /// </summary>
        /// <param name="x">The input tensor. The 4-D input tensor is treated as a 3-D array
        ///    of 1D vectors (along the last dimension), and each vector is
        ///    normalized independently.</param>
        /// <param name="depthRadius">The number of adjacent channels or spatial locations of the
        ///    1D normalization window. In Tensorflow this param is called
        ///    'depth_radius' because only 'acrossChannels' mode is supported.</param>
        /// <param name="bias">A constant bias term for the basis.</param>
        /// <param name="alpha">A scale factor, usually positive.</param>
        /// <param name="beta">An exponent.</param> 
        /// <returns></returns>
        public static Tensor localResponseNormalization(this Tensor x,
            float depthRadius = 5, float bias = 1, float alpha = 1, float beta = 0.5f)
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


            Engine e = ENV.engine;

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x4D", () =>
                {

                    ForwardFunc fgrad = (IBackend bk, Func<Tensor, Tensor> saved) =>
                    {
                        var outputImage = s[0];
                        return bk.LRNGrad(
dy, x4D, outputImage , depthRadius, bias, alpha, beta);
                    };  
                    return e.runKernel(fgrad, new Dictionary<string, Tensor>());
                
                
                });
                return g;
            };
             
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return saved( bk.localResponseNormalization4D(
            x4D, depthRadius, bias, alpha, beta));
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x4D", x4D);
            var res = e.runKernel(f, inputs);

            if (reshapedTo4D)
            {
                return res.as3D(res.Shape[1], res.Shape[2], res.Shape[3]);
            }
            else
            {
                return res;
            }
        }

    }
}
