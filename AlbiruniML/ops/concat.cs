using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public static partial class Ops
    {
        public static Tensor concat1d(Tensor[] tensors, int axis = 0)
        {
            if (tensors.Where(p => p.Shape.Length > 1).Count() > 0)
            {
                throw new Exception("concat1d expects a rank-1 tensors");

            }

            return concat(tensors, axis);
        }
        public static Tensor concat2d(Tensor[] tensors, int axis = 0)
        {
            if (tensors.Where(p => p.Shape.Length > 2).Count() > 0)
            {
                throw new Exception("concat2d expects a rank-2 tensors");

            }

            return concat(tensors, axis);
        }
        public static Tensor concat3d(Tensor[] tensors, int axis = 0)
        {
            if (tensors.Where(p => p.Shape.Length > 3).Count() > 0)
            {
                throw new Exception("concat3d expects a rank-3 tensors");

            }

            return concat(tensors, axis);
        }
        public static Tensor concat4d(Tensor[] tensors, int axis = 0)
        {
            if (tensors.Where(p=>p.Shape.Length>4).Count()  > 0)
            {
                throw new Exception("concat4d expects a rank-4 tensors");
           
            }

            return concat(tensors, axis);
        }



        /// <summary>
        /// Concatenates a list of `Tensor`s along a given axis.
        ///
        /// The tensors ranks and types must match, and their sizes must match in all
        /// dimensions except `axis`.
        /// </summary>
        /// <param name="tensors">A list of tensors to concatenate.</param>
        /// <param name="axis">The axis to concate along. Defaults to 0 (the first dim).</param>
        /// <returns></returns>
        public static Tensor concat(Tensor[] tensors, int axis = 0)
        {
            var result = tensors[0];
            var axes = Util.parseAxisParam(new int[] { axis }, result.Shape);

            for (var i = 1; i < tensors.Length; ++i)
            {
                result = concat2Tensors(result, tensors[i], axes[0]);
            }
            return result;
        }
        /// <summary>
        /// Concatenates a list of `Tensor`s along a given axis.
        ///
        /// The tensors ranks and types must match, and their sizes must match in all
        /// dimensions except `axis`.
        /// </summary>
        /// <param name="tensors">A list of tensors to concatenate.</param>
        /// <param name="axis">The axis to concate along. Defaults to 0 (the first dim).</param>
        /// <returns></returns>
        public static Tensor concat(this Tensor tensor ,Tensor tensor2, int axis = 0)
        { 
            var axes = Util.parseAxisParam(new int[] { axis }, tensor.Shape);

            return concat2Tensors(tensor, tensor2, axes[0]);
        }
        private static Tensor concat2Tensors(  Tensor a, Tensor b, int axis)
        {
            var outShape = Util.computeOutShape(a.Shape, b.Shape, axis);

            var fs = new ArraySegment<int>(a.Shape, axis, a.Shape.Length - axis);
            var fs2 = new ArraySegment<int>(b.Shape, axis, b.Shape.Length - axis);

            var a2D = a.as2D(-1,
                Util.SizeFromShape(

               fs.ToArray()
            
                
                ));
            var b2D = b.as2D(-1,
                Util.SizeFromShape(fs2.ToArray()));

            var slices = Util.computeGradientSliceShapes(a2D.Shape, b2D.Shape);

            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("a", () => { return dy.slice(slices.aBegin, slices.aSize); });
                g.gradient.Add("b", () => { return dy.slice(slices.bBegin, slices.bSize); });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return
                    bk.concat(a2D, b2D);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("a", a2D);
            inputs.Add("b", b2D);
            var res = e.runKernel(f, inputs, grad);
            return res.reshape(outShape);
        }


    }
}
