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
        /// Extracts a 1D slice from 1D array starting at coordinates `begin` and is
        /// of length `size`. See `slice` for details.
        /// </summary>   
        public static Tensor slice1d(this Tensor x, int  begin, int  size )
        {
            if (x.Rank != 1)
            {
                throw new Exception("slice1d expects a rank-1 tensor, but got a rank-" + x.Rank.ToString() + " tensor");
            }
            return slice(x, new int[]{begin}, new int[]{size});
        }
        /// <summary>
        /// Extracts a 2D slice from 2D array starting at coordinates `begin` and is
        /// of length `size`. See `slice` for details.
        /// </summary>   
        public static Tensor slice2d(this Tensor x, int[] begin, int[] size)
        {
            if (x.Rank != 2)
            {
                throw new Exception("slice2d expects a rank-2 tensor, but got a rank-" + x.Rank.ToString() + " tensor");
            }
            return slice(x, begin, size);
        }
        /// <summary>
        /// Extracts a 3D slice from 3D array starting at coordinates `begin` and is
        /// of length `size`. See `slice` for details.
        /// </summary>   
        public static Tensor slice3d(this Tensor x, int[] begin, int[] size)
        {
            if (x.Rank != 3)
            {
                throw new Exception("slice3d expects a rank-3 tensor, but got a rank-" + x.Rank.ToString() + " tensor");
            }
            return slice(x, begin, size);
        }

        /// <summary>
        /// Extracts a 4D slice from 4D array starting at coordinates `begin` and is
        /// of length `size`. See `slice` for details.
        /// </summary>   
        public static Tensor slice4d(this Tensor x, int[] begin, int[] size)
        {
            if (x.Rank != 4)
            {
                throw new Exception("slice4d expects a rank-4 tensor, but got a rank-" + x.Rank.ToString() + " tensor");
            }
            return slice(x, begin, size);
        }


        /// <summary>
        ///Extracts a slice from a `Tensor` starting at coordinates `begin`
        ///and is of size `size`.
        /// </summary> 
        /// <param name="x">The input `Tensor` to slice from.</param>
        /// <param name="begin">
        /// The coordinates to start the slice from. The length can be
        /// less than the rank of x - the rest of the axes will have implicit 0 as
        /// start. Can also be a single number, in which case it specifies the
        /// first axis.
        /// </param>
        /// <param name="size">
        /// The size of the slice. The length can be less than the rank of
        ///x - the rest of the axes will have implicit -1. A value of -1 requests
        ///the rest of the dimensions in the axis. Can also be a single number,
        ///in which case it specifies the size of the first axis.
        /// </param>
        /// <returns></returns>
        public static Tensor slice(this Tensor x, int[] begin, int[] size = null) 
        {

            if (x.Rank == 0)
            {
                throw new Exception("Slicing scalar is not possible");
            }
            // The following logic allows for more ergonomic calls.
            int[] begin_ = new int[x.Rank];
            if (begin.Length < x.Rank)
            {
               Array.Copy(begin,begin_,begin.Length);
            }
            else
            {
                begin_ = begin;
            }

            int[] size_ = new int[x.Rank];

            if (size==null)
            {
                for (int i = 0; i < size_.Length; i++)
                {
                    size_[i] = -1;
                }
            }
            else if (size.Length < x.Rank)
            {
                for (int i = 0; i < size_.Length; i++)
                {
                    size_[i] = -1;
                }
                Array.Copy(size, size_, size.Length); 
            }
            else
            {
                size_ = size;
            }
            size_ = size_.Select((d, i) =>
            {
                if (d >= 0)
                {
                    return d;
                }
                else
                {
                    if (d == -1)
                    {
                        throw new Exception("Bad value in size");
                    }
                    // util.assert(d === -1, 'Bad value in size');
                    return x.Shape[i] - begin_[i];
                }
            }).ToArray();
            var inputShape = x.Shape;
            Func<Tensor, List<Tensor>, NamedGradientMap> grad = (Tensor dy, List<Tensor> s) =>
            {
                // Create an Nx2 padding where the first column represents how many
                // zeros are prepended (at start) for each dimension, and the second
                // column indicates how many zeros are appended (at end).

                // The number of zeros to append is the shape of the input
                // elementwise-subtracted by both the begin vector and sizes vector.
                List<int[]> paddings = new List<int[]>();
                for (var i = 0; i < dy.Rank; i++)
                {
                    paddings.Add(new int[] { begin_[i], inputShape[i] - begin_[i] - size_[i] });
                }
                NamedGradientMap g = new NamedGradientMap();
                g.gradient = new Dictionary<string, Func<Tensor>>();
                g.gradient.Add("x", () => { return dy.pad(paddings.ToArray()); });
                return g;
            };
            Engine e = ENV.engine;
            ForwardFunc f = (IBackend bk, Func<Tensor, Tensor> saved) =>
            {
                return bk.slice(x, begin_, size_);
            };

            var inputs = new Dictionary<string, Tensor>();
            inputs.Add("x", x);
            return e.runKernel(f, inputs, grad) ;
        }
     
    }
}
