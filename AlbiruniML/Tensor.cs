
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ops = AlbiruniML.Ops;
namespace AlbiruniML
{

    public class TensorData
    {
        public WeakReference dataId { get; set; }
        public float[] values { get; set; }
        public TensorData(WeakReference dataid, float[] values)
        {
            this.dataId = dataid;
            this.values = values;
        }
        public TensorData(WeakReference dataid)
        {
            this.dataId = dataid;
        }
        public TensorData(float[] values)
        {
            this.values = values;
        }
    }
    
    /// <summary>
    /// A mutable object, similar to `Tensor`, that allows users to set values
    /// at locations before converting to an immutable `Tenso
    /// </summary>
    public class TensorBuffer
    {
        public int Size { get; set; }
        public int[] shape { get; set; }
        public float[] values { get; set; }
        public int[] strides { get; set; } 

        public int rank
        {
            get
            {
                return this.shape.Length;
            }
        }
        public TensorBuffer(int[] shape, float[] values)
        { 
            if (values != null)
            {
                var n = values.Length;
                var size = Util.SizeFromShape(shape);
                if (n != size)
                {
                    throw new ArgumentException
                        ("Length of values " + n.ToString() +
                        " does not match the size inferred by the shape " + size.ToString());
                }
            }
            this.shape = new List<int>( shape ).ToArray();
            this.strides = ComputeStrides(shape);
            this.Size = Util.SizeFromShape(shape);
            this.values = values == null ?
                new float[Size] : values;

        }
        public void Set(float value, params int[] locs)
        {
            if (locs.Length == 0)
            {
                locs = new int[] { 0 };

            }
            if (locs.Length != this.rank)
            {
                throw new ArgumentException("The number of provided coordinates ("
                    + locs.Length.ToString()
                    + ") must  match the rank (" + this.rank.ToString() + ")");
            }
            var index = this.locToIndex(locs);
            this.values[index]=value;

        }

        public int locToIndex(int[] locs)
        {
            if (this.rank == 0)
            {
                return 0;
            }
            else if (this.rank == 1)
            {
                return locs[0];
            }
            var index = locs[locs.Length - 1];
            for (int i = 0; i < locs.Length - 1; ++i)
            {
                index += this.strides[i] * locs[i];
            }
            return index;
        }

        public int[] indexToLoc(int index)
        {
            if (this.rank == 0)
            {
                return new int[0];
            }
            else if (this.rank == 1)
            {
                return new int[1] { index };
            }

            int[] locs = new int[this.shape.Length];
            for (var i = 0; i < locs.Length - 1; ++i)
            {
                locs[i] = (int)Math.Floor(index / (float)this.strides[i]);
                index -= locs[i] * this.strides[i];
            }
            locs[locs.Length - 1] = index;
            return locs;
        }

        public float Get(params int[] locs)
        {
            if (locs.Length == 0)
            {
                locs = new int[1] { 0 };
            }
            var index = locs[locs.Length - 1];
            for (int i = 0; i < locs.Length - 1; ++i)
            {
                index += this.strides[i] * locs[i];
            }

            return this.values[index] ;
        }

        int[] ComputeStrides(int[] shape)
        {
            if (shape.Length < 2)
            {
                return new int[0];
            }
            var strid = new int[shape.Length - 1];

            strid[shape.Length - 2] = shape[shape.Length - 1];
            for (int i = shape.Length - 3; i >= 0; --i)
            {
                strid[i] = strid[i + 1] * shape[i + 1];
            }

            return strid;
        }

        public Tensor toTensor()
        {
            return Tensor.Make(this.shape, new TensorData(this.values));
        }
    }
    /// <summary>
    /// A `Tensor` object represents an immutable, multidimensional array of numbers
    /// that has a shape .
    /// </summary>
    public class Tensor
    {
        private static int nextId = 0;
        public int id { get; set; } 

        public WeakReference dataId { get; set; }
        public int[] Shape { get; set; }
        public int Size { get; set; }
        public int[] Strides { get; set; }
        public int Rank { get { return Shape.Length; } }
         

        public Tensor()
        {

        }
        public Tensor(int[] shape,
            float[] values = null, WeakReference dataId = null)
        {
            this.Size = Util.SizeFromShape(shape);

            this.Shape =new List<int>( shape ).ToArray();
            
            this.Strides = ComputeStrides(shape);
            this.dataId = dataId != null ? dataId : new WeakReference(values);
            this.id = Tensor.nextId++;
            ENV.engine.registerTensor(this);
            if (values != null)
            {
                ENV.engine.write(this.dataId, values);
            }
        }
        public float[] dataSync()
        {
            return ENV.engine.readSync(this.dataId);
        }
        public int[] ComputeStrides(int[] shape)
        {
            if (shape.Length < 2)
            {
                return new int[0];
            }
            var strid = new int[shape.Length - 1];

            strid[shape.Length - 2] = shape[shape.Length - 1];
            for (int i = shape.Length - 3; i >= 0; --i)
            {
                strid[i] = strid[i + 1] * shape[i + 1];
            }

            return strid;
        }

        public int locToIndex(int[] locs)
        {
            if (this.Rank == 0)
            {
                return 0;
            }
            else if (this.Rank == 1)
            {
                return locs[0];
            }
            var index = locs[locs.Length - 1];
            for (int i = 0; i < locs.Length - 1; ++i)
            {
                index += this.Strides[i] * locs[i];
            }
            return index;
        }

        public int[] indexToLoc(int index)
        {
            if (this.Rank == 0)
            {
                return new int[0];
            }
            else if (this.Rank == 1)
            {
                return new int[1] { index };
            }

            int[] locs = new int[this.Shape.Length];
            for (var i = 0; i < locs.Length - 1; ++i)
            {
                locs[i] = (int)Math.Floor(index / (float)this.Strides[i]);
                index -= locs[i] * this.Strides[i];
            }
            locs[locs.Length - 1] = index;
            return locs;
        }

        public float Get(params int[] locs)
        {
            if (locs.Length == 0)
            {
                locs = new int[1] { 0 };
            }
            var index = locs[locs.Length - 1];
            for (int i = 0; i < locs.Length - 1; ++i)
            {
                index += this.Strides[i] * locs[i];
            }

            return this.dataSync()[index];// as float;// this.Values[index];
        }

        public TensorBuffer buffer()
        {
            return ops.buffer(this.Shape, this.dataSync());
        }

        public int SizeFromShape(int[] shape)
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
        public bool IsScalar()
        {
            return this.Shape.Length == 0;
        }


        public static Tensor Make(int[] shape, TensorData data)
        {
            return new Tensor(shape, data.values, data.dataId);
        }


        public void dispose()
        {
           

            if (this.isDisposed())
            {
                return;
            }
            ENV.engine.disposeTensor(this);
            this.isDisposedInternal = true;
        }
        private bool isDisposedInternal = false;

        public bool isDisposed()
        {
            return this.isDisposedInternal;
        }
        #region Ops

        public Tensor as2D(int rows, int columns)
        {
            return ops.reshape(this, new int[] { rows, columns });
            // return this.reshape<Rank.R3>([rows, columns, depth]);
        }

        public Tensor as3D(int rows, int columns, int depth)
        {
            return ops.reshape(this, new int[] { rows, columns, depth });
            // return this.reshape<Rank.R3>([rows, columns, depth]);
        }

        public Tensor as4D(int rows, int columns, int depth, int depth2)
        {
            return ops.reshape(this, new int[] { rows, columns, depth, depth2 });
            // return this.reshape<Rank.R3>([rows, columns, depth]);
        }


        public Tensor as1D()
        {
            return ops.reshape(this, new int[] { Size });
            // return this.reshape<Rank.R3>([rows, columns, depth]);
        }
        

        public Tensor asScalar()
        {
            return ops.reshape(this, new int[] { 1 });
            // return this.reshape<Rank.R3>([rows, columns, depth]);
        }
         
        public Tensor reshapeAs(Tensor x)
        {
            return this.reshape(x.Shape);
            // return this.reshape<Rank.R3>([rows, columns, depth]);
        }

         
        public Variable variable(bool trainable = true, string name = null)
        {
            return new Variable(this, trainable, name);
        }


         
        public Tensor softmaxCrossEntropy(Tensor labels, int dim = -1)
        {
            return ops.loss.softmaxCrossEntropy(labels, this, dim);
        }



        public Tensor clone()
        {
            return ops.clone(this);
        }
         

        #endregion
         
        #region Conversion
     

        public static implicit operator double(Tensor m)
        {
            float[] a = m.dataSync();
            return (double)a[0];
        }
        public static explicit operator Tensor(double m)
        {
            return m.ToTensor();
        }

        public static implicit operator float(Tensor m)
        {
            float[] a = m.dataSync();
            return a[0];
        }
        public static explicit operator Tensor(float m)
        {
            return m.ToTensor();
        }

        public static implicit operator int[](Tensor m)
        {
            float[] a = m.dataSync();
            int[] r = new int[a.Length];
            for (int i = 0; i < r.Length; i++)
            {
                r[i] = (int)a[i];
            }
            return r;
        }
        public static explicit operator Tensor(int[] m)
        {
            return m.ToTensor();
        }

        public static implicit operator float[](Tensor m)
        {
            float[] a = m.dataSync();
            return a;
        }
        public static explicit operator Tensor(float[] m)
        {
            return m.ToTensor();
        }


        //public static implicit operator float[][](Tensor m)
        //{
        //    float[] a = m.dataSync();
        //    return a;
        //}
        //public static explicit operator Tensor(float[][] m)
        //{
        //    return m.ToTensor();
        //}

        public static implicit operator double[](Tensor m)
        {
            float[] a = m.dataSync();
            double[] r = new double[a.Length];
            for (int i = 0; i < r.Length; i++)
            {
                r[i] = (double)a[i];
            }
            return r;
        }
        public static explicit operator Tensor(double[] m)
        {
            return m.ToTensor();
        }

        #endregion


        #region Operators

        public static Tensor operator +(Tensor b, Tensor c)
        {
            return b.add(c);
        }
        public static Tensor operator +(int b, Tensor c)
        {
            return b.ToTensor().add(c);
        }
        public static Tensor operator +(Tensor b, int c)
        {
            return b.add(c.ToTensor());
        }
        public static Tensor operator +(float b, Tensor c)
        {
            return b.ToTensor().add(c);
        }
        public static Tensor operator +(Tensor b, float c)
        {
            return b.add(c.ToTensor());
        }
        public static Tensor operator +(double b, Tensor c)
        {
            return b.ToTensor().add(c);
        }
        public static Tensor operator +(Tensor b, double c)
        {
            return b.add(c.ToTensor());
        }


        public static Tensor operator -(Tensor b, Tensor c)
        {
            return b.sub(c);
        }
        public static Tensor operator -(int b, Tensor c)
        {
            return b.ToTensor().sub(c);
        }
        public static Tensor operator -(Tensor b, int c)
        {
            return b.sub(c.ToTensor());
        }
        public static Tensor operator -(float b, Tensor c)
        {
            return b.ToTensor().sub(c);
        }
        public static Tensor operator -(Tensor b, float c)
        {
            return b.sub(c.ToTensor());
        }
        public static Tensor operator -(double b, Tensor c)
        {
            return b.ToTensor().sub(c);
        }
        public static Tensor operator -(Tensor b, double c)
        {
            return b.sub(c.ToTensor());
        }

        public static Tensor operator /(Tensor b, Tensor c)
        {
            return b.div(c);
        }
        public static Tensor operator /(int b, Tensor c)
        {
            return b.ToTensor().div(c);
        }
        public static Tensor operator /(Tensor b, int c)
        {
            return b.div(c.ToTensor());
        }
        public static Tensor operator /(float b, Tensor c)
        {
            return b.ToTensor().div(c);
        }
        public static Tensor operator /(Tensor b, float c)
        {
            return b.div(c.ToTensor());
        }
        public static Tensor operator /(double b, Tensor c)
        {
            return b.ToTensor().div(c);
        }
        public static Tensor operator /(Tensor b, double c)
        {
            return b.div(c.ToTensor());
        }


        public static Tensor operator *(Tensor b, Tensor c)
        {
            return b.mul(c);
        }
        public static Tensor operator *(int b, Tensor c)
        {
            return b.ToTensor().mul(c);
        }
        public static Tensor operator *(Tensor b, int c)
        {
            return b.mul(c.ToTensor());
        }
        public static Tensor operator *(float b, Tensor c)
        {
            return b.ToTensor().mul(c);
        }
        public static Tensor operator *(Tensor b, float c)
        {
            return b.mul(c.ToTensor());
        }
        public static Tensor operator *(double b, Tensor c)
        {
            return b.ToTensor().mul(c);
        }
        public static Tensor operator *(Tensor b, double c)
        {
            return b.mul(c.ToTensor());
        }

        #endregion 

        public override string ToString()
        {

            return ((Shape)this.Shape).ToString();
        }

        public float[] ToArray()
        {
            return this.dataSync();
        }
        public double[] ToDoubleArray()
        {
            return ( double[])this    ;
        }
        public int[] ToIntArray()
        {
            return (int[])this;
        }
    }


     
     /// <summary>
    /// A mutable `Tensor`, useful for persisting state, e.g. for training.
     /// </summary>
    public class Variable : Tensor
    {
        private static int nextVarId = 0;
        public string Name { get; set; }
        public bool trainable { get; set; }

        public Variable(Tensor initialValue, bool trainable = true, string name = null)
            : base(initialValue.Shape,   null /* values */,
        initialValue.dataId)
        {
            this.Name = name;
            this.trainable = trainable;
            if (this.Name == null)
            {
                this.Name = Variable.nextVarId.ToString();
                Variable.nextVarId++;
            }
            ENV.engine.registerVariable(this);

        }

        public void assign(Tensor newValue)
        {
            ENV.engine.disposeTensor(this);
            this.dataId = newValue.dataId;

            ENV.engine.registerTensor(this);
        }
    }
}
