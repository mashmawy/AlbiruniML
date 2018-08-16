using System;
using System.Collections.Generic;
using System.Linq;
using System.Data.Common;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using ops = AlbiruniML.Ops;
namespace AlbiruniML
{

    public partial class Backend_CPU : IBackend
    {
        public int blockSize = 48;
        public Dictionary<WeakReference, float[]> data = new Dictionary<WeakReference, float[]>();
        public static double SELU_SCALEALPHA = 1.7580993408473768599402175208123;
        public static double SELU_SCALE = 1.0507009873554804934193349852946;
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

        public MemoryInfo memory()
        {
            return new MemoryInfo() { unreliable = true };
        }
        public Tensor slice(Tensor x, int[] begin, int[] size)
        {
            var buffer = ops.buffer(size);

            for (var i = 0; i < buffer.Size; ++i)
            {
                var loc = buffer.indexToLoc(i);
                var xLoc = loc.Select((idx, j) => idx + begin[j]).ToArray();
                buffer.Set(x.Get(xLoc), loc);
            }
            return buffer.toTensor();
        }

        public Tensor stridedSlice(Tensor x, int[] begin, int[] end, int[] strides, int beginMask, int endMask)
        {
            var ssi = Util.getStridedSlicedInfo(x.Shape, begin, end, strides, beginMask, endMask);
            var beginIndex = ssi[0];
            var size = ssi[1];

            if (size.Any(axis => axis == 0))
            {
                return ops.tensor(new float[Util.SizeFromShape(size)], size);
            }

            var buffer = ops.buffer(size);

            for (var i = 0; i < buffer.Size; i++)
            {
                var loc = buffer.indexToLoc(i);

                var newLoc = new int[loc.Length];
                for (var j = 0; j < newLoc.Length; j++)
                {
                    newLoc[j] = loc[j] * strides[j] + beginIndex[j];
                }
                buffer.Set(x.Get(newLoc), loc);
            }

            return buffer.toTensor();
        }

        public Tensor reverse(Tensor x, int[] axis)
        {
            var buffer = ops.buffer(x.Shape);
            var xBuffer = x;
            for (int i = 0; i < buffer.Size; i++)
            {
                var outLoc = buffer.indexToLoc(i);
                var inLoc = new List<int>(outLoc).ToArray();
                foreach (var ax in axis)
                {
                    inLoc[ax] = x.Shape[ax] - 1 - inLoc[ax];
                }
                buffer.Set(xBuffer.Get(inLoc), outLoc);
            }
            return buffer.toTensor();
        }

        public Tensor concat(Tensor a, Tensor b)
        {
            var outShape = Util.computeOutShape(
         a.Shape, b.Shape, 1);
            var buffer = ops.buffer(outShape);

            if (a.Shape[0] == 1 && b.Shape[0] == 1)
            {
                var aVals = a.dataSync();
                var bVals = b.dataSync();
                var vals = buffer.values;

                for (int i = 0; i < a.Size; i++)
                {
                    vals[i] = aVals[i];
                }
                for (int i = a.Size; i < b.Size + a.Size; i++)
                {
                    vals[i] = bVals[i - a.Size];
                }
                return buffer.toTensor();
            }



            for (var i = 0; i < outShape[0]; ++i)
            {
                for (var j = 0; j < a.Shape[1]; ++j)
                {
                    buffer.Set(a.Get(i, j), i, j);
                }
                for (var j = 0; j < b.Shape[1]; ++j)
                {
                    buffer.Set(b.Get(i, j), i, j + a.Shape[1]);
                }
            }

            return buffer.toTensor();
        }

        public Tensor neg(Tensor a)
        {
            return this.Multiply(ops.scalar(-1), a);
        }

        public Tensor Add(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (avalue, bvalue) =>
            {
                return avalue + bvalue;
            });

        }

        public Tensor Subtract(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (avalue, bvalue) =>
            {
                return avalue - bvalue;
            });

        }

        public Tensor Pow(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (avalue, bvalue) =>
            {
                return (float)Math.Pow(avalue, bvalue);
            });

        }

        public Tensor Multiply(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (avalue, bvalue) =>
            {
                return avalue * bvalue;
            });
        }

        public Tensor Divide(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (avalue, bvalue) =>
            {
                return avalue / bvalue;
            });
        }

        public Tensor Sum(Tensor x, int[] axes)
        {
            var shapes = Util.computeOutAndReduceShapes(x.Shape, axes);
            var outShape = shapes.Item1;
            var reduceShape = shapes.Item2;
            var result = ops.zeros(outShape);
            int reduceSize = SizeFromShape(reduceShape);
            var vals = result.dataSync();
            var aVals = x.dataSync();

            for (var i = 0; i < vals.Length; ++i)
            {
                var offset = i * reduceSize;
                var sum = 0.0f;
                for (int j = 0; j < reduceSize; ++j)
                {

                    sum += aVals[offset + j];

                }
                vals[i] = sum;
            }

            return result;
        }

        public Tensor unsortedSegmentSum(Tensor x, Tensor segmentIds, int numSegments)
        {
            List<Tensor> res = new List<Tensor>();
            // Reshape the segment id's so that they can be broadcast with
            // x. The new shape should be [segmentIds.shape, 1, ..., 1]
            var numIters = x.Rank - segmentIds.Rank;
            for (var i = 0; i < numIters; ++i)
            {
                segmentIds = segmentIds.expandDims(i + 1);
            }

            for (var i = 0; i < numSegments; ++i)
            {
                var segmentId = ops.scalar(i);
                var mask = ops.equal(segmentId, segmentIds);
                var sum = mask.mul(x).sum(new int[] { 0 });
                res.Add(sum);
            }

            return ops.stack(res.ToArray());
        }

        public Tensor ArgMin(Tensor x, int[] axes)
        {
            var shapes = Util.computeOutAndReduceShapes(x.Shape, axes);
            var outShape = shapes.Item1;
            var reduceShape = shapes.Item2;
            Tensor result = ops.zeros(outShape);
            int reduceSize = SizeFromShape(reduceShape);
            var vals = result.dataSync() as float[];
            var aVals = x.dataSync();
            for (var i = 0; i < vals.Length; ++i)
            {
                int offset = i * reduceSize;
                var min = aVals[offset];
                var minIndex = 0;
                for (var j = 0; j < reduceSize; ++j)
                {
                    var value = aVals[offset + j];

                    if ((float)value < (float)min)
                    {
                        min = value;
                        minIndex = j;
                    }
                }
                vals[i] = minIndex;
            }
            return result;
        }

        public Tensor ArgMax(Tensor x, int[] axes)
        {
            var shapes = Util.computeOutAndReduceShapes(x.Shape, axes);
            var outShape = shapes.Item1;
            var reduceShape = shapes.Item2;
            Tensor result = ops.zeros(outShape);
            int reduceSize = SizeFromShape(reduceShape);
            var vals = result.dataSync() as float[];
            var aVals = x.dataSync();
            for (var i = 0; i < vals.Length; ++i)
            {
                int offset = i * reduceSize;
                var max = aVals[offset];
                int maxIndex = 0;
                for (int j = 0; j < reduceSize; ++j)
                {
                    var value = aVals[offset + j];

                    if (value > max)
                    {
                        max = value;
                        maxIndex = j;
                    }
                }
                vals[i] = maxIndex;
            }
            return result;
        }

        public Tensor cumsum(Tensor x, int axis, bool exclusive, bool reverse)
        {
            if (axis != x.Rank - 1)
            {
                throw new Exception(
                    "backend.cumsum in CPU expects an inner-most axis=" + (x.Rank - 1).ToString() +
                    " but got axis=" + axis.ToString());
            }

            var result = ops.zeros(x.Shape);
            var vals = result.dataSync();

            var aVals = x.dataSync();
            var finalDim = x.Shape[x.Rank - 1];
            Func<int, int, int> indexAdjuster;


            if (reverse)
            {
                indexAdjuster = (int i, int j) => { return i + finalDim - j - 1; };
            }

            else
            {
                indexAdjuster = (int i, int j) => { return i + j; };
            }
            for (var i = 0; i < aVals.Length; i += finalDim)
            {
                for (var j = 0; j < finalDim; j++)
                {
                    var idx = indexAdjuster(i, j);
                    if (j == 0)
                    {
                        vals[idx] = exclusive ? 0 : aVals[idx];
                    }
                    else
                    {
                        var prevIdx = indexAdjuster(i, j - 1);
                        vals[idx] = exclusive ? aVals[prevIdx] + vals[prevIdx] :
                        aVals[idx] + vals[prevIdx];
                    }
                }
            }
            return result;

        }

        public Tensor equal(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (aVal, bVal) =>
            {
                return (aVal == bVal) ? 1 : 0;
            });


        }

        public Tensor notEqual(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (aVal, bVal) =>
            {
                return (aVal != bVal) ? 1 : 0;
            });

        }

        public Tensor less(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (aVal, bVal) =>
            {
                return (aVal < bVal) ? 1 : 0;
            });


        }

        public Tensor lessEqual(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (aVal, bVal) =>
            {
                return (aVal <= bVal) ? 1 : 0;
            });
        }

        public Tensor greater(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (aVal, bVal) =>
            {
                return (aVal > bVal) ? 1 : 0;
            });

        }

        public Tensor greaterEqual(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (aVal, bVal) =>
            {
                return (aVal >= bVal) ? 1 : 0;
            });
        }

        public Tensor logicalNot(Tensor x)
        {

            var values = x.dataSync();
            int[] newValues = new int[values.Length];
            for (int i = 0; i < values.Length; ++i)
            {

                newValues[i] = values[i] == 1 ? 0 : 1;

            }
            return Tensor.Make(x.Shape, new TensorData(values));
        }

        public Tensor logicalAnd(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (aVal, bVal) =>
            {
                return (aVal == 1 && bVal == 1) ? 1 : 0;
            });

        }

        public Tensor logicalOr(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (aVal, bVal) =>
            {
                return (aVal == 1 || bVal == 1) ? 1 : 0;
            });

        }

        public Tensor logicalXor(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (aVal, bVal) =>
            {
                return (aVal != bVal) ? 1 : 0;
            });

        }

        public Tensor where(Tensor condition, Tensor a, Tensor b)
        {
            var values = condition.dataSync();
            var aValues = a.dataSync();
            var bValues = b.dataSync();
            var result = ops.zeros(a.Shape);
            var newValues = result.dataSync();

            var index = 0;

            var offset = condition.Rank == 0 || condition.Rank > 1 || a.Rank == 1 ? 1 : a.Shape[1];
            for (int i = 0; i < values.Length; i++)
            {
                for (int j = 0; j < offset; j++)
                {
                    if (values[i] == 1)
                    {
                        newValues[index++] = aValues[i];
                    }
                    else
                    {
                        newValues[index++] = bValues[i];

                    }
                }
            }
            return result;
        }

        public Tensor topKValues(Tensor x, int k)
        {
            return this.topK(x, k).Item1;
        }

        public Tensor topKIndices(Tensor x, int k)
        {
            return this.topK(x, k).Item2;
        }

        public Tuple<Tensor, Tensor> topK(Tensor x, int k)
        {
            var values = x.dataSync();
            var valuesAndIndices = new Stack<ValueIndex>();
            for (int i = 0; i < values.Length; i++)
            {
                valuesAndIndices.Push(new ValueIndex() { value = values[i], index = i });
            }

            var sorted = valuesAndIndices.OrderBy(p => p.value).ToArray();
            float[] topkValues = new float[k];
            float[] topkIndices = new float[k];
            for (var i = 0; i < k; i++)
            {
                topkValues[i] = sorted[i].value;
                topkIndices[i] = sorted[i].index;
            }
            Tensor t1 = ops.tensor1d(topkValues);
            Tensor t2 = ops.tensor1d(topkIndices);// Tensor.Make(new int[] { k }, topkIndices);
            return new Tuple<Tensor, Tensor>(t1, t2);

        }

        public Tensor min(Tensor x, int[] axes)
        {
            var shapes = Util.computeOutAndReduceShapes(x.Shape, axes);
            var outShape = shapes.Item1;
            var reduceShape = shapes.Item2;
            Tensor result = ops.zeros(outShape);
            int reduceSize = SizeFromShape(reduceShape);
            var vals = result.dataSync();
            var aVals = x.dataSync();
            for (int i = 0; i < vals.Length; ++i)
            {
                int offset = i * reduceSize;
                var min = aVals[0];
                for (var j = 0; j < reduceSize; ++j)
                {
                    var value = aVals[offset + j];
                    if (value < min)
                    {
                        min = value;
                    }
                }
                vals[i] = min;
            }
            return result;
        }

        public Tensor minimum(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (avalue, bvalue) =>
          {
              return (float)Math.Min(avalue, bvalue);
          });

        }

        public Tensor mod(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (aVal, bVal) =>
            {
                var rem = aVal % bVal;
                if ((aVal < 0 && bVal < 0) || (aVal >= 0 && bVal >= 0))
                {
                    return rem;
                }
                else
                {
                    return (rem + bVal) % bVal;
                }
            });
        }

        public Tensor max(Tensor x, int[] axes)
        {
            var shapes = Util.computeOutAndReduceShapes(x.Shape, axes);
            var outShape = shapes.Item1;
            var reduceShape = shapes.Item2;
            Tensor result = ops.zeros(outShape);
            int reduceSize = SizeFromShape(reduceShape);
            var vals = result.dataSync();
            var aVals = x.dataSync();
            for (int i = 0; i < vals.Length; ++i)
            {
                int offset = i * reduceSize;
                var max = aVals[offset];
                for (var j = 0; j < reduceSize; ++j)
                {
                    var value = aVals[offset + j];

                    if (value > max)
                    {
                        max = value;
                    }
                }
                vals[i] = max;
            }
            return result;
        }

        public Tensor maximum(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (avalue, bvalue) =>
            {
                return (float)Math.Max(avalue, bvalue);
            });
        }

        public Tensor all(Tensor x, int[] axes)
        {
            var dd = Util.computeOutAndReduceShapes(x.Shape, axes);
            var outShape = dd.Item1;
            var reduceShape = dd.Item2;
            var result = ops.zeros(outShape);
            var reduceSize = Util.SizeFromShape(reduceShape);
            var vals = result.dataSync();

            var aVals = x.dataSync();
            for (var i = 0; i < vals.Length; ++i)
            {
                var offset = i * reduceSize;
                var all = aVals[offset];
                for (var j = 0; j < reduceSize; ++j)
                {
                    var value = aVals[offset + j];
                    all = Convert.ToSingle(
                        Convert.ToBoolean(all) && Convert.ToBoolean(value));
                }
                vals[i] = all;
            }
            return result;
        }

        public Tensor any(Tensor x, int[] axes)
        {
            var dd = Util.computeOutAndReduceShapes(x.Shape, axes);
            var outShape = dd.Item1;
            var reduceShape = dd.Item2;
            var result = ops.zeros(outShape);
            var reduceSize = Util.SizeFromShape(reduceShape);
            var vals = result.dataSync();

            var aVals = x.dataSync();
            for (var i = 0; i < vals.Length; ++i)
            {
                var offset = i * reduceSize;
                var anyVal = aVals[offset];
                for (var j = 0; j < reduceSize; ++j)
                {
                    var value = aVals[offset + j];
                    anyVal = Convert.ToSingle(
                        Convert.ToBoolean(anyVal) || Convert.ToBoolean(value));
                }
                vals[i] = anyVal;
            }
            return result;
        }

        public Tensor squaredDifference(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (aVal, bVal) =>
            {
                var diff = aVal - bVal;
                return diff * diff;
            });

        }

        public Tensor ceil(Tensor x)
        {
            var values = x.dataSync();
            float[] newValues = new float[values.Length];
            for (int i = 0; i < values.Length; ++i)
            {
                newValues[i] = (float)Math.Ceiling(values[i]);
            }
            return Tensor.Make(x.Shape, new TensorData(newValues));
        }
        public Tensor floor(Tensor x)
        {
            var values = x.dataSync();
            float[] newValues = new float[values.Length];
            for (int i = 0; i < values.Length; ++i)
            {
                newValues[i] = (float)Math.Floor(values[i]);
            }
            return Tensor.Make(x.Shape, new TensorData(newValues));
        }

        public Tensor sign(Tensor x)
        {
            var values = x.dataSync();
            float[] newValues = new float[values.Length];
            for (int i = 0; i < values.Length; ++i)
            {
                newValues[i] = (float)Math.Floor(values[i]);
            }
            for (var i = 0; i < values.Length; ++i)
            {
                var val = values[i];
                if (val < 0)
                {
                    newValues[i] = -1;
                }
                else if (val > 0)
                {
                    newValues[i] = 1;
                }
                else
                {
                    newValues[i] = 0;
                }
            }
            return Tensor.Make(x.Shape, new TensorData(newValues));
        }

        public Tensor round(Tensor x)
        {
            var values = x.dataSync();
            float[] newValues = new float[values.Length];
            for (int i = 0; i < values.Length; ++i)
            {
                newValues[i] = (float)Math.Floor(values[i]);
            }
            for (var i = 0; i < values.Length; ++i)
            {
                // The algorithm is based on banker's rounding.
                var val = values[i];
                var baset = (float)Math.Floor(val);
                if (val - baset < 0.5f)
                {
                    newValues[i] = (float)Math.Floor(val);
                }
                else if (val - baset > 0.5)
                {
                    newValues[i] = (float)Math.Ceiling(val);
                }
                else
                {
                    if (baset % 2.0 == 0.0f)
                    {
                        newValues[i] = baset;
                    }
                    else
                    {
                        newValues[i] = baset + 1.0f;
                    }
                }
            }
            return Tensor.Make(x.Shape, new TensorData(newValues));
        }
        public Tensor exp(Tensor x)
        {
            var values = x.dataSync();
            float[] newValues = new float[values.Length];
            for (int i = 0; i < values.Length; ++i)
            {
                newValues[i] = (float)Math.Exp(values[i]);
            }
            return Tensor.Make(x.Shape, new TensorData(newValues));
        }
        public Tensor expm1(Tensor x)
        {
            var values = x.dataSync();
            float[] newValues = new float[values.Length];
            for (int i = 0; i < values.Length; ++i)
            {
                newValues[i] = (float)Math.Exp(values[i]) - 1;
            }
            return Tensor.Make(x.Shape, new TensorData(newValues));
        }

        public Tensor log(Tensor x)
        {
            var values = x.dataSync();
            float[] newValues = new float[values.Length];
            for (int i = 0; i < values.Length; ++i)
            {
                newValues[i] = (float)Math.Log(values[i]);
            }
            return Tensor.Make(x.Shape, new TensorData(newValues));
        }

        public Tensor log1p(Tensor x)
        {
            var values = x.dataSync();
            float[] newValues = new float[values.Length];
            for (int i = 0; i < values.Length; ++i)
            {
                newValues[i] = (float)Math.Log10(values[i]);
            }
            return Tensor.Make(x.Shape, new TensorData(newValues));
        }

        public Tensor sqrt(Tensor x)
        {
            var values = x.dataSync();
            float[] newValues = new float[values.Length];
            for (int i = 0; i < values.Length; ++i)
            {
                newValues[i] = (float)Math.Sqrt(values[i]);
            }
            return Tensor.Make(x.Shape, new TensorData(newValues));
        }

        public Tensor rsqrt(Tensor x)
        {
            var values = x.dataSync();
            float[] newValues = new float[values.Length];
            for (int i = 0; i < values.Length; ++i)
            {
                newValues[i] = 1.0f / (float)Math.Sqrt(values[i]);
            }
            return Tensor.Make(x.Shape, new TensorData(newValues));
        }
        public Tensor square(Tensor x)
        {
            var values = x.dataSync();
            float[] newValues = new float[values.Length];
            for (int i = 0; i < values.Length; ++i)
            {
                newValues[i] = values[i] * values[i];
            }
            return Tensor.Make(x.Shape, new TensorData(newValues));
        }
        public Tensor reciprocal(Tensor x)
        {
            var values = x.dataSync();
            float[] newValues = new float[values.Length];

            for (int i = 0; i < values.Length; ++i)
            {
                newValues[i] = 1 / values[i];
            }
            return Tensor.Make(x.Shape, new TensorData(newValues));
        }

        public Tensor relu(Tensor x)
        {
            Tensor res = ops.zeros(x.Shape);
            var resVals = res.dataSync();
            var inVals = x.dataSync();
            for (int i = 0; i < inVals.Length; ++i)
            {
                var val = inVals[i];
                var newval = Math.Max(0, val);
                resVals[i] = newval;
            }
            return res;
        }

        public Tensor elu(Tensor x)
        {
            float[] resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                if (v >= 0)
                {
                    resultValues[i] = v;
                }
                else
                {
                    resultValues[i] = ((float)Math.Exp(v) - 1);
                }
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }

        public Tensor eluDer(Tensor y, Tensor dy)
        {
            float[] resultValues = new float[y.Size];
            var values = y.dataSync();
            var dyValues = dy.dataSync();

            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                if (v >= 1)
                {
                    resultValues[i] = dyValues[i];
                }
                else
                {
                    resultValues[i] = dyValues[i] * (v + 1);
                }
            }
            return Tensor.Make(y.Shape, new TensorData(resultValues));
        }
        public Tensor selu(Tensor x)
        {
            // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
            // see: https://arxiv.org/abs/1706.02515
            var scaleAlpha = SELU_SCALEALPHA;
            var scale = SELU_SCALE;
            float[] resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                if (v >= 0)
                {
                    resultValues[i] = (float)(scale * v);
                }
                else
                {
                    resultValues[i] = (float)(scaleAlpha * ((float)Math.Exp(v) - 1f));
                }
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }

        public Tensor clip(Tensor x, float min, float max)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Min(max, (float)Math.Max(min, v));
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }
        public Tensor abs(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Abs(v);
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }
        public Tensor integer(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = (int)values[i];
                resultValues[i] = v;
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }
        public Tensor sigmoid(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = 1.0f / (1.0f + (float)Math.Exp(-v));
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }
        public Tensor softplus(Tensor x)
        {
            // mirrors the implementation of tf.nn.softplus: https://goo.gl/vkcvwX

            // epsilon is the difference between 1.0 and the next representable float.
            // For a single precision 32 bit float this should be 2^-23, see:
            // https://math.byu.edu/~schow/work/IEEEFloatingPoint.htm
            var epsilon = 1.1920928955078125e-7;
            var threshold = Math.Log(epsilon) + 2.0;

            var resultValues = new float[x.Size];
            var values = x.dataSync();

            for (var i = 0; i < values.Length; ++i)
            {
                // Value above which exp(x) may overflow, but softplus(x) == x
                // is within machine epsilon.

                var v = values[i];
                var tooLarge = v > -threshold;

                // Value below which exp(x) may underflow, but softplus(x) == exp(x)
                // is within machine epsilon.
                var tooSmall = v < threshold;

                var expX = Math.Exp(v);
                float result = 0.0f;

                if (tooSmall)
                {
                    result = (float)expX;
                }
                else if (tooLarge)
                {
                    result = v;
                }
                else
                {
                    result = (float)Math.Log(1.0f + expX);
                }
                resultValues[i] = result;
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }
        public Tensor sin(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Sin(v);
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }

        public Tensor cos(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Cos(v);
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }

        public Tensor tan(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Tan(v);
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }

        public Tensor asin(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Asin(v);
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }
        public Tensor acos(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Acos(v);
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }
        public Tensor atan(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Atan(v);
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }

        public Tensor atan2(Tensor a, Tensor b)
        {
            return this.broadcastedBinaryOp(a, b, (aVal, bVal) =>
             {
                 return (float)Math.Atan2(aVal, bVal);
             });

        }
        public Tensor sinh(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Sinh(v);
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }
        public Tensor cosh(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Cosh(v);
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }
        public Tensor tanh(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Tanh(v);
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }

        public Tensor asinh(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Log(v + Math.Sqrt((v * v) + 1));
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }

        public Tensor acosh(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Log(v + Math.Sqrt((v * v) - 1));
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }

        public Tensor atanh(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                resultValues[i] = (float)Math.Log((1.0f / v + 1.0f) / (1.0f / v - 1.0f)) / 2.0f;
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }

        public Tensor erf(Tensor x)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            var p = Util.ERF_P;
            var a1 = Util.ERF_A1;
            var a2 = Util.ERF_A2;
            var a3 = Util.ERF_A3;
            var a4 = Util.ERF_A4;
            var a5 = Util.ERF_A5;
            for (int i = 0; i < values.Length; ++i)
            {
                var v = values[i];
                var t = 1.0f / (1.0f + p * v);
                resultValues[i]
                    = (float)(
                    1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1)
                    * t * Math.Exp(-v * v));
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }

        public Tensor step(Tensor x, float alpha = 0)
        {
            var resultValues = new float[x.Size];
            var values = x.dataSync();
            for (int i = 0; i < values.Length; ++i)
            {
                var value = values[i];
                if (float.IsNaN(value))
                {
                    resultValues[i] = float.NaN;
                }
                else
                {
                    resultValues[i] = value > 0 ? 1 : alpha;
                }
            }
            return Tensor.Make(x.Shape, new TensorData(resultValues));
        }



        public Tensor broadcastedBinaryOp(Tensor a, Tensor b,
            Func<float, float, float> op)
        {
            var shapeA = a.Shape;
            var shapeB = b.Shape;

            int l = Math.Max(shapeA.Length, shapeB.Length);
            var newShape = new int[l];


            var ARank = shapeA.Length;
            int[] aBroadcastDims = new int[shapeA.Length];

            var BRank = shapeB.Length;
            int[] bBroadcastDims = new int[shapeB.Length];
            ComputeBroadcastShapes(shapeA, shapeB, l, newShape, ARank, aBroadcastDims, BRank, bBroadcastDims);

            var result = ops.buffer(newShape);
            var aValues = a.dataSync();
            var bValues = b.dataSync();
            ApplyBroadcastOp(a, b, op, aBroadcastDims, bBroadcastDims, result, aValues, bValues);
            return result.toTensor();
        }

        private static void ApplyBroadcastOp(Tensor a, Tensor b, Func<float, float, float> op, int[] aBroadcastDims, int[] bBroadcastDims, TensorBuffer result, float[] aValues, float[] bValues)
        {
            for (var i = 0; i < result.values.Length; ++i)
            {
                var resi = i;
                int aIndex = 0;
                int aShapeDim = 0;
                int bIndex = 0;
                int bShapeDim = 0;
                for (int j = 0; j < result.shape.Length; j++)
                {
                    int current = 0;
                    if (j == result.shape.Length - 1)
                    {
                        current = resi;
                    }
                    else
                    {
                        if (result.rank == 1)
                        {
                            current = resi;
                        }
                        else
                        {
                            current = (int)Math.Floor(resi / (float)result.strides[j]);
                            resi -= current * result.strides[j];
                        }
                    }

                    int locIndex = current;
                    if (j >= result.shape.Length - a.Rank)
                    {

                        int temp = locIndex;
                        if (aBroadcastDims[aShapeDim] > 0) temp = 0;
                        if (aValues.Length == 1)
                        {
                            aIndex = 0;
                        }
                        else if (a.Strides.Length == 0)
                        {
                            aIndex = temp;
                        }
                        else if (j < result.shape.Length - 1)
                        {
                            aIndex += a.Strides[aShapeDim] * temp;
                            aShapeDim++;
                        }
                        else if (j == result.shape.Length - 1)
                        {
                            aIndex += temp;
                        }
                    }
                    if (j >= result.shape.Length - b.Rank)
                    {

                        int temp = locIndex;
                        if (bBroadcastDims[bShapeDim] > 0) temp = 0;
                        if (bValues.Length == 1)
                        {
                            bIndex = 0;
                        }
                        else if (b.Strides.Length == 0)
                        {
                            bIndex = temp;
                        }
                        else if (j < result.shape.Length - 1)
                        {
                            bIndex += b.Strides[bShapeDim] * temp;
                            bShapeDim++;
                        }
                        else if (j == result.shape.Length - 1)
                        {
                            bIndex += temp;
                        }
                    }

                }

                var aval = aValues[aIndex];

                var bval = bValues[bIndex];

                var newval = op(aval, bval);
                result.values[i] = newval;

            }
        }

        private static void ComputeBroadcastShapes(int[] shapeA, int[] shapeB, int l, int[] newShape, int ARank, int[] aBroadcastDims, int BRank, int[] bBroadcastDims)
        {
            for (var i = 0; i < l; i++)
            {
                int az = (shapeA.Length - i - 1) < shapeA.Length && (shapeA.Length - i - 1) > -1 ? shapeA[shapeA.Length - i - 1] : 1;
                int bz = (shapeB.Length - i - 1) < shapeB.Length && (shapeB.Length - i - 1) > -1 ? shapeB[shapeB.Length - i - 1] : 1;
                if (az > 1 && bz > 1 && az != bz)
                {
                    throw new Exception("Operands could not be broadcast together");
                }
                newShape[newShape.Length - 1 - i] = Math.Max(az, bz);
                if (i < ARank)
                {
                    var dimA = ARank - 1 - i;
                    var azA = (dimA) > -1 && (dimA) < shapeA.Length ? shapeA[dimA] : 1;
                    var bzA = (newShape.Length - i - 1) > -1
                        && (newShape.Length - 1 - i) < newShape.Length ? newShape[newShape.Length - 1 - i] : 1;
                    if (bzA > 1 && azA == 1)
                    {
                        aBroadcastDims[(aBroadcastDims.Length - 1) - i] = 1;
                    }

                }

                if (i < BRank)
                {
                    var dimB = BRank - 1 - i;
                    var azB = (dimB) > -1 && (dimB) < shapeB.Length ? shapeB[dimB] : 1;
                    var bzB = (newShape.Length - i - 1) > -1
                        && (newShape.Length - 1 - i) < newShape.Length ? newShape[newShape.Length - 1 - i] : 1;
                    if (bzB > 1 && azB == 1)
                    {
                        bBroadcastDims[(bBroadcastDims.Length - 1) - i] = 1;
                    }

                }
            }
        }



        public Tensor conv2d(Tensor x, Tensor filter, Conv2DInfo convInfo)
        {
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var dilationHeight = convInfo.dilationHeight;
            var dilationWidth = convInfo.dilationWidth;
            var padLeft = convInfo.padInfo.left;
            var padTop = convInfo.padInfo.top;
            var y = ops.buffer(convInfo.outShape);

            var xVals = x.dataSync();
            var wVals = filter.dataSync();
            var yVals = y.values;

            for (var b = 0; b < convInfo.batchSize; ++b)
            {
                var xOffset1 = b * x.Strides[0];
                var yOffset1 = b * y.strides[0];
                for (var yR = 0; yR < convInfo.outHeight; ++yR)
                {
                    var yOffset2 = yOffset1 + yR * y.strides[1];
                    var xRCorner = yR * convInfo.strideHeight - padLeft;
                    for (var wR = 0; wR < filterHeight; wR++)
                    {
                        var xR = xRCorner + wR * dilationHeight;
                        if (xR < 0 || xR >= convInfo.inHeight)
                        {
                            continue;
                        }
                        var wOffset1 = wR * filter.Strides[0];
                        var xOffset2 = xOffset1 + xR * x.Strides[1];
                        for (var yC = 0; yC < convInfo.outWidth; ++yC)
                        {
                            var yOffset3 = yOffset2 + yC * convInfo.outChannels;
                            var xCCorner = yC * convInfo.strideWidth - padTop;
                            for (var wC = 0; wC < filterWidth; wC++)
                            {
                                var xC = xCCorner + wC * dilationWidth;
                                if (xC < 0 || xC >= convInfo.inWidth)
                                {
                                    continue;
                                }
                                var wOffset2 = wOffset1 + wC * filter.Strides[1];
                                var xOffset3 = xOffset2 + xC * convInfo.inChannels;
                                var wOffset3 = wOffset2;
                                for (var d1 = 0; d1 < convInfo.inChannels; ++d1)
                                {
                                    var xVal = xVals[xOffset3 + d1];
                                    for (var d2 = 0; d2 < convInfo.outChannels; ++d2)
                                    {
                                        yVals[yOffset3 + d2] += xVal * wVals[wOffset3 + d2];
                                    }
                                    wOffset3 += convInfo.outChannels;
                                }
                            }
                        }
                    }
                }
            }
            return y.toTensor();
        }

        public Tensor conv2dDerInput(Tensor dy, Tensor filter, Conv2DInfo convInfo)
        {
            var dx = ops.buffer(convInfo.inShape);
            var dxValues = dx.values as float[];
            int dxS0 = dx.strides[0];
            int dxS1 = dx.strides[1];
            int dxS2 = dx.strides[2];
            var dyValues = dy.dataSync() as float[];

            int dyS0 = dy.Strides[0];
            int dyS1 = dy.Strides[1];
            int dyS2 = dy.Strides[2];
            var fltValues = filter.dataSync() as float[];

            int fltS0 = filter.Strides[0];
            int fltS1 = filter.Strides[1];
            int fltS2 = filter.Strides[2];
            var batchSize = convInfo.batchSize;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var inChannels = convInfo.inChannels;
            var inHeight = convInfo.inHeight;
            var inWidth = convInfo.inWidth;
            var outChannels = convInfo.outChannels;
            var outHeight = convInfo.outHeight;
            var outWidth = convInfo.outWidth;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var topPad = filterHeight - 1 - convInfo.padInfo.top;
            var leftPad = filterWidth - 1 - convInfo.padInfo.left;

            for (var b = 0; b < batchSize; ++b)
            {
                for (var d1 = 0; d1 < inChannels; ++d1)
                {
                    for (var xR = 0; xR < inHeight; ++xR)
                    {
                        var xRCorner = xR - topPad;
                        var xRMin = (int)Math.Max(0, Math.Ceiling((double)xRCorner / strideHeight));
                        var yRMax = (int)
                            Math.Min(outHeight, (filterHeight + xRCorner) / strideHeight);

                        for (var xC = 0; xC < inWidth; ++xC)
                        {
                            var xCCorner = xC - leftPad;
                            var xCMin = (int)Math.Max(0, Math.Ceiling((double)xCCorner / strideWidth));
                            var yCMax = (int)
                                Math.Min(outWidth, (filterWidth + xCCorner) / strideWidth);

                            var dotProd = 0.0f;
                            for (var yR = xRMin; yR < yRMax; ++yR)
                            {
                                var wR = yR * strideHeight - xRCorner;

                                for (var yC = xCMin; yC < yCMax; ++yC)
                                {
                                    var wC = yC * strideWidth - xCCorner;
                                    var dyOffset = dyS0 * b + dyS1 * yR + dyS2 * yC;
                                    var fltOffset = fltS0 * (filterHeight - 1 - wR) +
                                        fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;

                                    for (var d2 = 0; d2 < outChannels; ++d2)
                                    {
                                        var pixel = dyValues[dyOffset + d2];
                                        var weight = fltValues[fltOffset + d2];
                                        dotProd += pixel * weight;
                                    }
                                }
                            }
                            dxValues[dxS0 * b + dxS1 * xR + dxS2 * xC + d1] = dotProd;
                        }
                    }
                }
            }
            return dx.toTensor();

        }

        public Tensor conv2dDerFilter(Tensor x, Tensor dy, Conv2DInfo convInfo)
        {
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var dW = ops.buffer(convInfo.filterShape);

            var leftPad = convInfo.padInfo.left;
            var topPad = convInfo.padInfo.top;

            for (var wR = 0; wR < filterHeight; ++wR)
            {
                var yRMin = (int)(float)Math.Max(0, (float)Math.Ceiling((topPad - wR) / (float)strideHeight));
                var yRMax = (float)Math.Min(
                    convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);

                for (var wC = 0; wC < filterWidth; ++wC)
                {
                    var yCMin = (int)(float)Math.Max(0, (float)Math.Ceiling((leftPad - wC) / (float)strideWidth));
                    var yCMax = (float)Math.Min(
                convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);

                    for (var d1 = 0; d1 < convInfo.inChannels; ++d1)
                    {
                        for (var d2 = 0; d2 < convInfo.outChannels; ++d2)
                        {
                            // Need to convolve.
                            var dotProd = 0.0f;
                            for (var b = 0; b < convInfo.batchSize; ++b)
                            {
                                for (var yR = yRMin; yR < yRMax; ++yR)
                                {
                                    var xR = wR + yR * strideHeight - topPad;
                                    for (var yC = yCMin; yC < yCMax; ++yC)
                                    {
                                        var xC = wC + yC * strideWidth - leftPad;
                                        dotProd += (float)x.Get(b, xR, xC, d1) * (float)dy.Get(b, yR, yC, d2);
                                    }
                                }
                            }
                            dW.Set(dotProd, wR, wC, d1, d2);
                        }
                    }
                }
            }
            return dW.toTensor();
        }
        public Tensor depthwiseConv2D(Tensor x, Tensor filter, Conv2DInfo convInfo)
        {
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var dilationHeight = convInfo.dilationHeight;
            var dilationWidth = convInfo.dilationWidth;
            var padLeft = convInfo.padInfo.left;
            var padTop = convInfo.padInfo.top;
            var chMul = convInfo.outChannels / convInfo.inChannels;
            var y = ops.buffer(convInfo.outShape);
            var xVals = x.dataSync();
            var wVals = filter.dataSync();
            var yVals = y.values;

            for (var b = 0; b < convInfo.batchSize; ++b)
            {
                var xOffset1 = b * x.Strides[0];
                var yOffset1 = b * y.strides[0];
                for (var yR = 0; yR < convInfo.outHeight; ++yR)
                {
                    var yOffset2 = yOffset1 + yR * y.strides[1];
                    var xRCorner = yR * convInfo.strideHeight - padLeft;
                    for (var wR = 0; wR < filterHeight; ++wR)
                    {
                        var xR = xRCorner + wR * dilationHeight;
                        if (xR < 0 || xR >= convInfo.inHeight)
                        {
                            continue;
                        }
                        var wOffset1 = wR * filter.Strides[0];
                        var xOffset2 = xOffset1 + xR * x.Strides[1];
                        for (var yC = 0; yC < convInfo.outWidth; ++yC)
                        {
                            var yOffset3 = yOffset2 + yC * y.strides[2];
                            var xCCorner = yC * convInfo.strideWidth - padTop;
                            for (var wC = 0; wC < filterWidth; ++wC)
                            {
                                var xC = xCCorner + wC * dilationWidth;
                                if (xC < 0 || xC >= convInfo.inWidth)
                                {
                                    continue;
                                }
                                var wOffset2 = wOffset1 + wC * filter.Strides[1];
                                var xOffset3 = xOffset2 + xC * convInfo.inChannels;
                                var yOffset4 = yOffset3;
                                var wOffset3 = wOffset2;
                                for (var d1 = 0; d1 < convInfo.inChannels; ++d1)
                                {
                                    var xVal = xVals[xOffset3 + d1];
                                    for (var q = 0; q < chMul; ++q)
                                    {
                                        yVals[yOffset4 + q] += xVal * wVals[wOffset3 + q];
                                    }
                                    yOffset4 += chMul;
                                    wOffset3 += chMul;
                                }
                            }
                        }
                    }
                }
            }

            return y.toTensor();

        }

        public Tensor depthwiseConv2DDerInput(Tensor dy, Tensor filter, Conv2DInfo convInfo)
        {
            var dx = ops.buffer(convInfo.inShape);
            var dxValues = dx.values;
            int dxS0 = dx.strides[0];
            int dxS1 = dx.strides[1];
            int dxS2 = dx.strides[2];
            var dyValues = dy.dataSync();

            int dyS0 = dy.Strides[0];
            int dyS1 = dy.Strides[1];
            int dyS2 = dy.Strides[2];
            var fltValues = filter.dataSync();

            int fltS0 = filter.Strides[0];
            int fltS1 = filter.Strides[1];
            int fltS2 = filter.Strides[2];
            var batchSize = convInfo.batchSize;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var inChannels = convInfo.inChannels;
            var inHeight = convInfo.inHeight;
            var inWidth = convInfo.inWidth;
            var outChannels = convInfo.outChannels;
            var outHeight = convInfo.outHeight;
            var outWidth = convInfo.outWidth;
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var topPad = filterHeight - 1 - convInfo.padInfo.top;
            var leftPad = filterWidth - 1 - convInfo.padInfo.left;
            var chMul = outChannels / inChannels;

            for (var b = 0; b < batchSize; ++b)
            {
                for (var d1 = 0; d1 < inChannels; ++d1)
                {
                    for (var xR = 0; xR < inHeight; ++xR)
                    {
                        var xRCorner = xR - topPad;
                        var xRMin = (int)Math.Max(0, Math.Ceiling((double)xRCorner / strideHeight));
                        var yRMax = (int)
                            Math.Min(outHeight, (filterHeight + xRCorner) / strideHeight);

                        for (var xC = 0; xC < inWidth; ++xC)
                        {
                            var xCCorner = xC - leftPad;
                            var xCMin = (int)Math.Max(0, Math.Ceiling((double)xCCorner / strideWidth));
                            var yCMax = (int)
                                Math.Min(outWidth, (filterWidth + xCCorner) / strideWidth);

                            var dotProd = 0.0f;
                            for (var yR = xRMin; yR < yRMax; ++yR)
                            {
                                var wR = yR * strideHeight - xRCorner;

                                for (var yC = xCMin; yC < yCMax; ++yC)
                                {
                                    var wC = yC * strideWidth - xCCorner;
                                    var dyOffset = dyS0 * b + dyS1 * yR + dyS2 * yC;
                                    var fltOffset = fltS0 * (filterHeight - 1 - wR) +
                                        fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;

                                    for (var dm = 0; dm < chMul; ++dm)
                                    {
                                        var d2 = d1 * chMul + dm;
                                        var pixel = dyValues[dyOffset + d2];
                                        var weight = fltValues[fltOffset + dm];
                                        dotProd += pixel * weight;
                                    }
                                }
                            }
                            dxValues[dxS0 * b + dxS1 * xR + dxS2 * xC + d1] = dotProd;
                        }
                    }
                }
            }
            return dx.toTensor();
        }

        public Tensor depthwiseConv2DDerFilter(Tensor dy, Tensor x, Conv2DInfo convInfo)
        {
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var dW = ops.buffer(convInfo.filterShape);

            var leftPad = convInfo.padInfo.left;
            var topPad = convInfo.padInfo.top;
            var chMul = convInfo.outChannels / convInfo.inChannels;

            for (var wR = 0; wR < filterHeight; ++wR)
            {
                var yRMin = (int)Math.Max(0, Math.Ceiling((double)(topPad - wR) / strideHeight));
                var yRMax = (int)Math.Min(
                    convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);

                for (var wC = 0; wC < filterWidth; ++wC)
                {
                    var yCMin = (int)Math.Max(0, Math.Ceiling((double)(leftPad - wC) / strideWidth));
                    var yCMax = (int)Math.Min(
                      convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);

                    for (var d2 = 0; d2 < convInfo.outChannels; ++d2)
                    {
                        var d1 = (int)Math.Truncate((double)d2 / chMul);
                        var dm = d2 % chMul;

                        var dotProd = 0.0f;
                        for (var b = 0; b < convInfo.batchSize; ++b)
                        {
                            for (var yR = yRMin; yR < yRMax; ++yR)
                            {
                                var xR = wR + yR * strideHeight - topPad;
                                for (var yC = yCMin; yC < yCMax; ++yC)
                                {
                                    var xC = wC + yC * strideWidth - leftPad;
                                    dotProd += x.Get(b, xR, xC, d1) * dy.Get(b, yR, yC, d2);
                                }
                            }
                        }
                        dW.Set(dotProd, wR, wC, d1, dm);
                    }
                }
            }
            return dW.toTensor();
        }

        public Tensor matMul(Tensor a, Tensor b, bool transposeA, bool transposeB)
        {
            var sharedDim = transposeA ? a.Shape[0] : a.Shape[1];
            var leftDim = transposeA ? a.Shape[1] : a.Shape[0];
            var rightDim = transposeB ? b.Shape[0] : b.Shape[1];

            var aValues = a.dataSync() as float[];
            var bValues = b.dataSync() as float[];

            int aOuterStep = 0;
            int aInnerStep = 0;
            if (transposeA)
            {
                aOuterStep = 1; aInnerStep = a.Strides[0];

            }
            else
            {

                aInnerStep = 1; aOuterStep = a.Strides[0];
            }


            int bOuterStep = 0;
            int bInnerStep = 0;

            if (transposeB)
            {
                bInnerStep = 1; bOuterStep = b.Strides[0];

            }
            else
            {

                bOuterStep = 1; bInnerStep = b.Strides[0];
            }


            var aOuterEnd = leftDim * aOuterStep;
            var bOuterEnd = rightDim * bOuterStep;

            var result = new float[leftDim * rightDim];
           
            var blockSize = this.blockSize;

            for (var i0 = 0; i0 < leftDim; i0 += blockSize)
            {
                for (var j0 = 0; j0 < rightDim; j0 += blockSize)
                {
                    for (var k0 = 0; k0 < sharedDim; k0 += blockSize)
                    {
                        // for when blockSize doesn't evenly divide the input
                        var iBlock = Math.Min(i0 + blockSize, leftDim);
                        var jBlock = Math.Min(j0 + blockSize, rightDim);
                        var kBlock = Math.Min(k0 + blockSize, sharedDim);

                        for (var i = i0; i < iBlock; i++)
                        {
                            for (var j = j0; j < jBlock; j++)
                            {
                                var sum = 0.0f;

                                for (var k = k0; k < kBlock; k++)
                                {
                                    sum += aValues[i * aOuterStep + k * aInnerStep] *
                                        bValues[k * bInnerStep + j * bOuterStep];
                                }
                                result[i * rightDim + j] += sum;
                            }
                        }
                    }
                }
            }
            return Ops.tensor2d(result, leftDim, rightDim);
        }


        public Tensor tile(Tensor x, int[] reps)
        {
            var newShape = new int[x.Rank];
            for (int i = 0; i < newShape.Length; i++)
            {
                newShape[i] = x.Shape[i] * reps[i];
            }
            var result = ops.buffer(newShape);
            var xBuf = x.buffer();
            for (int i = 0; i < result.values.Length; ++i)
            {
                var newLoc = result.indexToLoc(i);

                var originalLoc = new int[x.Rank];
                for (int j = 0; j < originalLoc.Length; j++)
                {
                    originalLoc[j] = newLoc[j] % x.Shape[j];
                }

                var originalIndex = xBuf.locToIndex(originalLoc);

                result.values[i] = xBuf.values[originalIndex];
            }


            return result.toTensor();
        }

        public Tensor pad(Tensor x, int[][] paddings, float varantValue)
        {
            var outShape = paddings.Select(
        (p, i) => p[0] /* beforePad */ + x.Shape[i] + p[1] /* afterPad */).ToArray();
            var start = paddings.Select(p => p[0]).ToArray();
            var xBuffer = x.buffer();
            var buffer = ops.buffer(outShape);
            if (varantValue != 0)
            {
                for (int i = 0; i < buffer.values.Length; i++)
                {
                    buffer.values[i] = varantValue;
                }
            }
            for (int i = 0; i < x.Size; i++)
            {
                var coords = xBuffer.indexToLoc(i);
                var outCoords = coords.Select((c, j) => c + start[j]).ToArray();
                buffer.Set(x.Get(coords), outCoords);
            }
            return buffer.toTensor();
        }

        public Tensor transpose(Tensor x, int[] perm)
        {
            int[] newShape = new int[x.Rank];
            for (int i = 0; i < newShape.Length; i++)
            {
                newShape[i] = x.Shape[perm[i]];
            }
            var values = x.dataSync();
            var xBuf = x.buffer();
            var result = ops.buffer(newShape);
            for (int i = 0; i < x.Size; ++i)
            {
                var loc = xBuf.indexToLoc(i);

                // Permute location.
                var newLoc = new int[loc.Length];
                for (int j = 0; j < newLoc.Length; j++)
                {
                    newLoc[j] = loc[perm[j]];
                }

                var newIndex = result.locToIndex(newLoc);
                result.values[newIndex] = values[i];
            }
            return result.toTensor();
        }

        public Tensor gather(Tensor x, Tensor indices, int axis)
        {
            var newShape = new List<int>(x.Shape).ToArray();
            var indicesValues = indices.dataSync();
            newShape[axis] = indicesValues.Length;
            var xBuf = x.buffer();
            var result = ops.buffer(newShape);


            for (int i = 0; i < result.Size; ++i)
            {
                var newLoc = result.indexToLoc(i);
                var originalLoc = new List<int>(newLoc).ToArray();
                originalLoc[axis] = (int)indicesValues[newLoc[axis]];

                var originalIndex = xBuf.locToIndex(originalLoc);

                result.values[i] = xBuf.values[originalIndex];
            }

            return result.toTensor();
        }

        private Tensor pool(Tensor x, Conv2DInfo convInfo, PoolType poolType)
        {
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var y = ops.buffer(convInfo.outShape);
            var padTop = convInfo.padInfo.top;
            var padLeft = convInfo.padInfo.left;


            for (var b = 0; b < convInfo.batchSize; ++b)
            {
                for (var d = 0; d < convInfo.inChannels; ++d)
                {
                    for (var yR = 0; yR < convInfo.outHeight; ++yR)
                    {
                        var xRCorner = yR * strideHeight - padTop;
                        var xRMin = Math.Max(0, xRCorner);
                        var xRMax = Math.Min(convInfo.inHeight, filterHeight + xRCorner);
                        for (var yC = 0; yC < convInfo.outWidth; ++yC)
                        {
                            var xCCorner = yC * strideWidth - padLeft;
                            var xCMin = Math.Max(0, xCCorner);
                            var xCMax = Math.Min(convInfo.inWidth, filterWidth + xCCorner);

                            var minMaxValue =
                        (poolType == PoolType.max ? float.NegativeInfinity :
                              float.PositiveInfinity);
                            var avgValue = 0.0f;
                            for (var xR = xRMin; xR < xRMax; ++xR)
                            {
                                for (var xC = xCMin; xC < xCMax; ++xC)
                                {
                                    var pixel = (float)x.Get(b, xR, xC, d);

                                    if ((poolType == PoolType.max && pixel > minMaxValue) ||
                                (poolType == PoolType.min && pixel < minMaxValue))
                                    {
                                        minMaxValue = pixel;
                                    }
                                    else if (poolType == PoolType.avg)
                                    {
                                        avgValue += pixel / (filterHeight * filterWidth);
                                    }
                                }
                                if (float.IsNaN(minMaxValue))
                                {
                                    break;
                                }
                            }
                            y.Set(poolType == PoolType.avg ? avgValue : minMaxValue, b, yR, yC, d);
                        }
                    }
                }
            }

            return y.toTensor();

        }

        public Tensor maxPool(Tensor x, Conv2DInfo convInfo)
        {
            return this.pool(x, convInfo, PoolType.max);
        }

        public Tensor maxPoolPositions(Tensor x, Conv2DInfo convInfo)
        {
            var maxPositions = ops.buffer(convInfo.outShape);
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var padTop = convInfo.padInfo.top;
            var padLeft = convInfo.padInfo.left;


            for (var b = 0; b < convInfo.batchSize; ++b)
            {
                for (var d = 0; d < convInfo.inChannels; ++d)
                {
                    for (var yR = 0; yR < convInfo.outHeight; ++yR)
                    {
                        var xRCorner = yR * strideHeight - padTop;
                        var xRMin = (float)Math.Max(0, xRCorner);
                        var xRMax = (float)Math.Min(convInfo.inHeight, filterHeight + xRCorner);
                        for (var yC = 0; yC < convInfo.outWidth; ++yC)
                        {
                            var xCCorner = yC * strideWidth - padLeft;
                            var xCMin = (float)Math.Max(0, xCCorner);
                            var xCMax = (float)Math.Min(convInfo.inWidth, filterWidth + xCCorner);
                            var maxValue = float.NegativeInfinity;
                            var maxPosition = -1;
                            for (var xR = xRMin; xR < xRMax; ++xR)
                            {
                                var wR = (int)xR - xRCorner;
                                for (var xC = xCMin; xC < xCMax; ++xC)
                                {
                                    var wC = (int)xC - xCCorner;
                                    var pixel = (float)x.Get(b, (int)xR, (int)xC, d);
                                    if (pixel > maxValue)
                                    {
                                        maxValue = pixel;
                                        maxPosition = (int)wR * filterWidth + (int)wC;
                                    }
                                }
                            }
                            maxPositions.Set(maxPosition, b, yR, yC, d);
                        }
                    }
                }
            }
            return maxPositions.toTensor();

        }

        public Tensor maxPoolBackprop(Tensor dy, Tensor x, Tensor y, Conv2DInfo convInfo)
        {
            var maxPositions = this.maxPoolPositions(x, convInfo);
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var padLeft = filterWidth - 1 - convInfo.padInfo.left;
            var padTop = filterHeight - 1 - convInfo.padInfo.top;
            var dx = ops.buffer(x.Shape);

            for (var b = 0; b < convInfo.batchSize; ++b)
            {
                for (var d = 0; d < convInfo.inChannels; ++d)
                {
                    for (var dxR = 0; dxR < convInfo.inHeight; ++dxR)
                    {
                        for (var dxC = 0; dxC < convInfo.inWidth; ++dxC)
                        {
                            // Shader code begins.
                            var dyRCorner = dxR - padTop;
                            var dyCCorner = dxC - padLeft;
                            var dotProd = 0f;
                            for (var wR = 0; wR < filterHeight; ++wR)
                            {
                                var dyR = (dyRCorner + wR) / strideHeight;
                                if (dyR < 0 || dyR >= convInfo.outHeight ||
                                    Math.Floor((float)dyR) != dyR)
                                {
                                    continue;
                                }
                                for (var wC = 0; wC < filterWidth; ++wC)
                                {
                                    var dyC = (dyCCorner + wC) / strideWidth;
                                    if (dyC < 0 || dyC >= convInfo.outWidth ||
                                Math.Floor((float)dyC) != dyC)
                                    {
                                        continue;
                                    }
                                    var maxPos = filterHeight * filterWidth - 1 -
                                maxPositions.Get(b, dyR, dyC, d);
                                    var curPos = wR * filterWidth + wC;

                                    var mask = maxPos == curPos ? 1 : 0;
                                    if (mask == 0)
                                    {
                                        continue;
                                    }

                                    var pixel = dy.Get(b, dyR, dyC, d);
                                    dotProd += pixel * mask;
                                }
                            }
                            dx.Set(dotProd, b, dxR, dxC, d);
                        }
                    }
                }
            }
            return dx.toTensor();
        }

        public Tensor avgPoolBackprop(Tensor dy, Tensor x, Conv2DInfo convInfo)
        {
            var strideHeight = convInfo.strideHeight;
            var strideWidth = convInfo.strideWidth;
            var filterHeight = convInfo.filterHeight;
            var filterWidth = convInfo.filterWidth;
            var padLeft = filterWidth - 1 - convInfo.padInfo.left;
            var padTop = filterHeight - 1 - convInfo.padInfo.top;
            var dx = ops.buffer(x.Shape);

            var avgMultiplier = 1 / (filterHeight * filterWidth);

            for (var b = 0; b < convInfo.batchSize; ++b)
            {
                for (var d = 0; d < convInfo.inChannels; ++d)
                {
                    for (var dxR = 0; dxR < convInfo.inHeight; ++dxR)
                    {
                        for (var dxC = 0; dxC < convInfo.inWidth; ++dxC)
                        {
                            // Shader code begins.
                            var dyRCorner = dxR - padTop;
                            var dyCCorner = dxC - padLeft;
                            var dotProd = 0.0f;
                            for (var wR = 0; wR < filterHeight; ++wR)
                            {
                                var dyR = (dyRCorner + wR) / (float)strideHeight;
                                if (dyR < 0 || dyR >= convInfo.outHeight ||
                                    (float)Math.Floor(dyR) != dyR)
                                {
                                    continue;
                                }
                                for (var wC = 0; wC < filterWidth; ++wC)
                                {
                                    var dyC = (dyCCorner + wC) / (float)strideWidth;
                                    if (dyC < 0 || dyC >= convInfo.outWidth ||
                                (float)Math.Floor(dyC) != dyC)
                                    {
                                        continue;
                                    }

                                    var pixel = (float)dy.Get(b, (int)dyR, (int)dyC, d);
                                    dotProd += pixel;
                                }
                            }
                            dx.Set(dotProd * avgMultiplier, b, dxR, dxC, d);
                        }
                    }
                }
            }

            return dx.toTensor();
        }

        public Tensor reshape(Tensor x, int[] shape)
        {
            return Tensor.Make(shape, new TensorData(x.dataId));


        }

        public Tensor avgPool(Tensor x, Conv2DInfo convInfo)
        {
            return this.pool(x, convInfo, PoolType.avg);
        }
        public Tensor resizeBilinear(Tensor x, int newHeight, int newWidth, bool alignCorners)
        {
            //  var [batch, oldHeight, oldWidth, numChannels] = x.shape;
            int batch = x.Shape[0];
            int oldHeight = x.Shape[1];
            int oldWidth = x.Shape[2];
            int numChannels = x.Shape[3];
            int[] newShape = new int[] { batch, newHeight, newWidth, numChannels };
            var output = ops.buffer(newShape);

            int[] effectiveInputSize;
            effectiveInputSize = alignCorners ? new int[2] { oldHeight - 1, oldWidth - 1 } : new int[2] { oldHeight, oldWidth };

            int[] effectiveOutputSize;
            effectiveOutputSize = alignCorners ? new int[2] { newHeight - 1, newWidth - 1 } : new int[2] { newHeight, newWidth };

            for (var b = 0; b < batch; b++)
            {
                for (var r = 0; r < newHeight; r++)
                {
                    for (var c = 0; c < newWidth; c++)
                    {
                        for (var d = 0; d < numChannels; d++)
                        {
                            // Begin shader.

                            // Compute the fractional index of the source.
                            var sourceFracRow =
                        (effectiveInputSize[0]) * r / (float)(effectiveOutputSize[0]);
                            var sourceFracCol =
                        (effectiveInputSize[1]) * c / (float)(effectiveOutputSize[1]);

                            var sourceRowFloor = (float)Math.Floor(sourceFracRow);
                            var sourceRowCeil =
                        (float)Math.Min(oldHeight - 1, (float)Math.Ceiling(sourceFracRow));
                            var sourceColFloor = (float)Math.Floor(sourceFracCol);
                            var sourceColCeil =
                        (float)Math.Min(oldWidth - 1, (float)Math.Ceiling(sourceFracCol));

                            var topLeft = (float)x.Get(b, (int)sourceRowFloor, (int)sourceColFloor, d);
                            var bottomLeft = (float)x.Get(b, (int)sourceRowCeil, (int)sourceColFloor, d);
                            var topRight = (float)x.Get(b, (int)sourceRowFloor, (int)sourceColCeil, d);
                            var bottomRight = (float)x.Get(b, (int)sourceRowCeil, (int)sourceColCeil, d);

                            var rowFrac = sourceFracRow - sourceRowFloor;
                            var colFrac = sourceFracCol - sourceColFloor;

                            var top = topLeft + (topRight - topLeft) * colFrac;
                            var bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
                            var newValue = top + (bottom - top) * rowFrac;

                            output.Set(newValue, b, r, c, d);
                        }
                    }
                }
            }

            return output.toTensor();
        }

        public Tensor resizeBilinearBackprop(Tensor dy, Tensor x, bool alignCorners)
        {
            var batch = x.Shape[0];
            var xHeight = x.Shape[1];
            var xWidth = x.Shape[2];
            var depth = x.Shape[3];
            var yHeight = dy.Shape[1];
            var yWidth = dy.Shape[2];
            var output =
        ops.buffer(new int[] { batch, xHeight, xWidth, depth });


            // In the backwards pass, we want to find the pixels that were generated for
            // each pixel in the input image the forward pass and add the corresponding
            // coefficient from dy to the gradient (with some interpolation).
            int[] effectiveXSize = new int[]{
      (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
      (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
    };


            int[] effectiveYSize = new int[]{
      (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
      (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
     };


            var heightScale = effectiveXSize[0] / effectiveYSize[0];
            var widthScale = effectiveXSize[1] / effectiveYSize[1];


            // Reference implementation
            // tslint:disable-next-line:max-line-length
            // https://github.com/tensorflow/tensorflow/blob/3039375c86a5bbc9610c7725dcaa95d635f87ba2/tensorflow/core/kernels/resize_bilinear_op.cc#L275
            for (var b = 0; b < batch; b++)
            {
                for (var r = 0; r < yHeight; r++)
                {
                    int dxR = r * heightScale;
                    int topDxRIndex = (int)Math.Floor((float)dxR);
                    int bottomDxRIndex = (int)Math.Min(Math.Ceiling((float)dxR), xHeight - 1);
                    int dxRLerp = dxR - topDxRIndex;
                    int inverseDxRLerp = 1 - dxRLerp;

                    for (var c = 0; c < yWidth; c++)
                    {
                        int dxC = c * widthScale;
                        int leftDxCIndex = (int)Math.Floor((float)dxC);
                        int rightDxCIndex = (int)Math.Min(Math.Ceiling((float)dxC), xWidth - 1);
                        int dxCLerp = dxC - leftDxCIndex;
                        int inverseDxCLerp = 1 - dxCLerp;

                        for (var d = 0; d < depth; d++)
                        {
                            var dyVal = dy.Get(b, r, c, d);

                            var topLeft = output.Get(b, topDxRIndex, leftDxCIndex, d);
                            topLeft += dyVal * inverseDxRLerp * inverseDxCLerp;
                            output.Set(topLeft, b, topDxRIndex, leftDxCIndex, d);

                            var topRight = output.Get(b, topDxRIndex, rightDxCIndex, d);
                            topRight += dyVal * inverseDxRLerp * dxCLerp;
                            output.Set(topRight, b, topDxRIndex, rightDxCIndex, d);

                            var bottomLeft = output.Get(b, bottomDxRIndex, leftDxCIndex, d);
                            bottomLeft += dyVal * dxRLerp * inverseDxCLerp;
                            output.Set(bottomLeft, b, bottomDxRIndex, leftDxCIndex, d);

                            var bottomRight = output.Get(b, bottomDxRIndex, rightDxCIndex, d);
                            bottomRight += dyVal * dxRLerp * dxCLerp;
                            output.Set(bottomRight, b, bottomDxRIndex, rightDxCIndex, d);
                        }
                    }
                }
            }

            return output.toTensor();
        }

        public Tensor resizeNearestNeighbor(Tensor x, int newHeight, int newWidth, bool alignCorners)
        {
            //  var [batch, oldHeight, oldWidth, numChannels] = x.shape;
            int batch = x.Shape[0];
            int oldHeight = x.Shape[1];
            int oldWidth = x.Shape[2];
            int numChannels = x.Shape[3];
            int[] newShape = new int[] { batch, newHeight, newWidth, numChannels };
            var output = ops.buffer(newShape);

            int[] effectiveInputSize;
            effectiveInputSize = alignCorners ? new int[2] { oldHeight - 1, oldWidth - 1 } : new int[2] { oldHeight, oldWidth };

            int[] effectiveOutputSize;
            effectiveOutputSize = alignCorners ? new int[2] { newHeight - 1, newWidth - 1 } : new int[2] { newHeight, newWidth };

            for (var b = 0; b < batch; b++)
            {
                for (var r = 0; r < newHeight; r++)
                {
                    for (var c = 0; c < newWidth; c++)
                    {
                        for (var d = 0; d < numChannels; d++)
                        {
                            // Begin shader.
                            // Compute the fractional index of the source.
                            var sourceFracRow =
                        (effectiveInputSize[0]) * r / (effectiveOutputSize[0]);
                            var sourceFracCol =
                        (effectiveInputSize[1]) * c / (effectiveOutputSize[1]);
                            var sourceNearestRow = (int)
                        Math.Min(oldHeight - 1, Math.Round((float)sourceFracRow));
                            var sourceNearestCol = (int)
                        Math.Min(oldWidth - 1, Math.Round((float)sourceFracCol));
                            var newValue = x.Get(b, sourceNearestRow, sourceNearestCol, d);
                            output.Set(newValue, b, r, c, d);
                        }
                    }
                }
            }

            return output.toTensor();
        }

        public Tensor batchNormalization(Tensor x, Tensor mean, Tensor variance, float varianceEpsilon,
            Tensor scale = null, Tensor offset = null)
        {
            var xVals = x.dataSync();
            var mVals = mean.dataSync();
            var varVals = variance.dataSync();
            var sVals = scale != null ? scale.dataSync() : new float[] { 1 };
            var offVals = offset != null ? offset.dataSync() : new float[] { 0 };
            var outVals = new float[xVals.Length];

            var offValsLength = offVals.Length;
            var sValsLength = sVals.Length;
            var varValsLength = varVals.Length;
            var mValsLength = mVals.Length;

            var offi = 0;
            var mi = 0;
            var si = 0;
            var vi = 0;
            for (var i = 0; i < xVals.Length; ++i)
            {
                outVals[i] = offVals[offi++] +
                    (xVals[i] - mVals[mi++]) * sVals[si++] /
                      (float)Math.Sqrt(varVals[vi++] + varianceEpsilon);
                if (offi >= offValsLength)
                {
                    offi = 0;
                }
                if (mi >= mValsLength)
                {
                    mi = 0;
                }
                if (si >= sValsLength)
                {
                    si = 0;
                }
                if (vi >= varValsLength)
                {
                    vi = 0;
                }
            }
            return ops.tensor(outVals, x.Shape);

        }

        public Tensor LRNGrad(Tensor dy, Tensor inputImage, Tensor outputImage, float depthRadius
            , float bias, float alpha, float beta)
        {
            var batch = dy.Shape[0];
            var rows = dy.Shape[1];
            var cols = dy.Shape[2];
            var depth = dy.Shape[3];
            var output = ops.buffer(ops.shape(batch, rows, cols, depth));

            for (var b = 0; b < batch; ++b)
            {
                for (var r = 0; r < rows; ++r)
                {
                    for (var c = 0; c < cols; ++c)
                    {
                        for (var d = 0; d < depth; ++d)
                        {
                            var depthBegin = (int)System.Math.Max(0, d - depthRadius);
                            var depthEnd = (int)System.Math.Min(depth, d + depthRadius + 1);

                            var norm = 0.0f;
                            for (var k = depthBegin; k < depthEnd; ++k)
                            {
                                norm += inputImage.Get(b, r, c, k) * inputImage.Get(b, r, c, k);
                            }
                            norm = alpha * norm + bias;
                            for (var k = depthBegin; k < depthEnd; ++k)
                            {
                                var dyi = -2 * alpha * beta * inputImage.Get(b, r, c, k) *
                                    outputImage.Get(b, r, c, d) / norm;
                                if (d == k)
                                {
                                    dyi += (float)System.Math.Pow(norm, -beta);
                                }
                                dyi *= dy.Get(b, r, c, d);
                                output.Set(dyi + output.Get(b, r, c, k), b, r, c, k);
                            }
                        }
                    }
                }
            }
            return output.toTensor();
        }

        public Tensor localResponseNormalization4D(Tensor x, float radius, float bias, float alpha, float beta)
        {
            var output = ops.buffer(x.Shape);
            var rad = radius;
            var maxW = output.shape[1] - 1;
            var maxH = output.shape[2] - 1;
            var maxD = output.shape[3] - 1;

            Func<int, int, int, int, float> sumAcrossChannels = (int b, int r, int c, int d) =>
            {
                var sum = 0.0f;
                for (int j = (int)(float)Math.Max(0, d - rad); j <= (float)Math.Min(d + rad, maxD);
                     j++)
                {
                    var z = (float)x.Get(b, r, c, j);
                    sum += z * z;
                }
                return sum;
            };

            Func<int, int, int, int, float> sumWithinChannel = (int b, int r, int c, int d) =>
            {
                var sum = 0.0f;
                for (int u = (int)(float)Math.Max(0, r - rad); u <= (float)Math.Min(r + rad, maxW);
                     u++)
                {
                    for (int v = (int)(float)Math.Max(0, c - rad); v <= (float)Math.Min(c + rad, maxH);
                 v++)
                    {
                        sum += (float)Math.Pow((float)x.Get(b, u, v, d), 2);
                    }
                }
                return sum;
            };


            for (int b = 0; b < output.shape[0]; b++)
            {
                for (int r = 0; r <= output.shape[1]; r++)
                {
                    for (int c = 0; c < output.shape[2]; c++)
                    {
                        for (int d = 0; d < output.shape[3]; d++)
                        {
                            float sum =
                        sumAcrossChannels(b, r, c, d);
                            float val = (float)x.Get(b, r, c, d) * (float)Math.Pow(bias + alpha * sum, -beta);
                            output.Set(val, b, r, c, d);
                        }
                    }
                }
            }

            return output.toTensor();

        }

        public Tensor multinomial(Tensor logits, bool normalized, int numSamples, double seed)
        {
            var probabilities = normalized ? logits : ops.softmax(logits);
            var batchSize = probabilities.Shape[0];
            var numEvents = probabilities.Shape[1];
            var res = ops.zeros(new int[] { batchSize, numSamples });
            var resVals = res.dataSync() as float[];
            var probVals = probabilities.dataSync() as float[];


            for (int b = 0; b < batchSize; ++b)
            {
                int offset = b * numEvents;
                // The cdf won'Tensor include the last event. It will be implicit if no other
                // event happened.
                var cdf = new float[numEvents - 1];
                cdf[0] = probVals[offset];
                for (int ev = 1; ev < cdf.Length; ++ev)
                {
                    cdf[ev] = cdf[ev - 1] + probVals[offset + ev];
                }

                AleaRandomization random = new AleaRandomization(seed.ToString());
                var outOffset = b * numSamples;
                for (int sampleId = 0; sampleId < numSamples; ++sampleId)
                {
                    var r = random.random();

                    // Assume last event happened by default.
                    resVals[outOffset + sampleId] = cdf.Length;

                    for (var ev = 0; ev < cdf.Length; ev++)
                    {
                        if (r < cdf[ev])
                        {
                            resVals[outOffset + sampleId] = ev;
                            break;
                        }
                    }
                }
            }

            return res;
        }


        public Tensor oneHot(Tensor indices, int depth, float onValue, float offValue)
        {
            float[] res = new float[indices.Size * depth];
            float[] indval = indices.dataSync();
            for (int i = 0; i < res.Length; i++)
            {

                res[i] = offValue;
            }

            for (var ev = 0; ev < indices.Size; ++ev)
            {
                if (indices.Get(ev) >= 0 && indices.Get(ev) < depth)
                {
                    res[ev * depth + (int)indval[ev]] = onValue;
                }
            }
            return ops.tensor2d(res, indices.Size, depth);
        }

        public void register(WeakReference dataId, int[] shape)
        {
            if (!data.Keys.Contains(dataId))
            {
                this.data.Add(dataId, null);
            }


        }
        public void write(WeakReference dataId, float[] values)
        {
            this.data[dataId] = values;
        }
        public float[] readSync(WeakReference dataId)
        {
            return this.data[dataId];
        }
        public void disposeData(WeakReference dataId)
        {
            this.data[dataId] = null;
            dataId.Target = null;
            this.data.Remove(dataId);
            dataId = null;

        }

        public long time(Action f)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            f();
            sw.Stop();
            return sw.ElapsedMilliseconds;
        }


        public Tensor batchToSpaceND(Tensor x, int[] blockShape, int[][] crops)
        {
            var prod = blockShape.Aggregate((a, b) => a * b);

            var reshaped = Util.getReshaped(x.Shape, blockShape, prod);
            var permuted =
                Util.getPermuted(reshaped.Length, blockShape.Length);
            var reshapedPermuted =
                Util.getReshapedPermuted(x.Shape, blockShape, prod);
            var sliceBeginCoords =
                Util.getSliceBeginCoords(crops, blockShape.Length);
            var sliceSize =
                Util.getSliceSize(reshapedPermuted, crops, blockShape.Length);
            return x.reshape(reshaped)
              .transpose(permuted)
              .reshape(reshapedPermuted)
        .slice(sliceBeginCoords, sliceSize);
        }
    }


}
