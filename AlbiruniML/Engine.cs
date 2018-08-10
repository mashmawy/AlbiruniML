using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{



    public static partial class Ops
    {
        public static void setBackend(IBackend bk)
        {
            ENV.engine.SetBackEnd(bk);
        }
    }
    public class ScopeState
    { 
        public List<Tensor> track = new List<Tensor>();
        public string name;
    }

    public class MemoryInfo
    {
        public int numTensors { get; set; }
        public int numDataBuffers { get; set; }
        public int numBytes { get; set; }
        public bool unreliable { get; set; }

    }
    public interface BackendTimingInfo { int kernelMs { get; set; } }

    public delegate Tensor ForwardFunc(IBackend bk, Func<Tensor, Tensor> saved = null);
    public delegate CustomGradientResults CustomGradientFunc(params Tensor[] args);

    public class Engine
    {
        public Dictionary<string, Variable> registeredVariables = new Dictionary<string, Variable>();
        private Dictionary<WeakReference, int> refCounter = new Dictionary<WeakReference, int>();
        private int nextTapeNodeId = 0;
        private int numBytes = 0;
        private int numTensors = 0;
        private int numDataBuffers = 0;

        private List<TapeNode> activeTape;
        public int gradientScopeCount = 0;
        private int customGradientDepth = 0;

        // Keep Tensors that parallel the tapes.
        private ScopeState activeScope;
        private Stack<ScopeState> scopeStack = new Stack<ScopeState>();
        private IBackend backend;
        public bool customBackend;
        public bool safeMode;
        private List<int> keepTensors = new List<int>();
        int fNumbers = 0;
        public   void tidy(string name, Action fn, bool gradMode = false)
        {
            if (name==null)
            {
                name = "func" + fNumbers.ToString();
                fNumbers++;
            }
            if (name.Trim().Length==0)
            {
                name = "func" + fNumbers.ToString();
                fNumbers++;
            }
            ENV.engine.startScope(name, gradMode);
            fn();

            ENV.engine.endScope(new List<Tensor>(), gradMode);

        }
        public   List<Tensor> tidy(string name, Func<List<Tensor>> fn, bool gradMode = false)
        {
            if (name == null)
            {
                name = "func" + fNumbers.ToString();
                fNumbers++;
            }
            if (name.Trim().Length == 0)
            {
                name = "func" + fNumbers.ToString();
                fNumbers++;
            }
            ENV.engine.startScope(name, gradMode);
            var result = fn();

            ENV.engine.endScope(result, gradMode);
            return result;
        }
        public   Tensor tidy(string name, Func<Tensor> fn, bool gradMode = false)
        {
            if (name == null)
            {
                name = "func" + fNumbers.ToString();
                fNumbers++;
            }
            if (name.Trim().Length == 0)
            {
                name = "func" + fNumbers.ToString();
                fNumbers++;
            }
            ENV.engine.startScope(name, gradMode);
            var result = fn();

            ENV.engine.endScope(new List<Tensor>() { result }, gradMode);
            return result;
        }

        public void tidy(  Action fn, bool gradMode = false)
        {
            string name = "func" + fNumbers.ToString();
            fNumbers++;
            if (name.Trim().Length == 0)
            {
                name = "func" + fNumbers.ToString();
                fNumbers++;
            }
            ENV.engine.startScope(name, gradMode);
            fn();

            ENV.engine.endScope(new List<Tensor>(), gradMode);

        }
        public List<Tensor> tidy(  Func<List<Tensor>> fn, bool gradMode = false)
        {
            string name = "func" + fNumbers.ToString();
            fNumbers++;
            ENV.engine.startScope(name, gradMode);
            var result = fn();

            ENV.engine.endScope(result, gradMode);
            return result;
        }
        public Tensor tidy( Func<Tensor> fn, bool gradMode = false)
        {
            string  name = "func" + fNumbers.ToString();
                fNumbers++;
            
            ENV.engine.startScope(name, gradMode);
            var result = fn();

            ENV.engine.endScope(new List<Tensor>() { result }, gradMode);
            return result;
        }
        

        public Engine()
        {
            this.backend = new Backend_CPU();
            this.activeScope = new ScopeState() {  name="default_scope", track=new List<Tensor>()} ;
            this.scopeStack.Push(this.activeScope);
        }

        public void SetBackEnd(IBackend bk)
        {
            this.backend = bk;
        }
        public Tensor runKernel(ForwardFunc forwardFunc,
            Dictionary<string, Tensor> inputs, Func<Tensor, List<Tensor>, NamedGradientMap> grad = null)
        {
            Tensor result;
            List<Tensor> saved = new List<Tensor>();
            Func<Tensor, Tensor> saveFunc = (Tensor x) =>
            {
                saved.Add(x);
                return x;
            };
            var scopeName = this.activeScope.name;
            // Stop recording to a tape when running a kernel.
            this.customGradientDepth++;  
            result = forwardFunc(this.backend, saveFunc);
            // Continue recording after the kernel is done.
            this.customGradientDepth--;
               if (this.shouldRecord())
                
            {
                var tapeNode = new TapeNode()
                {
                    id = this.nextTapeNodeId++,
                    name = scopeName,
                    inputs = inputs,
                    output = result
                };

                if (grad != null)
                {
                    tapeNode.gradient = (Tensor dy) =>
                    {
                        return grad(dy, saved);
                    };
                }
                this.activeTape.Add(tapeNode);

            }
           

            return result;
        }

        public void registerTensor(Tensor a)
        {
            var refCount = this.refCounter.ContainsKey(a.dataId) ? this.refCounter[a.dataId] : 0;

            this.numTensors++;

            if (refCount == 0)
            {
                this.numDataBuffers++;
                this.numBytes +=
                    backend.SizeFromShape(a.Shape) * 4;
                this.backend.register(a.dataId, a.Shape);
            }
            this.refCounter[a.dataId] = refCount + 1;
            if (!(a.GetType() == typeof(Variable)))
            {
                this.track(a);
            }
        }
        public void registerVariable(Variable v)
        {
            if (this.registeredVariables.ContainsKey(v.Name))
            {
                throw new Exception("Variable with name " + v.Name + " was already registered");
            }
            this.registeredVariables.Add(v.Name, v);

        }
        public void disposeTensor(Tensor a)
        {
            if (!this.refCounter.ContainsKey(a.dataId))
            {
                return;
            }
            if (this.keepTensors.Contains(a.id))
            {
                this.keepTensors.Remove(a.id);
            }
            this.numTensors--;
            var refCount = this.refCounter[a.dataId];


            if (refCount <= 1)
            {
                this.refCounter.Remove(a.dataId);
                this.backend.disposeData(a.dataId);
                this.numDataBuffers--;
                this.numBytes -=
                    Util.SizeFromShape(a.Shape) * 4; 
            }
            else
            {

                this.refCounter[a.dataId] = refCount - 1;
            }
        }
        public void disposeVariables()
        {
            foreach (var item in this.registeredVariables.Keys.ToList())
            {
                this.registeredVariables[item].dispose();
                this.registeredVariables.Remove(item);
            }
        }

        public MemoryInfo memory()
        {
            var info = this.backend.memory();
            info.numTensors = this.numTensors;
            info.numDataBuffers = this.numDataBuffers;
            info.numBytes = this.numBytes;
            return info;
        }
        private bool shouldRecord()
        {
            return this.activeTape != null && this.customGradientDepth == 0;
        }

        private void addTapeNode(Tensor[] inputs, Tensor result, Func<Tensor, Tensor[]> gradientsFunc)
        {
            var inputsMap = new Dictionary<string, Tensor>();
            for (int i = 0; i < inputs.Length; i++)
            {
                inputsMap.Add(i.ToString(), inputs[i]);
            }

            Func<Tensor, NamedGradientMap> gradient = (Tensor dy) =>
            {
                var res = gradientsFunc(dy);
                var resMap = new NamedGradientMap();
                
                var outer = 0;
                foreach (var item in res)
                {

                    resMap.gradient.Add(outer.ToString(), () => { return item; });
                    outer++;
                }
                return resMap;
            };

            TapeNode tapeNode = new TapeNode()
            {
                id = this.nextTapeNodeId++,
                name = this.activeScope.name,
                inputs = inputsMap,
                output = result,
                gradient = gradient
            };

            this.activeTape.Add(tapeNode);
        }


        public Tensor keep(Tensor t)
        {
              if (this.scopeStack.Count == 1 && this.safeMode) {
      throw new Exception(
          "Safe mode is ON. Enclose all tensor operations inside alb.tidy(): " +
          "alb.tidy(() => {...}) to avoid memory leaks.");
}
            this.keepTensors.Add(t.id);
            return t;
        }
        private Tensor track(Tensor t)
        {
            this.activeScope.track.Add(t);
            return t;
        }

        public void startScope(string name = null, bool gradientsMode = false)
        {
            if (gradientsMode && this.gradientScopeCount == 0)
            {
                this.activeTape = new List<TapeNode>();
            }
            if (gradientsMode)
            {
                this.gradientScopeCount++;
            }
            ScopeState scopeInfo = new ScopeState(); 
            scopeInfo.track = new List<Tensor>();
            if (name != null)
            {
                scopeInfo.name = name;
            }
            this.scopeStack.Push(scopeInfo);
            this.activeScope = scopeInfo;
        }


        public void endScope(List<Tensor> result, bool gradientsMode = false)
        {
            if (gradientsMode)
            {
                this.gradientScopeCount--;
                if (this.gradientScopeCount == 0)
                {
                    this.activeTape = null;
                }
            }

            var tensorsToKeep = new List<int>(this.keepTensors.ToArray());
            var tensorsToTrackInParent = new List<Tensor>(result.ToArray());
            tensorsToKeep.AddRange(tensorsToTrackInParent.Select(p => p.id).ToArray());
            for (var i = 0; i < this.activeScope.track.Count; i++)
            {
                var tensor = this.activeScope.track[i];
                if (tensorsToKeep.Contains(tensor.id))
                {
                    continue;
                }

                if (this.activeTape != null)
                {
                    tensorsToTrackInParent.Add(tensor);
                }
                else
                {
                    tensor.dispose();
                }
            }

            var oldScope = this.scopeStack.Pop();


            this.activeScope = this.scopeStack.Count == 0 ?
         new ScopeState() { track = new List<Tensor>() } :
        this.scopeStack.FirstOrDefault();

            foreach (var tensor in tensorsToTrackInParent)
            {
                if (!this.keepTensors.Contains(tensor.id)
                    && oldScope.track.Where(p=>p.id==tensor.id).Count()>0

                    )
                {
                    this.track(tensor);
                }
            }

        }


        public GradientResults gradients(Func<Tensor> f, Tensor[] xs, Tensor dy = null, bool allowNoGradients = false)
        {
            GradientResults res = new GradientResults();
            this.tidy("gradients", () =>
            {
                var y = f();

                var filteredTape = Tape.getFilteredNodesXToY(this.activeTape.ToArray(), xs, y);
                var accumulatedGradientMap = new Dictionary<int, Tensor>();
                accumulatedGradientMap.Add(y.id, (dy == null) ?
                    Ops.ones(y.Shape) : dy);

                Tape.backpropagateGradients(accumulatedGradientMap, filteredTape);

                var grades = xs.Select(p => accumulatedGradientMap[p.id]).ToList();
                res.value = y;
                res.Grades = grades;
                return grades;
            }, true);

            return res;

        }



        public GradientResults gradients(Func<Tensor> f, Variable[] xs, Tensor dy = null, bool allowNoGradients = false)
        {
            GradientResults res = new GradientResults();
            this.tidy("gradients", () =>
            {
                var y = f();

                var filteredTape = Tape.getFilteredNodesXToY(this.activeTape.ToArray(), xs, y);
                var accumulatedGradientMap = new Dictionary<int, Tensor>();
                accumulatedGradientMap.Add(y.id, (dy == null) ?
                    Ops.ones(y.Shape) : dy);

                Tape.backpropagateGradients(accumulatedGradientMap, filteredTape);

                var grades = xs.Select(p => accumulatedGradientMap[p.id]).ToList();
                res.value = y;
                res.Grades = grades;
                return grades;
            }, true);

            return res;

        }

        public Func<Tensor[], Tensor> customGrad(CustomGradientFunc f, string fName)
        {
            return (Tensor[] inputs) =>
            {

                this.customGradientDepth++;
                Func<Tensor, List<Tensor>> gradientsFunc = null;
                var gradientsMode = true;

                var result = this.tidy("gradients", () =>
                {
                    var gr = f(inputs);
                    gradientsFunc = gr.gradFunc;
                    return gr.value;

                }, gradientsMode);

                this.customGradientDepth--;

                if (this.shouldRecord())
                {
                    Func<Tensor, Tensor[]> gradFunc = (Tensor dy) =>
                    {
                        return gradientsFunc(dy).ToArray();

                    };
                    this.addTapeNode(inputs, result, gradFunc);
                }
                return result;
            };
        }
        public void write(WeakReference dataId, float[] values)
        {
            this.backend.write(dataId, values);
        }
        public float[] readSync(WeakReference dataId)
        {

            return this.backend.readSync(dataId);
        }


    }

    public class GradientResults
    {
        public Tensor value { get; set; }
        public List<Tensor> Grades { get; set; }

    }
    public class CustomGradientResults
    {
        public Tensor value { get; set; }
        public Func<Tensor, List<Tensor>> gradFunc { get; set; }

    }
}
