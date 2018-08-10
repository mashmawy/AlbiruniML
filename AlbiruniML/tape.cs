using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{ 
    public class NamedGradientMap
    {
        public NamedGradientMap()
        {
            this.gradient = new Dictionary<string, Func<Tensor>>();
        }
        public Dictionary<string, Func<Tensor>> gradient { get; set; } 
    }

    public class TapeNode
    {
        public int id { get; set; }
        public string name { get; set; }
        public Tensor output { get; set; }
        public Dictionary<string, Tensor> inputs { get; set; }
        public Func<Tensor, NamedGradientMap> gradient { get; set; }

    }

    public class Tape
    {
        public static T CreateFromObjects<T>(params T[] sources) where T : new()
        {
            var ret = new T();
            MergeObjects(ret, sources);

            return ret;
        }

        public static void MergeObjects<T>(T target, params T[] sources)
        {
            Func<PropertyInfo, T, bool> predicate = (p, s) =>
            {
                if (p.GetValue(s).Equals(GetDefault(p.PropertyType)))
                {
                    return false;
                }

                return true;
            };

            MergeObjects(target, predicate, sources);
        }

        public static void MergeObjects<T>(T target, Func<PropertyInfo, T, bool> predicate, params T[] sources)
        {
            foreach (var propertyInfo in typeof(T).GetProperties().Where(prop => prop.CanRead && prop.CanWrite))
            {
                foreach (var source in sources)
                {
                    if (predicate(propertyInfo, source))
                    {
                        propertyInfo.SetValue(target, propertyInfo.GetValue(source));
                    }
                }
            }
        }

        private static object GetDefault(Type type)
        {
            if (type.IsValueType)
            {
                return Activator.CreateInstance(type);
            }
            return null;
        }
         
        public static TapeNode[] getFilteredNodesXToY(TapeNode[] tape, Tensor[] xs, Tensor y)
        { 
            // Forward pass to compute all the nodes and Tensors that are transitively a
            // function of x.
            Dictionary<int, bool> tensorsFromX = new Dictionary<int, bool>();
            Dictionary<int, bool> nodesFromX = new Dictionary<int, bool>();
            for (var i = 0; i < xs.Length; i++)
            {
                tensorsFromX.Add(xs[i].id, true);
            }

            for (int i = 0; i < tape.Length; i++)
            {
                var node = tape[i];
                var nodeInputs = node.inputs;
                foreach (var input in nodeInputs)
                {

                    var inputT = input.Value;
                    var anyInputFromX = false;
                    for (var j = 0; j < xs.Length; j++)
                    {
                        if (tensorsFromX.ContainsKey(inputT.id))
                        {
                            tensorsFromX.Add(node.output.id, true);// = true;
                            anyInputFromX = true;
                            nodesFromX.Add(node.id, true);
                            break;
                        }
                    }

                    if (anyInputFromX)
                    {
                        break;
                    }
                }
            }

            // Backwards pass to find all of the nodes and Tensors that lead to y.
            Dictionary<int, bool> tensorsLeadToY = new Dictionary<int, bool>();
            tensorsLeadToY.Add(y.id, true);


            Dictionary<int, bool> nodesToY = new Dictionary<int, bool>();


            for (var i = tape.Length - 1; i >= 0; i--)
            {
                var node = tape[i];
                var nodeInputs = node.inputs;

                List<Tensor> outputs = new List<Tensor>();
                outputs.Add(node.output);

                for (var j = 0; j < outputs.Count; j++)
                {
                    if (tensorsLeadToY.ContainsKey(outputs[j].id))
                    {
                         

                            foreach (var item in nodeInputs)
                            {
                                if (tensorsLeadToY.ContainsKey(nodeInputs[item.Key].id))
                                {
                                    tensorsLeadToY[nodeInputs[item.Key].id] = true;
                                }
                                else
                                {

                                    tensorsLeadToY.Add(nodeInputs[item.Key].id, true);
                                }

                                if (nodesToY.ContainsKey(node.id))
                                {
                                    nodesToY[node.id] = true;
                                }
                                else
                                {

                                    nodesToY.Add(node.id, true);
                                }
                            }
                            break;
                        
                    }
                    
                }
            }

            // Return the paths that come from x and lead to y.

            List<TapeNode> filteredTape = new List<TapeNode>();
            for (var i = 0; i < tape.Length; i++)
            {
                var node = tape[i];

                if (nodesFromX.ContainsKey(node.id) && nodesToY.ContainsKey(node.id))
                {
                    Dictionary<string, Tensor> prunedInputs = new Dictionary<string, Tensor>();
                    foreach (var item in node.inputs)
                    {
                        var nodeInput = item.Value;
                        if (tensorsFromX.ContainsKey(nodeInput.id))
                        {
                            prunedInputs.Add(item.Key, nodeInput);
                        }
                    }
                    TapeNode prunedNode = new TapeNode(){
                      id = node.id,
                     name = node.name,
                     gradient = node.gradient
                    }   ;

                    prunedNode.inputs = prunedInputs;
                    prunedNode.output = node.output;
                    filteredTape.Add(prunedNode);
                }

            }
            return filteredTape.ToArray();

        }


        public static void backpropagateGradients(Dictionary<int, Tensor>
            tensorAccumulatedGradientMap, TapeNode[] filteredTape)
        {
            // Walk the tape backwards and keep a map of Tensor to its gradient
            for (var i = filteredTape.Length - 1; i >= 0; i--)
            {
                var node = filteredTape[i];
                var dy = tensorAccumulatedGradientMap[node.output.id];

                //Backprop dy through this node and accumulate gradients over the inputs.
                var inputGradients = node.gradient(dy);

                foreach (var inputName in node.inputs)
                {

                    // Call the gradient function.
                    var dx = inputGradients.gradient[inputName.Key]();

                    var x = inputName.Value;
                    if (!tensorAccumulatedGradientMap.ContainsKey(x.id) )
                    { 
                        tensorAccumulatedGradientMap.Add(x.id, dx);
                    }
                    else
                    {
                        var curGradient = tensorAccumulatedGradientMap[x.id];
                        tensorAccumulatedGradientMap[x.id] = curGradient.add(dx);
                        curGradient.dispose();
                    }
                }
            }
        }
    }

}
