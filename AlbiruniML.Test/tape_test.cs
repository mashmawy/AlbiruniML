using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using alb = AlbiruniML.Ops;
using AlbiruniML;
using System.Collections.Generic;
namespace AlbiruniML.Test
{
    [TestClass]
    public class tape_test
    {
        [TestMethod]
        public void getFilteredNodesXToY_no_paths_from_x_to_y()
        {
            ENV.engine = new Engine();
            var x = alb.scalar(1);
            var intermediate1 = alb.scalar(0);

            var intermediate2 = alb.scalar(0);
            var y = alb.scalar(2);
            var input1 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input1.Add("x", x);
            var input2 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input2.Add("intermediate2", intermediate2);
            TapeNode[] tape = new TapeNode[]{
                new TapeNode(){
                     gradient=null
                     , id=0
                     , inputs=input1, 
                      name="node0",
                       output=intermediate1
                } ,
                new TapeNode(){
                     gradient=null
                     , id=1
                     , inputs=input2, 
                      name="node1",
                       output=y
                } 
            };
            var filteredTapeNodes = Tape.getFilteredNodesXToY(tape, new Tensor[] { x }, y);


            Assert.AreEqual(filteredTapeNodes.Length, 0);

        }
        [TestMethod]
        public void getFilteredNodesXToY_one_operation_x_to_y()
        {
            ENV.engine = new Engine();
            var x = alb.scalar(1);
            var y = alb.scalar(2);
            var input1 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input1.Add("x", x);
            TapeNode[] tape = new TapeNode[]{
                new TapeNode(){
                     gradient=null
                     , id=0
                     , inputs=input1, 
                      name="node0",
                       output=y
                }  
            };
            var filteredTapeNodes =
                Tape.getFilteredNodesXToY(tape, new Tensor[] { x }, y);


            Assert.AreEqual(filteredTapeNodes.Length, 1);
        }
          
        [TestMethod]
        public void getFilteredNodesXToY_1_operation_x0_x1_y_all_input_paths()
        {
            ENV.engine = new Engine();
            var x0 = alb.scalar(0);
            var x1 = alb.scalar(1);
            var y = alb.scalar(2);

            var input1 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input1.Add("x0", x0);
            input1.Add("x1", x1);
            TapeNode[] tape = new TapeNode[]{
                new TapeNode(){
                     gradient=null
                     , id=0
                     , inputs=input1, 
                      name="node0",
                       output=y
                }  
            };
            var filteredTapeNodes =
                Tape.getFilteredNodesXToY(tape, new Tensor[] { x0, x1 }, y);


            Assert.AreEqual(filteredTapeNodes.Length, 1);
            Assert.AreEqual(filteredTapeNodes[0].inputs.Count, 2);
        }
         
        [TestMethod]
        public void getFilteredNodesXToY_two_operations_x_intermediate_y()
        {
            ENV.engine = new Engine();
            var x = alb.scalar(1);
            var intermediate = alb.scalar(0);
            var y = alb.scalar(2);

            var input1 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input1.Add("x", x);



            var input2 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input2.Add("intermediate", intermediate);
            TapeNode[] tape = new TapeNode[]{
                new TapeNode(){
                     gradient=null
                     , id=0
                     , inputs=input1, 
                      name="node0",
                       output=intermediate
                }  ,   new TapeNode(){
                     gradient=null
                     , id=1
                     , inputs=input2, 
                      name="node1",
                       output=y
                }  
            };
            var filteredTapeNodes =
                Tape.getFilteredNodesXToY(tape, new Tensor[] { x }, y);


            Assert.AreEqual(filteredTapeNodes.Length, 2);
            AssertTools.TapeIsEqual(filteredTapeNodes, tape);
        }
         
        [TestMethod]
        public void getFilteredNodesXToY_two_operations_x0_x1_x2_intermediate_y()
        {
            ENV.engine = new Engine();
            var x0 = alb.scalar(1);
            var x1 = alb.scalar(2);
            var x2 = alb.scalar(3);
            var intermediate = alb.scalar(4);
            var y = alb.scalar(2);

            var input1 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input1.Add("x0", x0);
            input1.Add("x1", x1);



            var input2 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input2.Add("x2", x2);
            input2.Add("intermediate", intermediate);

            TapeNode[] tape = new TapeNode[]{
                new TapeNode(){
                     gradient=null
                     , id=0
                     , inputs=input1, 
                      name="node0",
                       output=intermediate
                }  ,   new TapeNode(){
                     gradient=null
                     , id=1
                     , inputs=input2, 
                      name="node1",
                       output=y
                }  
            };

            var filteredTapeNodes =
           Tape.getFilteredNodesXToY(tape, new Tensor[] { x0, x1, x2 }, y);
            Assert.AreEqual(filteredTapeNodes.Length, 2);
            AssertTools.TapeIsEqual(filteredTapeNodes, tape);

        }
         
        [TestMethod]
        public void getFilteredNodesXToY_x_y_and_x_orphan()
        {
            ENV.engine = new Engine();
            var x = alb.scalar(1);
            var orphan = alb.scalar(0);
            var y = alb.scalar(2);


            var input1 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input1.Add("x", x);

            var input2 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input2.Add("x", x);
            TapeNode[] tape = new TapeNode[]{
                new TapeNode(){
                     gradient=null
                     , id=0
                     , inputs=input1, 
                      name="node0",
                       output=orphan
                }  ,   new TapeNode(){
                     gradient=null
                     , id=1
                     , inputs=input2, 
                      name="node1",
                       output=y
                }  
            };
            var filteredTapeNodes =
             Tape.getFilteredNodesXToY(tape, new Tensor[] { x }, y);


            Assert.AreEqual(filteredTapeNodes.Length, 1);
            AssertTools.TapeNodeIsEqual(filteredTapeNodes[0], tape[1]);
        }
         
        [TestMethod]
        public void getFilteredNodesXToY_x_y_and_orphan_y()
        {
            ENV.engine = new Engine();
            var x = alb.scalar(1);
            var orphan = alb.scalar(0);
            var y = alb.scalar(2);
             
            var input1 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input1.Add("x", x);
            input1.Add("orphan", orphan);

            var input2 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input2.Add("x", x);
            TapeNode[] tape = new TapeNode[]{
                new TapeNode(){
                     gradient=null
                     , id=0
                     , inputs=input1, 
                      name="node0",
                       output=y
                }   
            };
            var filteredTapeNodes =
             Tape.getFilteredNodesXToY(tape, new Tensor[] { x }, y);
             
            Assert.AreEqual(filteredTapeNodes.Length, 1);
            AssertTools.TapeNodeIsEqual(filteredTapeNodes[0],

                new TapeNode()
                {
                    id = 0,
                    name = "node0",
                    gradient = null,
                    inputs = input2,
                    output = y
                }

                );
        }

        
        [TestMethod]
        public void basic_backprop_with_1_node()
        {
            ENV.engine = new Engine();
            var x = alb.scalar(0);
            var y = alb.scalar(1);

            var dy = alb.scalar(1);

            Dictionary<int, Tensor> accumulatedGradientsMap = new Dictionary<int, Tensor>();
            accumulatedGradientsMap.Add(y.id, dy);

            var input1 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input1.Add("x", x);

            TapeNode[] tape = new TapeNode[]{
                new TapeNode(){
                     gradient=(Tensor dy2) =>
            {
                NamedGradientMap ngm = new NamedGradientMap();
                ngm.gradient.Add("x", () =>
                {
                    return dy2.add(alb.scalar(1));
                });
                return ngm;
            }
                     , id=0
                     , inputs=input1, 
                      name="node0",
                       output=y
                }   
            };

            Tape.backpropagateGradients(accumulatedGradientsMap, tape);


            AssertTools.ArrayIsEqual(accumulatedGradientsMap[x.id].dataSync(), new float[] { 2 });
        }


        [TestMethod]
        public void basic_backprop_with_2_node()
        {
            ENV.engine = new Engine();
            var x = alb.scalar(0);
            var intermediate = alb.scalar(1);
            var y = alb.scalar(2);

            var dy = alb.scalar(1);

            Dictionary<int, Tensor> accumulatedGradientsMap = new Dictionary<int, Tensor>();
            accumulatedGradientsMap.Add(y.id, dy);

            var input1 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input1.Add("x", x);

            var input2 = new System.Collections.Generic.Dictionary<string, Tensor>();
            input2.Add("intermediate", intermediate);

            TapeNode[] tape = new TapeNode[]{
                new TapeNode(){
                     gradient=(Tensor dy2) =>
                        {
                            NamedGradientMap ngm = new NamedGradientMap();
                            ngm.gradient.Add("x", () =>
                            {
                                return dy2.add(alb.scalar(1));
                            });
                            return ngm;
                        }
                     , id=0
                     , inputs=input1, 
                      name="node0",
                       output=intermediate
                }   

                ,

                  new TapeNode(){
                     gradient=(Tensor dy2) =>
                        {
                            NamedGradientMap ngm = new NamedGradientMap();
                            ngm.gradient.Add("intermediate", () =>
                            {
                                return dy2.add(alb.scalar(1));
                            });
                            return ngm;
                        }
                     , id=1
                     , inputs=input2, 
                      name="node1",
                       output=y
                }   
            };

            Tape.backpropagateGradients(accumulatedGradientsMap, tape);


            AssertTools.ArrayIsEqual(accumulatedGradientsMap[x.id].dataSync(), new float[] { 3 });
        }

        [TestMethod]
        public static void basic_backprop_with_a_split_node_accumulates_gradients()
        {
            ENV.engine = new Engine();
            var x = alb.scalar(0);
            var intermediate1 = alb.scalar(1);
            var intermediate2 = alb.scalar(2);
            var y = alb.scalar(3);

            var dy = alb.scalar(1);
            Dictionary<int, Tensor> accumulatedGradientsMap =
                new Dictionary<int, Tensor>();
            accumulatedGradientsMap.Add(y.id, dy);



            TapeNode node0 = new TapeNode();
            node0.id = 0;
            node0.name = "node0";
            node0.inputs = new Dictionary<string, Tensor>();
            node0.inputs.Add("x", x);
            node0.output = intermediate1;
            node0.gradient = (Tensor dy2) =>
            {
                NamedGradientMap ngm = new NamedGradientMap();
                ngm.gradient.Add("x", () =>
                {
                    return dy2.add(alb.scalar(1));
                });
                return ngm;
            };



            TapeNode node1 = new TapeNode();
            node1.id = 1;
            node1.name = "node1";
            node1.inputs = new Dictionary<string, Tensor>();
            node1.inputs.Add("x", x);
            node1.output = intermediate2;
            node1.gradient = (Tensor dy2) =>
            {
                NamedGradientMap ngm = new NamedGradientMap();
                ngm.gradient.Add("x", () =>
                {
                    return dy2.add(alb.scalar(1));
                });
                return ngm;
            };





            TapeNode node2 = new TapeNode();
            node2.id = 2;
            node2.name = "node2";
            node2.inputs = new Dictionary<string, Tensor>();
            node2.inputs.Add("intermediate1", intermediate1);
            node2.inputs.Add("intermediate2", intermediate2);
            node2.output = y;
            node2.gradient = (Tensor dy2) =>
            {
                NamedGradientMap ngm = new NamedGradientMap();
                ngm.gradient.Add("intermediate1", () =>
                {
                    return dy2.add(alb.scalar(1));
                });
                ngm.gradient.Add("intermediate2", () =>
                {
                    return dy2.add(alb.scalar(1));
                });
                return ngm;
            };

            TapeNode[] tape = new TapeNode[] { node0, node1, node2 };

            Tape.backpropagateGradients(accumulatedGradientsMap, tape);
            AssertTools.ArrayIsEqual(accumulatedGradientsMap[x.id].dataSync(), new float[] { dy.dataSync()[0] + 5 });
      
        }

    }
}
