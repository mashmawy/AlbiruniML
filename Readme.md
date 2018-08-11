# AlbiruniML
AlbiruniML is a linear algebra and machine learning library written in pure c# langauge, inspired from tensorflow and the great work of tensorflowjs team.
For now only CPU is supported, cblas native support will be added soon and GPU kernel "CUDA" will be also supported.
# Main Features
  - Only float datatype can be used.
  - Support NDArray and Tensor operations 
  - Automatic Differentiation
  - Easy to import many of tensorflow models.
# Building the library
Currently there is no third party components and you can build directly from Visual Studio in x64 bit Environment.
Other platforms is not tested yet.

# Using the library
Like tensorflow and tensorflowjs, AlbiruniML use Tensor data structure as its basic data structure for all its operations.
Tensor is a generalization to vectors, matrix, volume or higher dimension array. 
To use AlbiruniML first use the required namespaces.
```
using Albiruni;
using alb = Albiruni.Ops;
```
For a simple logistic regression task where numbers less than 15 is false and numbers greater then 15 is true, we generate the dataset and labels as follow
```cs
Tensor xs1 = new float[] 
{ 1f, 20f, 11f, 21f, 15, 25, 5, 30, 4, 20, 6, 11.5f, 22 }.ToTensor();
Tensor ys1 = new float[] 
{ 0f, 1f, 0f, 1f, 0, 1, 0, 1, 0, 1, 0, 0, 1 }.ToTensor();
```  
next we define variable a and b to be trained with random value.
```cs
Random r = new Random();
var a = alb.variable(alb.scalar((float)r.NextDouble()));
var c = alb.variable(alb.scalar(0));
```
To optimize those variables we need an optimizer, we will use a simple stochastic gradient descent optimizer with learning rate 0.1:
```cs
var learningRate = 0.1f;
var optimizer = alb.train.sgd(learningRate);
```

next we define our simple logistic regression model :
```cs
Func<Tensor, Tensor> model = (Tensor x) =>
{  
        var y = a * x  + c;   
        return y.sigmoid(); 
};
```
next we define the training loop and execute minimize function of the optimizer.
Minimize is where the magic happens, we must return a numerical estimate (i.e. loss) of how well we are doing using the current state of the variables we created at the start. 
Here we use meanSquaredError loss function
This optimizer does the 'backward' step of our training data updating variables defined previously in order to minimize the loss.
```cs
Action<Tensor, Tensor, int, Action> train = (Tensor examples, Tensor label, int numIterations, Action done) =>
{
    for (int iter = 0; iter < numIterations; iter++)
    {  
            optimizer.minimize(() =>
            { 
                //Feed the examples into the model
                var pred = model(examples);
                var predLoss = alb.loss.meanSquaredError(label, pred); 
                return predLoss;
            }); 
    } 
    done();
};
```


next we define a test function to check our model performance against the labels
```cs 
Action<Tensor, Tensor> test = (Tensor xs, Tensor ys) =>
{
    var xvalues = xs.ToArray();
    var yvalues = ys.ToArray();
    //Execute the model
    var predictedYs = model(xs).ToArray(); 
    Console.Write("Expected : ");
    for (int i = 0; i < yvalues.Length; i++)
    {
        Console.Write( yvalues[i].ToString()+", " );
    }
    Console.WriteLine();
    Console.Write("Got      : ");
    for (int i = 0; i < predictedYs.Length; i++)
    {
        var pred = predictedYs[i] > 0.5f ? 1 : 0;
        Console.Write( pred.ToString()+", " );
    } 
};
```

Finally we start the training process and test the trained variables.
```cs
train(xs1, ys1, 1000, () =>
{ 
    test(xs1, ys1); 
});
```
and the output should be like that:
```sh
Expected : 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
Got      : 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
```
and here is the full example :
```cs
using System;
using Albiruni;
using alb = Albiruni.Ops;
namespace SimpleExample
{
    class Program
    {
        static void Main(string[] args)
        {
            Tensor xs1 = new float[] 
            { 1f, 20f, 11f, 21f, 15, 25, 5, 30, 4, 20, 6, 11.5f, 22 }.ToTensor();

            Tensor ys1 = new float[] 
            { 0f, 1f, 0f, 1f, 0, 1, 0, 1, 0, 1, 0, 0, 1 }.ToTensor();

            Random r = new Random();
            var a = alb.variable(alb.scalar((float)r.NextDouble()));
            var c = alb.variable(alb.scalar(0));
             
            var learningRate = 0.1f;
            var optimizer = alb.train.sgd(learningRate);
             
            Func<Tensor, Tensor> model = (Tensor x) =>
            {
                var y = a * x + c;
                return y.sigmoid();
            };
             
            Action<Tensor, Tensor, int, Action> train = 
            (Tensor examples, Tensor label, int numIterations, Action done) =>
            {
                for (int iter = 0; iter < numIterations; iter++)
                {
                    optimizer.minimize(() =>
                    {
                        // Feed the examples into the model
                        var pred = model(examples); 
                        var predLoss = alb.loss.meanSquaredError(label, pred);
                        return predLoss;
                    });
                }
                done();
            };

            Action<Tensor, Tensor> test = (Tensor xs, Tensor ys) =>
            {
                var xvalues = xs.ToArray();
                var yvalues = ys.ToArray();
                var predictedYs = model(xs).ToArray();
                Console.Write("Expected : ");
                for (int i = 0; i < yvalues.Length; i++)
                {
                    Console.Write(yvalues[i].ToString() + ", ");
                }
                Console.WriteLine();
                Console.Write("Got      : ");
                for (int i = 0; i < predictedYs.Length; i++)
                {
                    var pred = predictedYs[i] > 0.5f ? 1 : 0;
                    Console.Write(pred.ToString() + ", ");
                }
            };

            train(xs1, ys1, 1000, () =>
            {
                test(xs1, ys1); 
            });
        } 
    }
}

```


