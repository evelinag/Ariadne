(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
#I "../../bin/"
#I "../../packages/MathNet.Numerics.3.2.3/lib/net40/"
#I "../../packages/MathNet.Numerics.FSharp.3.2.3/lib/net40/"
#I "../../packages/FSharp.Charting.0.90.7/"
#I "../../packages/FSharp.Data.2.0.14/lib/net40/"

(**
Getting started: Simple regression with Ariadne 
==========================================

This tutorial shows how to use the Ariadne library for Gaussian process regression.

As a first step, we need to get `Ariadne.dll` from [NuGet](https://nuget.org/packages/Ariadne).
Then we can load the Ariadne library inside F#.

*) 
#r "Ariadne.dll"
open Ariadne.GaussianProcess
open Ariadne.Kernels

(**
We will introduce Ariadne through a practical example. Consider a time series dataset with
missing values. For example, we might be interested in income distribution and inequality 
in Russia over the past 30 years. We can download available data from the World Bank
using [FSharp.Data library](http://fsharp.github.io/FSharp.Data/). 
We will use [FSharp.Charting](http://fsharp.github.io/FSharp.Charting/)
to display the data. We will also need a reference to [Math.NET Numerics](http://numerics.mathdotnet.com/).
*)

(*** define-output:rawPlot ***)

#r "FSharp.Data.dll"
#r "FSharp.Charting.dll"
#r "MathNet.Numerics.dll"

open FSharp.Data
open FSharp.Charting

let wb = WorldBankData.GetDataContext()

let rawData = 
    wb.Countries.``Russian Federation``.Indicators.``GINI index`` 
    |> Seq.toArray

Chart.Point(rawData)
(*** include-it:rawPlot ***)

(**
The snippet downloads raw data from the World Bank on the Gini index, which measures
income inequality. We can see that the time series contains missing data, for example there 
are no values between years 1988 and 1993.
*)
(*** include-value:rawData ***)

(**
We can use Gaussian process regression model to interpolate the observed data and estimate
Gini index in the years when it was not measured. As a first step, we need to prepare the 
downloaded data.

Data preprocessing
----------------------------
*)

let years, values = Array.unzip rawData

// inputs
let xs = years |> Array.map float   
// centred outputs
let ys = 
    let avg = Array.average values
    values |> Array.map (fun y -> y - avg)

// Gaussian process data
let data = [{Locations = xs; Observations = ys}]

(**
We formulated the data in the form expected by the Gaussian process library. General data
for Gaussian process consist of a sequence of series measurements. 
In our case, we have only one time series. 
Each series of measurements contains locations of each measurement (time points in our example)
and observed values for each location (Gini index values).
The `Locations` and `Observations` arrays should have equal length.

In general applications, locations and observations are not restricted to time series. 
For example locations can represent geographical coordinates and observations might record
rainfall at each location. 

Covariance function
----------------------------------
Covariance functions (also called kernels) are the key ingredient in using Gaussian processes. 
They encode all
assumptions about the form of function that we are modelling. In general, covariance represents
some form of distance or similarity. 
Consider two input points (locations) $ x_i $ and $ x_j $ with corresponding observed 
values $ y_i $ and $ y_j $. If the inputs $ x_i $ and $ x_j $ are close to each other,  we 
generally expect that 
 $ y_i $ and $ y_j $ will be close as well. This measure of similarity is embedded in the covariance
 function. 

Ariadne currently contains implementation of the most commonly used covariance function, 
the squared exponential kernel. For more information see [Covariance functions](covarianceFunctions.html).
The following snippet creates the squared exponential covariance function. 
*)

// hyperparameters
let lengthscale = 3.0
let signalVariance = 15.0
let noiseVariance = 1.0

let sqExp = SquaredExp.SquaredExp(lengthscale, signalVariance, noiseVariance)

(**
The hyperparameters regulate specific behaviour of the squared exponential kernel. For details see
the [Covariance functions](covarianceFunctions.html) section. Details on how to select
values for hyperparameters are in the [Optimization](optimization.html) section of this website.

Gaussian process regression
-------------------------------------------

There are two ways how to create a Gaussian process model. The first one uses directly 
the squared exponential
kernel that we created in the previous snippet.
The second option is to create Gaussian process model 'from scratch' by specifying a
covariance function, prior mean and optional noise variance. The library currently implements
only zero prior mean.
*)

// First option how to create a Gaussian process model
let gp = sqExp.GaussianProcess()

// Second option how to create a Gaussian process model
let covFun = sqExp.Kernel
let gp1 = GaussianProcess(covFun, ZeroMean, Some noiseVariance)

(**
Now that we created a Gaussian process, we can use it to compute regression function.
Please note that current implementation of inference in Gaussian processes is exact
and therefore requires $\mathcal{O} (N^3)$ time, where $N$ is the number of data points.

We can compute log likelihood to see how well the Gaussian process model 
fits our World bank data. The log likelihood may be used to compare different
models.
*)

let loglik = gp.LogLikelihood data

(*** include-value:loglik ***)

(**
We can also use Gaussian process to estimate values of the Gini index in 
years where there are no data. The `Predict` function gives us the mean 
estimate for each time location and variance of the estimate. 
*)

(*** define-output:predictedValues ***)
let allYears = [| 1985.0 .. 2015.0 |]
let predictValues, predictVariance = allYears |> gp.Predict data

Array.zip3 allYears predictValues predictVariance
|> Array.iteri (fun i (year, gini, var) -> 
    if i < 5 then
        printfn "%.0f : %.3f (+/- %.3f)" year gini (sqrt var)) 

(**
We can print the predicted values for each year together with their 
standard deviations.
*)
(*** include-output:predictedValues ***)

(**
To display the full posterior regression function, we can also automatically 
plot the Gaussian process using [F# Charting](http://fsharp.github.io/FSharp.Charting/). 
There are two functions for Gaussian process charts included in the library. 
There is a basic `plot` function, which uses Gaussian process to interpolate observed values.
The `plotRange` function extrapolates the Gaussian process over a specified 
range of values. We can use it to draw a graph of estimated Gini index values between years
1995 to 2015.
*)
(*** define-output:gpPlot ***)

gp |> plotRange (1985.0, 2015.0) data

(*** include-it:gpPlot ***)

(**
Continue to [Covariance functions](covarianceFunctions.html) to find out more about the
squared exponential covariance. [Optimization](optimization.html) provides an overview of 
how to fit hyperparameters of covariance functions (lengthscale, signal variance etc).
*)