
#r "../../packages/MathNet.Numerics.3.2.3/lib/net40/MathNet.Numerics.dll"
#r "../../packages/MathNet.Numerics.FSharp.3.2.3/lib/net40/MathNet.Numerics.FSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"

#load "GaussianProcess.fs"
#load "Kernels.fs"
#load "Optimization.fs"

open MathNet.Numerics.Distributions
open FSharp.Charting
open Ariadne
open Ariadne.GaussianProcess

let rnd = System.Random(1)

// Sample data
let n = 10
let xs = 
    [| for i in 0..n-1 -> rnd.NextDouble() * 10.0 |]
let ys = xs |> Array.map (fun x -> (sin x) + Normal.Sample(rnd, 0.0, 0.1))
let data = {Locations = xs; Observations = ys}

Chart.Point(Array.zip xs ys)
|> Chart.WithMarkers(Size=10,Color=System.Drawing.Color.Black)
|> Chart.WithXAxis(Min=(-1.0), Max=12.0, Title="inputs")
|> Chart.WithYAxis(Title="outputs")
|> Chart.WithYAxis(Min=(-2.0), Max=2.0)

// -------------------------------
// Basic Gaussian process

// Create a Gaussian process
let sigma = 2.0
let lengthscale = 2.0
let seBasic (x1,x2) = sigma**2.0 * exp( - ((x1 - x2)**2.0)/(2.0*lengthscale**2.0))

let gpBasic = GaussianProcess.GaussianProcess<float>(seBasic, Some(0.1))
gpBasic |> GaussianProcess.plot [data]

gpBasic.LogLikelihood [data]

// -------------------------------
// Squared exponential kernel
open Ariadne.Kernels

let lengthscalePrior = LogNormal.WithMeanVariance(1.0, 1.0, rnd)
let variancePrior = LogNormal.WithMeanVariance(1.0, 1.0, rnd)
let noisePrior = LogNormal.WithMeanVariance(0.1, 0.1, rnd)

let sep = SquaredExp.Prior(lengthscalePrior, variancePrior, noisePrior)
let se = sep.Sample()
let gp = GaussianProcess.GaussianProcess<float>(se.Kernel, Some(se.NoiseVariance))
gp |> GaussianProcess.plot [data]
gp |> GaussianProcess.plotRange (-1.0, 12.0) [data]

// Optimization
open Ariadne.Optimization

// Metropolis Hastings sampling
let newParams = 
    se
    |> SquaredExp.optimizeMetropolisHastings [data] MetropolisHastings.defaultSettings sep

let newGp = newParams.GaussianProcess()
newGp |> GaussianProcess.plot [data]
newGp |> GaussianProcess.plotRange (-1.0, 12.0) [data]
|> Chart.WithXAxis(Min=(-1.0), Max=12.0, Title="inputs")
|> Chart.WithYAxis(Title="outputs")

gp |> GaussianProcess.plot [data]
printfn "Original Gaussian process likelihood: %f" (gp.LogLikelihood [data])
printfn "Optimized Gaussian process likelihood: %f" (newGp.LogLikelihood [data])

// Gradient descent - experimental!
let settings = {GradientDescent.Iterations = 100000; GradientDescent.StepSize = 0.0000001}
let newParams = 
    gradientDescent (SquaredExp.fullGradient [data]) settings (se.Parameters)
    |> SquaredExp.ofParameters
let newGp = GaussianProcess.GaussianProcess<float>(newParams.Kernel, Some(newParams.NoiseVariance))
printfn "Original Gaussian process likelihood: %f" (gp.LogLikelihood [data])
printfn "Optimized Gaussian process likelihood: %f" (newGp.LogLikelihood [data])

newGp |> GaussianProcess.plot [data]
gp |> GaussianProcess.plot [data]




//--------------------------
let n = 10
let xs = 
    [| for i in 0..n-1 -> rnd.NextDouble() * 10.0 |]
let ys = xs |> Array.map (fun x -> (exp (-x)) * (sin x) + Normal.Sample(rnd, 0.0, 0.1))
let data = 
    [| for i in 0..4 ->
        {Locations = xs; Observations = ys |> Array.map (fun y -> y + Normal.Sample(rnd, 0.0, 0.1))} |]

Chart.Point(Array.zip xs ys)

let newParams = 
    se
    |> SquaredExp.optimizeMetropolisHastings data MetropolisHastings.defaultSettings sep

let newGp = newParams.GaussianProcess()
newGp |> GaussianProcess.plot data
newGp |> GaussianProcess.plotRange (-1.0, 11.0) data
|> Chart.WithXAxis(Min=(-1.0), Max=11.0)



//=========================================
#I "../../bin/"
#I "../../packages/MathNet.Numerics.3.2.3/lib/net40/"
#I "../../packages/MathNet.Numerics.FSharp.3.2.3/lib/net40/"
#I "../../packages/FSharp.Charting.0.90.7/"
#I "../../packages/FSharp.Data.2.0.14/lib/net40/"

#r "Ariadne.dll"
open Ariadne.GaussianProcess
open Ariadne.Kernels

#r "FSharp.Data.dll"
#r "FSharp.Charting.dll"
#r "MathNet.Numerics.dll"

open FSharp.Data
open FSharp.Charting

// Gaussian process data
let data =  [
   {Locations =
     [|1988.0; 1993.0; 1996.0; 1999.0; 2001.0; 2002.0; 2003.0; 2004.0; 2005.0;
       2006.0; 2007.0; 2008.0; 2009.0|];
    Observations =
     [|-15.52307692; 9.056923077; 6.786923077; -1.843076923; 0.2769230769;
       -3.623076923; -2.063076923; -2.183076923; -1.813076923; 2.806923077;
       4.386923077; 2.946923077; 0.7869230769|];}]



let lengthscale = 2.0
let signalVariance = 15.0
let noiseVariance = 10.0

let sqExp = SquaredExp.SquaredExp(lengthscale, signalVariance, noiseVariance)

let gp = sqExp.GaussianProcess()
gp |> plot data |> Chart.WithXAxis(false)