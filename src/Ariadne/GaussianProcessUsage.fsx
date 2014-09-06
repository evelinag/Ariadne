
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

let sep = SquaredExponential.Prior(lengthscalePrior, variancePrior, noisePrior)
let se = sep.Sample()
let gp = GaussianProcess.GaussianProcess<float>(se.Kernel, Some(se.NoiseVariance))
gp |> GaussianProcess.plot [data]
gp |> GaussianProcess.plotRange (-1.0, 12.0) [data]

// Optimization
open Ariadne.Optimization

// Metropolis Hastings sampling
let newParams = 
    se
    |> SquaredExponential.optimizeMetropolisHastings data MetropolisHastings.defaultSettings sep

let newGp = newParams.GaussianProcess()
newGp |> GaussianProcess.plot [data]
gp |> GaussianProcess.plot [data]
printfn "Original Gaussian process likelihood: %f" (gp.LogLikelihood [data])
printfn "Optimized Gaussian process likelihood: %f" (newGp.LogLikelihood [data])

// Gradient descent - experimental!
let settings = {GradientDescent.Iterations = 100000; GradientDescent.StepSize = 0.0000001}
let newParams = 
    gradientDescent (SquaredExponential.fullGradient [data]) settings (se.Parameters)
    |> SquaredExponential.ofParameters
let newGp = GaussianProcess.GaussianProcess<float>(newParams.Kernel, Some(newParams.NoiseVariance))
printfn "Original Gaussian process likelihood: %f" (gp.LogLikelihood [data])
printfn "Optimized Gaussian process likelihood: %f" (newGp.LogLikelihood [data])

newGp |> GaussianProcess.plot [data]
gp |> GaussianProcess.plot [data]
