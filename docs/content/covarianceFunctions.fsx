(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
#I "../../bin/"
#I "../../packages/MathNet.Numerics.3.2.3/lib/net40/"
#I "../../packages/MathNet.Numerics.FSharp.3.2.3/lib/net40/"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"
#I "../../packages/FSharp.Data.2.0.14/lib/net40/"

(**
Covariance functions  
==========================================

The only currently implemented covariance function is the squared exponential.

$$$
k(

Gaussian process - doesn't assume specific curve as a relation between 
data, assumes only some form of relation within the data (like smoothness)

- It's a Bayesian method - assumes some prior over the data, computes posterior
using the data etc. 

- 



How to fit a model to World Bank data

*)
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"

#r "Ariadne.dll"
open Ariadne.GaussianProcess
open Ariadne.Kernels

#r "FSharp.Data.dll"
open FSharp.Data

let data = WorldBankData.GetDataContext()

let gpData = 
    data
      .Countries.``United Kingdom``
      .Indicators.``School enrollment, tertiary (% gross)``
    |> Seq.toArray
    |> Array.map (fun (year, n) -> {Locations = [| float year|]; Observations = [|n|]})

let kernel = SquaredExp.SquaredExp(2.0, 10.0, 1.0)
let gp = kernel.GaussianProcess()
gp |> plot gpData

// Specify a prior
open MathNet.Numerics.Distributions
open Ariadne.Optimization

let rnd = System.Random()
let lengthscalePrior = LogNormal.WithMeanVariance(2.0, 5.0, rnd)
let variancePrior = LogNormal.WithMeanVariance(1.0, 5.0, rnd)
let noisePrior = LogNormal.WithMeanVariance(1.0, 1.0, rnd)

let prior = SquaredExp.Prior(lengthscalePrior, variancePrior, noisePrior)

// Metropolis Hastings sampling
let newParams = 
    kernel
    |> SquaredExp.optimizeMetropolisHastings gpData MetropolisHastings.defaultSettings prior

let newGp = newParams.GaussianProcess()
newGp |> plot gpData

(**
Explanation of kernel parameters 
- lengthscale - how close on the x axis points need to be to affect each other
- signal variance - how strong is the force driving the regression curve away from the mean
 (which is zero in this case)
- noise - how much noise do we expect
*)
