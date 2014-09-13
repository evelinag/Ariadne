(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
#I "../../bin/"
#I "../../packages/MathNet.Numerics.3.2.3/lib/net40/"
#I "../../packages/MathNet.Numerics.FSharp.3.2.3/lib/net40/"
#I "../../packages/FSharp.Charting.0.90.7/"
#I "../../packages/FSharp.Data.2.0.14/lib/net40/"

(**
Quick how-to
==========================================

This page provides a very brief overview of main 
functions available in Ariadne for Gaussian process regression.

First step is to download Ariadne from [NuGet](http://nuget.org/packages/Ariadne) and load it 
into F# interactive. We will also define some data in the Gaussian process format.
The rest of this document shows how to do specific tasks with Gaussian processes.
*) 
#r "Ariadne.dll"
#r "MathNet.Numerics.dll"
open Ariadne
open Ariadne.GaussianProcess
open Ariadne.Kernels
open Ariadne.Optimization

let data = 
  [{Locations = (*[omit:(...)]*)
     [|1988.0; 1993.0; 1996.0; 1999.0; 2001.0; 2002.0; 2003.0; 2004.0; 2005.0;
       2006.0; 2007.0; 2008.0; 2009.0|];(*[/omit]*)
    Observations = (*[omit:(...)]*)
     [|-15.52307692; 9.056923077; 6.786923077; -1.843076923; 0.2769230769;
       -3.623076923; -2.063076923; -2.183076923; -1.813076923; 2.806923077;
       4.386923077; 2.946923077; 0.7869230769|];(*[/omit]*) }]

(**
### Create covariance function (kernel)
*)

let lengthscale = 3.0
let signalVariance = 15.0
let noiseVariance = 1.0

let sqExp = SquaredExp.SquaredExp(lengthscale, signalVariance, noiseVariance)

(**
###  Create Gaussian process with a covariance function
*)

// method 1
let gp1 = GaussianProcess(sqExp.Kernel, Some sqExp.NoiseVariance)

// method 2
let gp2 = sqExp.GaussianProcess()

(**
###  Compute log likelihood
*)

let loglik = gp1.LogLikelihood data

(**
###  Use Metropolis-Hastings to find values for hyperparameters
*)
open MathNet.Numerics

// specify prior distribution
let lengthscalePrior = Distributions.LogNormal.WithMeanVariance(2.0, 4.0)
let variancePrior = Distributions.LogNormal.WithMeanVariance(1.0, 5.0)
let noisePrior = Distributions.LogNormal.WithMeanVariance(0.5, 1.0)
let prior = SquaredExp.Prior(lengthscalePrior, variancePrior, noisePrior)

// get posterior mean values for hyperparameters
let sqExpMH = 
    sqExp
    |> SquaredExp.optimizeMetropolis data MetropolisHastings.defaultSettings prior

(**
### Use gradient descent to find values for hyperparameters
*)

let settings = 
    { GradientDescent.Iterations = 10000; 
      GradientDescent.StepSize = 0.01}

let gradientFunction parameters = SquaredExp.fullGradient data parameters

// Run gradient descent
let sqExpGD = 
    gradientDescent gradientFunction settings (sqExp.Parameters)
    |> SquaredExp.ofParameters

(**
### Compute prediction for new time points (locations)
*)

let allYears = [| 1985.0 .. 2015.0 |]
let predictValues, predictVariance = allYears |> gp1.Predict data

(**
### Compute predictive log likelihood for new observed data
*)

let newObservation = {
    Locations = [| 2015.0 |]; Observations = [| 10.0 |]}

let predictiveLoglik = gp1.PredictiveLogLikelihood data newObservation

(**
### Plot the fitted Gaussian process 
*)
#r "FSharp.Charting.dll"

GaussianProcess.plot data gp1
GaussianProcess.plotRange (1980.0, 2020.0) data gp1