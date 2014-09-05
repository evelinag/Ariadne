#if INTERACTIVE
#r "../../packages/NUnit.2.6.3/lib/nunit.framework.dll"
#r "../../packages/FsUnit.1.3.0.1/Lib/Net40/FsUnit.NUnit.dll"
#r "../../packages/MathNet.Numerics.3.2.1/lib/net40/MathNet.Numerics.dll"
#r "../../packages/MathNet.Numerics.FSharp.3.2.1/lib/net40/MathNet.Numerics.FSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"

#load "../../src/Ariadne/GaussianProcess.fs"
#load "../../src/Ariadne/Kernels.fs"
#load "../../src/Ariadne/Optimization.fs"
#else
module Ariadne.Tests.Kernels
#endif

open NUnit.Framework
open FsUnit

open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra

open Ariadne
open Ariadne.GaussianProcess
open Ariadne.Kernels
open Ariadne.Optimization

// -------------------------------------------
// Squared exponential kernel
// -------------------------------------------
// Numerical values are checked against Matlab package GPML

[<Test>]
let ``Can create Squared exponential from parameters and give them back afterwards`` () = 
    // create squared exponential from parameters
    let parameters = [| 3.5; 1.1; 0.7 |]
    (SquaredExponential.ofParameters parameters).Parameters
    |> should equal parameters

[<Test>]
let ``Value of squared exponential kernel is correct`` () = 
    let xs = [ 0.0; 1.0; -2.0; 3.0; 4.0 ]
    let ys = [ 0.0; 1.5; -3.0; 7.0; -1.0 ] 
    let kernel = SquaredExponential.ofParameters [| 3.5; 1.1; 0.7 |]
    let values = List.map2 (fun x y -> kernel.Kernel (x,y)) xs ys
    let gpmlValues = [] // TODO
    values |> should (equalWithin 1e-9) gpmlValues

[<Test>]
let ``Prior log likelihood is correct`` () =
    // Compute prior log likelihood numerically

    let kernel = SquaredExponential.ofParameters [| 3.5; 1.1; 0.7 |]
    let lengthscalePrior = LogNormal.WithMeanVariance(1.1, 0.1)
    let signalPrior = LogNormal.WithMeanVariance(1.0, 0.5)
    let noisePrior = LogNormal.WithMeanVariance(0.1, 0.1)
    let prior = SquaredExponential.Prior(lengthscalePrior, signalPrior, noisePrior)
    let loglik = prior.DensityLn kernel

    let theoreticalLoglik = 0.0     // TODO
    loglik |> should (equalWithin 1e-9) theoreticalLoglik

[<Test>]
let ``Log likelihood is identical for parameters given individually and as a kernel`` () =
    let parameters = [| 3.5; 1.1; 0.7 |]
    let kernel = SquaredExponential.ofParameters parameters

    let lengthscalePrior = LogNormal.WithMeanVariance(1.1, 0.1)
    let signalPrior = LogNormal.WithMeanVariance(1.0, 0.5)
    let noisePrior = LogNormal.WithMeanVariance(0.1, 0.1)
    let prior = SquaredExponential.Prior(lengthscalePrior, signalPrior, noisePrior)

    let loglikKernel = prior.DensityLn kernel
    let loglikParams = prior.ParamsDensityLn parameters
    loglikKernel |> should (equalWithin 1e-9) loglikParams

let createSampleData n seed =
    let rnd = System.Random(seed)
    let xs = 
        [| for i in 0..n-1 -> rnd.NextDouble() * 10.0 |]
    let ys = xs |> Array.map (fun x -> (sin x) + Normal.Sample(rnd, 0.0, 0.1))
    {Locations = xs; Observations = ys}

let numericalDerivative f x =
    let epsilon = 1e-7
    (f (x + epsilon) - f x)/epsilon

/// Round number to a specified number of significant digits
let roundToSigDigits nDigits x =
    if x = 0.0 then 0.0
    else
        let scale = 10.0**((floor (log10 (abs x))) + 1.0 - float nDigits)
        scale * round(x/scale)

[<Test>]
let ``Log likelihood derivative wrt squared exponential kernel is correct`` () =
    // Set some values
    let n = 50
    let seed = 0
    let lengthscale = 5.1
    let signalVariance = 1.3
    let noiseVariance = 1.7
    let parameters = [| lengthscale; signalVariance; noiseVariance |]

    let data = [createSampleData n seed]
                   
    let se = parameters |> SquaredExponential.ofParameters
    let gp = GaussianProcess(se.Kernel, Some se.NoiseVariance)

    // Compute log likelihood with perturbed parameter
    let perturbLoglik paramIdx newValue =
        let seNew = 
            let ps = parameters
            ps.[paramIdx] <- newValue
            ps |> SquaredExponential.ofParameters
        GaussianProcess(seNew.Kernel, Some seNew.NoiseVariance).LogLikelihood data
    // Perturb a specific parameter and numerically compute gradient
    let numericalGradient idx = numericalDerivative (perturbLoglik idx) (parameters.[idx])

    // theoretical gradient
    let computedGradient = SquaredExponential.fullGradient data parameters

    // individual numerical gradients
    let lengthscaleGradient = numericalGradient 0
    let varianceGradient = numericalGradient 1    
    let noiseGradient = numericalGradient 2                
    
    let computedChecksum = 
        computedGradient |> Array.sum |> roundToSigDigits 3
    let expectedChecksum = 
        lengthscaleGradient + varianceGradient + noiseGradient
        |> roundToSigDigits 3

    computedChecksum |> should equal expectedChecksum
