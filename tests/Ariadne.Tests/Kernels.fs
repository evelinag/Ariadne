#if INTERACTIVE
#r "../../packages/NUnit.2.6.3/lib/nunit.framework.dll"
#r "../../packages/FsUnit.1.3.0.1/Lib/Net40/FsUnit.NUnit.dll"
#r "../../packages/FsCheck.1.0.0/lib/net45/FsCheck.dll"
#r "../../packages/MathNet.Numerics.3.2.1/lib/net40/MathNet.Numerics.dll"
#r "../../packages/MathNet.Numerics.FSharp.3.2.1/lib/net40/MathNet.Numerics.FSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"

#load "../../src/Ariadne/GaussianProcess.fs"
#load "../../src/Ariadne/Kernels.fs"
#load "../../src/Ariadne/Optimization.fs"
#else
module Ariadne.Tests.Kernels
#endif

open Ariadne.Tests.Utils
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
    (SquaredExp.ofParameters parameters).Parameters
    |> should equal parameters

[<Test>]
let ``Value of squared exponential kernel is correct`` () = 
    (* % Matlab code
    format long
    hyp = [log(3.5); log(sqrt(1.1))]; 
    x = [0; 1; -2; 3; 4]; y = [0; 1.5; -3; 7; -1];
    diag(covSEiso(hyp, x, y));
    *)
    let xs = [ 0.0; 1.0; -2.0; 3.0; 4.0 ]
    let ys = [ 0.0; 1.5; -3.0; 7.0; -1.0 ] 
    let kernel = SquaredExp.ofParameters [| 3.5; 1.1; 0.7 |]
    let values = List.map2 (fun x y -> kernel.Kernel (x,y)) xs ys
    let gpmlValues = [
       1.100000000000000;
       1.088832583716607;
       1.056005985414026;
       0.572495133122772;
       0.396492567457603 ]
    values |> should (equalWithin 1e-9) gpmlValues

[<Test>]
let ``Prior log likelihood is correct`` () =
    // Compute prior log likelihood numerically
    let parameters = [| 3.5; 1.1; 0.7 |]
    let kernel = SquaredExp.ofParameters parameters
    // Initialize LogNormal distributions with μ and σ
    let lengthscalePrior = LogNormal(0.1, 0.1)
    let signalPrior = LogNormal(1.0, 0.5)
    let noisePrior = LogNormal(-0.5, 0.1)
    let prior = SquaredExp.Prior(lengthscalePrior, signalPrior, noisePrior)
    let loglik = prior.DensityLn kernel

    // This is direct translation of probability density function computation 
    let logNormalLikelihood mu sigma x =
        let sigma2 = sigma*sigma 
        1.0/(x * sqrt (2.0 * System.Math.PI * sigma2)) * exp((-1.0/(2.0*sigma2)) * (log x - mu)*(log x - mu))
        |> log

    let theoreticalLoglik = 
        logNormalLikelihood lengthscalePrior.Mu lengthscalePrior.Sigma 3.5
        + logNormalLikelihood signalPrior.Mu signalPrior.Sigma 1.1
        + logNormalLikelihood noisePrior.Mu noisePrior.Sigma 0.7

    loglik |> should (equalWithin 1e-9) theoreticalLoglik

[<Test>]
let ``Log likelihood is identical for parameters given individually and as a kernel`` () =
    let parameters = [| 3.5; 1.1; 0.7 |]
    let kernel = SquaredExp.ofParameters parameters

    let lengthscalePrior = LogNormal.WithMeanVariance(1.1, 0.1)
    let signalPrior = LogNormal.WithMeanVariance(1.0, 0.5)
    let noisePrior = LogNormal.WithMeanVariance(0.1, 0.1)
    let prior = SquaredExp.Prior(lengthscalePrior, signalPrior, noisePrior)

    let loglikKernel = prior.DensityLn kernel
    let loglikParams = prior.ParamsDensityLn parameters
    loglikKernel |> should (equalWithin 1e-9) loglikParams

// ==========================
// Gradient computation 

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
                   
    let se = parameters |> SquaredExp.ofParameters
    let gp = GaussianProcess(se.Kernel, Some se.NoiseVariance)

    // Compute log likelihood with perturbed parameter
    let perturbLoglik paramIdx newValue =
        let seNew = 
            let ps = parameters
            ps.[paramIdx] <- newValue
            ps |> SquaredExp.ofParameters
        GaussianProcess(seNew.Kernel, Some seNew.NoiseVariance).LogLikelihood data
    // Perturb a specific parameter and numerically compute gradient
    let numericalGradient idx = numericalDerivative (perturbLoglik idx) (parameters.[idx])

    // theoretical gradient
    let computedGradient = 
        SquaredExp.fullGradient data parameters
        |> Array.map (roundToSigDigits 3)

    // individual numerical gradients
    let lengthscaleGradient = numericalGradient 0 |> roundToSigDigits 3
    let varianceGradient = numericalGradient 1 |> roundToSigDigits 3    
    let noiseGradient = numericalGradient 2 |> roundToSigDigits 3               
    
    computedGradient.[0] |> should equal lengthscaleGradient
    computedGradient.[1] |> should equal varianceGradient
    computedGradient.[2] |> should equal noiseGradient
