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
#load "Utils.fs"
#else
module Ariadne.Tests.GaussianProcess
#endif

open NUnit.Framework
open FsUnit
open FsCheck

open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra

open Ariadne
open Ariadne.Tests.Utils
open Ariadne.GaussianProcess
open Ariadne.Kernels
open Ariadne.Optimization

open System

// -------------------------------------------
// Gaussian process
// -------------------------------------------

[<Test>]
let ``Gaussian log likelihood is correct for squared exponenital kernel`` () =
  let computedLoglik = 6.0  // TODO
  let expectedLoglik = 6.0

  computedLoglik |> should (equalWithin 1e-9) expectedLoglik

[<Test>]
let ``Covariance matrix with small noise is positive definite`` () =
    let kernel = SquaredExponential.SquaredExponential(1.1, 1.5, 1e-7)

    let checkCovarianceMatrix (inputs: float[]) =
        let data = 
            inputs |> Array.filter (fun x -> not (Double.IsInfinity(x)) && not (Double.IsNaN(x)))
        
        (data.Length > 0)
        ==> lazy
            let covMatrix = 
                covarianceMatrix kernel.Kernel data data
                |> addObservationNoise (Some kernel.NoiseVariance)
            let evd = covMatrix.Evd()
            evd.EigenValues |> Vector.toSeq
            |> Seq.map (fun e -> e.Real > 0.0)
            |> Seq.fold (&&) true
            |> should equal true

    Check.QuickThrowOnFailure checkCovarianceMatrix

[<Test>]
let ``Multivariate Normal log likelihood is computed correctly`` () = 
    let kernel = SquaredExponential.SquaredExponential(1.1, 1.5, 0.3)

    let checkMvnLoglik (values : float[]) = 
        let meanValues = validValues values

        (meanValues.Length > 0)
        ==> lazy
            let noiseDist = Normal.WithMeanVariance(0.0, 1.0, System.Random(0))
            let data = 
                meanValues |> Array.map (fun x -> x + noiseDist.Sample())
            let covMatrix = 
                covarianceMatrix kernel.Kernel data data
                |> addObservationNoise (Some kernel.NoiseVariance)
            let computedValue = 
                mvNormalLoglik (DenseVector.ofArray meanValues) covMatrix (DenseVector.ofArray data)

            let expectedValue = 
                (*
                // uncomment with new version of Math.NET
                let meanVector = DenseMatrix.ofColumnArrays [|meanValues|]
                let U = covMatrix
                let V = DenseMatrix.ofColumnArrays [| [|1.0|] |]
                let dist = MatrixNormal(meanVector, U, V)
                let dataMatrix = DenseMatrix.ofColumnArrays [|data|]
                log (dist.Density dataMatrix)
                *)
                // Direct computation
                let d = data.Length
                let diff = (DenseVector.ofArray data) - (DenseVector.ofArray meanValues)
                - (float d)/2.0 * log (2.0 * Math.PI)
                - 0.5 * log (covMatrix.Determinant())
                - 0.5 * (diff * (covMatrix.Inverse())) * diff
                
            computedValue |> should (equalWithin 1e-9) expectedValue    

    Check.QuickThrowOnFailure checkMvnLoglik

[<Test>]
let ``Posterior Gaussian process mean and covariance is computed correctly`` () = 
    let kernel = SquaredExponential.SquaredExponential(1.1, 1.5, 0.3)
    let gp = GaussianProcess(kernel.Kernel, ZeroMean, Some kernel.NoiseVariance)

    let checkPosteriorGP (observedValues : float[]) (newValues : float[]) =
        let locations = validValues observedValues
        let newLocations = validValues newValues
        
        (newLocations.Length > 0)
        ==> lazy
            if locations.Length > 0 then
                // Test using a direct implementation (which is ineffective)
                let covMatrix = 
                    covarianceMatrix kernel.Kernel locations locations
                    |> addObservationNoise (Some kernel.NoiseVariance)
                let invCovMatrix = covMatrix.Inverse()
                let newCovMatrix = 
                    covarianceMatrix kernel.Kernel newLocations newLocations
                let crossCovMatrix = 
                    covarianceMatrix kernel.Kernel newLocations locations
                let observations = 
                    let dist = Normal(0.0, 1.0,System.Random(1))
                    locations |> Array.map (fun x -> x + dist.Sample())
                    |> DenseVector.ofArray

                let expectedMeanSum = 
                    (crossCovMatrix * invCovMatrix) * observations
                    |> Vector.sum

                let expectedCovarianceSum = 
                    newCovMatrix - (crossCovMatrix * invCovMatrix) * crossCovMatrix.Transpose()
                    |> addObservationNoise (Some kernel.NoiseVariance)
                    |> Matrix.sum

                let data = [{Observations = (observations |> Vector.toArray); Locations = locations}]
                let computedMean, computedCovariance = 
                    gp.PosteriorGaussianProcess data newLocations 
                    
                computedMean |> Vector.sum |> roundToSigDigits 10
                |> should equal (expectedMeanSum |> roundToSigDigits 10)
                
                computedCovariance |> Matrix.sum |> roundToSigDigits 10
                |> should equal (expectedCovarianceSum |> roundToSigDigits 10)

    Check.QuickThrowOnFailure checkPosteriorGP

[<Test>]
let ``Gaussian process log likelihood is corrext``() = 
    // TODO: Check using GPML in Matlab
    1.0 |> should equal 1.0        
    
       

