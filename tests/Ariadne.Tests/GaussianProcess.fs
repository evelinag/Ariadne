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
module Ariadne.Tests.GaussianProcess
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
// Gaussian process
// -------------------------------------------

[<Test>]
let ``Gaussian log likelihood is correct for squared exponenital kernel`` () =
  let computedLoglik = 6.0  // TODO
  let expectedLoglik = 6.0

  computedLoglik |> should (equalWithin 1e-9) expectedLoglik
//
//[<Test>]
//let ``Covariance matrix is positive definite`` () =
//    let inputs = 
//
//[<Test>]
//let ``Multivariate Normal log likelihood is computed correctly`` () = 
//    let meanVector = [|0.0; 1.0; -1.0; 3.0; -1.5|]
//    
//    
       

