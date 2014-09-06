module Ariadne.Tests.Utils

open System
open Ariadne.GaussianProcess
open Ariadne.Kernels
open Ariadne.Optimization
open MathNet.Numerics.Distributions

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

let validValues xs = 
    xs |> Array.filter (fun x -> 
            not (Double.IsInfinity(x)) && not (Double.IsNaN(x))
            && x < 1e+308)
 
