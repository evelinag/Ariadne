[<AutoOpen>]
module Ariadne.Kernels

// Kernel functions (covariance functions)

open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra

open Ariadne
open Ariadne.GaussianProcess

module SquaredExponential =
    /// Isotropic squared exponential covariance function
    /// also known as the Gaussian kernel or RBF kernel 
    /// 
    /// k(x,y) = σ² * exp(-1/(2*lengthscale²) * (x - y)²) + δ[x=y] σ²_{noise}
    ///
    /// σ² is the signal variance
    /// σ²_{noise} is the noise variance
    type SquaredExponential
        (lengthscale: float, signalVariance: float, noiseVariance: float) = 
        do if (lengthscale <= 0.0 || signalVariance <= 0.0 || noiseVariance <= 0.0) then
            failwith "Parameters of Squared exponential kernel must be positive."
        member this.Lengthscale = lengthscale
        member this.SignalVariance = signalVariance
        member this.NoiseVariance = noiseVariance

        with
            member this.Parameters = 
                [| this.Lengthscale; this.SignalVariance; this.NoiseVariance |]                

            override this.ToString() = 
                "Squared exponential, l = " 
                + this.Lengthscale.ToString("0.00") 
                + ", σ² = " + this.SignalVariance.ToString("0.00")
                + ", σ²_{noise} = " + this.NoiseVariance.ToString("0.00")
        
            /// Construct a kernel function that can be used in a Gaussian process 
            member this.Kernel (x1, x2) = 
                let exponent = (x1 - x2)**2.0 / (2.0 * this.Lengthscale**2.0) 
                this.SignalVariance * exp(-exponent)      
                
            /// Creates Gaussian process with squared exponential kernel function
            /// and zero mean.
            member this.GaussianProcess () =
                GaussianProcess.GaussianProcess(this.Kernel, Some this.NoiseVariance)            
            
    let ofParameters (parameters : float seq) =
        if Seq.length parameters <> 3 then 
            failwith "Squared exponential kernel requires 3 parameters."
        else
            let paramArray = parameters |> Seq.toArray
            SquaredExponential(paramArray.[0], paramArray.[1], paramArray.[2])


    type Prior
        (lengthscalePrior : LogNormal, signalVariancePrior : LogNormal, 
         noiseVariancePrior : LogNormal) = 
        
        member this.LengthscalePrior = lengthscalePrior
        member this.SignalVariancePrior = signalVariancePrior
        member this.NoiseVariancePrior = noiseVariancePrior

    with 
        /// Sample from the prior distribution for squared exponential kernel
        member this.Sample() =
            let l = this.LengthscalePrior.Sample()
            let v = this.SignalVariancePrior.Sample()
            let n = this.NoiseVariancePrior.Sample()
            SquaredExponential(l, v, n)

        /// Compute log likelihood of specific values of lengthscale and  under
        /// the prior distribution.
        member this.DensityLn (se:SquaredExponential) = 
            this.LengthscalePrior.DensityLn(se.Lengthscale)
            + this.SignalVariancePrior.DensityLn(se.SignalVariance)
            + this.NoiseVariancePrior.DensityLn(se.NoiseVariance)

        member this.ParamsDensityLn (parameters : float []) =
            if Array.length parameters <> 3 then 
                failwith "Number of parameters is incorrect for squared exponential."
            else
                this.LengthscalePrior.DensityLn(parameters.[0])
                + this.SignalVariancePrior.DensityLn(parameters.[1])
                + this.NoiseVariancePrior.DensityLn(parameters.[2])


    let randomWalkProposal (density : Normal []) (location : float []) =
        Array.map2 (fun (dist:Normal) (x:float) ->
            // sample new location x using distribution dist
            // and work in log space
            exp( log(x) + dist.Sample())) density location

    // Transition probability for Metropolis Hastings algorithm - symmetric
    let transitionProbability (density : Normal []) (oldLocation : float []) (newLocation : float []) =
        Array.zip3 density oldLocation newLocation
        |> Array.map (fun (dist, oldx, newx) ->
            Normal.WithMeanVariance(log oldx, dist.Variance).DensityLn(log newx))
        |> Array.sum

    // Compute derivative of log likelihood wrt parameters of the squared exponential
    // kernel
    let fullGradient (data: GaussianProcess.Observation<float> seq) (parameters : float[]) = 
        let allLocations = data |> Seq.map (fun d -> d.Locations) |> Array.concat
        let allObservations = 
            data |> Seq.map (fun d -> d.Observations) |> Array.concat
            |> DenseVector.ofArray

        let se = ofParameters parameters
        let covMatrix = 
            GaussianProcess.covarianceMatrix se.Kernel allLocations allLocations
            |> GaussianProcess.addObservationNoise (Some se.NoiseVariance)
        
        // Unstable
        let covMatrixInverse = covMatrix.Inverse()
        
        let gradientNoise = 
            - 0.5 * (covMatrixInverse).Trace() 
            + 0.5 * (allObservations * (covMatrixInverse * covMatrixInverse)) * allObservations

        let gradientSignal = 
            let dKernel (x, y) =
                exp ( - ((x - y)*(x - y))/(2.0*se.Lengthscale**2.0))
            let dCovMatrix = 
                GaussianProcess.covarianceMatrix dKernel allLocations allLocations

            let prod = covMatrixInverse * dCovMatrix
            - 0.5 * prod.Trace()
            + 0.5 * (allObservations * (prod * covMatrixInverse)) * allObservations
    
        let gradientLengthscale = 
            let dKernel (x,y) = 
                se.SignalVariance * exp(- ((x - y)*(x - y))/(2.0 * se.Lengthscale*se.Lengthscale))
                * (x - y)*(x - y) * se.Lengthscale**(-3.0)
            let dCovMatrix = 
                GaussianProcess.covarianceMatrix dKernel allLocations allLocations

            let prod = covMatrixInverse * dCovMatrix
            - 0.5 * prod.Trace()
            + 0.5 * ((allObservations * prod) * covMatrixInverse) * allObservations

        [| gradientLengthscale; gradientSignal; gradientNoise |]
