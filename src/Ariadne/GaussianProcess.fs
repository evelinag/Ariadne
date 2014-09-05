namespace Ariadne

// Ariadne: Gaussian process regression library


module GaussianProcess = 

    open MathNet.Numerics.LinearAlgebra
    open FSharp.Charting

    /// Type for data representation
    type Observation<'T> = 
        {
            Locations : 'T[]    // for example timepoints
            Observations : float []
        }

    type Kernel<'T> = 'T * 'T -> float

    // Mean function of the Gaussian process
    // - do I really need this? How does GPy deal with mean functions?
    type MeanFunction = 
        | ZeroMean

    /// Computes covariance matrix between two sets of inputs using
    /// a specified covariance function (kernel)
    let covarianceMatrix (kernel: Kernel<'T>) (input1: 'T []) (input2: 'T []) = 
        let n1 = Seq.length input1
        let n2 = Seq.length input2
        // TODO: speed this up for a special case?
        // use the fact that matrix should be symmetric if input1 = input2
        DenseMatrix.init n1 n2 (fun i j -> kernel (input1.[i], input2.[j]))          
        
    /// Add observational noise to a covariance matrix
    let addObservationNoise (noiseVariance:float option) (matrix: Matrix<float>) =
        match noiseVariance with
        | None -> matrix
        | Some(noise) ->
            if (matrix.ColumnCount <> matrix.RowCount) then 
                // Noise should be added only to a square covariance matrix
                failwith "Cannot add a noise term to a non-square covariance matrix."
            else
                let diagonalTerms = matrix.Diagonal()
                matrix.SetDiagonal (diagonalTerms + noise)
                matrix

    /// Compute log likelihood of a standard multivariate normal distribution
    let mvNormalLoglik (meanVector:Vector<float>) (covarianceMatrix:Matrix<float>) (x:Vector<float>) =
        let dimension = meanVector.Count
        if (x.Count <> dimension) || (covarianceMatrix.ColumnCount <> dimension && covarianceMatrix.RowCount <> dimension) then
            failwith "Multivariate normal likelihood: dimension mismatch"

        let choleskyCovMatrix = 
            try 
                let symmetricCovMatrix = (covarianceMatrix + covarianceMatrix.Transpose())/2.0
                symmetricCovMatrix.Cholesky()
            with e ->
                printfn "Cholesky decomposition: attempting to add preconditioning"   
                // TODO: Get eigenvalues and see how bad it is!
                let eigenvalues = covarianceMatrix.Evd().EigenValues.ToArray()
                let negativeEigenvals = eigenvalues |> Array.filter (fun x -> x.Real <= 0.0)
                printfn "Warning: %d negative eigenvalues: %A" (negativeEigenvals.Length) negativeEigenvals
                             
                // Preconditioning - adding small value to the diagonal
                let symmetricCovMatrix = 
                    (covarianceMatrix + covarianceMatrix.Transpose())/2.0
                    + DenseMatrix.diag dimension 0.0001
                symmetricCovMatrix.Cholesky()

        let difference = x - meanVector
        let interProduct = difference * (choleskyCovMatrix.Factor.Transpose().Inverse())

        - (float dimension)/2.0 * log (2.0 * System.Math.PI) 
        - 0.5 * choleskyCovMatrix.DeterminantLn
        - 0.5 * (interProduct * interProduct)

    type GaussianProcess<'T>
        (covariance : Kernel<'T>, meanFunction : MeanFunction, noiseVariance : float option) =
        member this.Mean = meanFunction
        member this.Covariance = covariance
        member this.NoiseVariance = noiseVariance

        new (covariance : Kernel<'T>, noiseVariance) =
            GaussianProcess(covariance, ZeroMean, noiseVariance)
            
        /// Computes mean vector and covariance matrix of a posterior Gaussian
        /// process given a set of observed data points
        /// and a set of new time points without observed values.
        member this.PosteriorGaussianProcess (data:Observation<'T> seq) (newLocations:'T []) = 
            if Seq.length data > 0 then
                // compute the posterior Gaussian process
                let allLocations = data |> Seq.map (fun d -> d.Locations) |> Array.concat
                let allObservations = data |> Seq.map (fun d -> d.Observations) |> Array.concat
                let observedCovariance = 
                    covarianceMatrix this.Covariance allLocations allLocations
                    |> addObservationNoise this.NoiseVariance

                let invCovariance = observedCovariance.Inverse()

                let newCovariance = covarianceMatrix this.Covariance newLocations newLocations
                let crossCovariance = covarianceMatrix this.Covariance newLocations allLocations

                let intermediateProduct = crossCovariance * invCovariance
                let newMean = intermediateProduct * (allObservations |> DenseVector.ofArray)
                let newCovariance = 
                    newCovariance - intermediateProduct * crossCovariance.Transpose()
                    |> addObservationNoise this.NoiseVariance
                newMean, newCovariance
            else
                // return prior Gaussian process in case no data are previously observed
                let newCovariance = 
                    covarianceMatrix this.Covariance newLocations newLocations 
                    |> addObservationNoise this.NoiseVariance
                let meanVector = DenseVector.zero newLocations.Length
                meanVector, newCovariance

        /// Computes posterior log likelihood of a Gaussian process
        /// given a set of observed data
        member this.LogLikelihood (data:Observation<'T> seq) =
            if Seq.length data = 0 then 0.0
            else
                let allLocations = data |> Seq.map (fun d -> d.Locations) |> Array.concat
                let allObservations = data |> Seq.map (fun d -> d.Observations) |> Array.concat
                let covMatrix = 
                    covarianceMatrix this.Covariance allLocations allLocations
                    |> addObservationNoise this.NoiseVariance

                let meanVector = 
                    match this.Mean with
                    | ZeroMean -> DenseVector.create (allLocations.Length) 0.0

                mvNormalLoglik meanVector covMatrix (allObservations |> DenseVector.ofArray)

        /// Predictive log likelihood of a Gaussian process
        /// TODO: Check numerically
        member this.PredictiveLogLikelihood (data:Observation<'T> seq) (x:Observation<'T>) = 
            let mean, covariance = this.PosteriorGaussianProcess data x.Locations
            let xObservations = x.Observations |> DenseVector.ofArray
            mvNormalLoglik mean covariance xObservations

        /// Predictive distribution of a Gaussian process given a set of observations
        ///
        /// ## Parameters
        /// - data - observed data 
        /// - timepoints - locations for prediction
        member this.Predict (data:Observation<'T> seq) (newLocations:'T[]) =
            let allLocations = data |> Seq.map (fun d -> d.Locations) |> Array.concat
            let allObservations = data |> Seq.map (fun d -> d.Observations) |> Array.concat
            
            let meanVector, covarianceMatrix = 
                this.PosteriorGaussianProcess data newLocations
            // meanVector is the mean of the posterior Gaussian process
            // variance - diagonal of the posterior covariance function
            let varianceVector = covarianceMatrix.Diagonal()
            meanVector, varianceVector


    /// Displays a Gaussian process regression curve given a set of data points
    /// Extrapolates the figure to [timeMin, timeMax] interval.
    /// Shows a region of +/- 1 standard deviations from the posterior mean.
    let plotRange (timeMin, timeMax) (data:Observation<float> seq) (gp:GaussianProcess<float>) =
        let step = (timeMax - timeMin)/100.0
        let t = [|timeMin .. step .. timeMax|]
        let meanVector, varianceVector = 
            gp.Predict data t

        let lower_s2 = meanVector - 1.0 * (varianceVector.Map (fun x -> sqrt x)) |> Array.ofSeq
        let upper_s2 = meanVector + 1.0 * (varianceVector.Map (fun x -> sqrt x)) |> Array.ofSeq

        let plotData =
            data |> Seq.map (fun dt -> Array.zip dt.Locations dt.Observations )
            |> Array.concat

        [ Chart.SplineRange(Array.zip3 t lower_s2 upper_s2, Color = System.Drawing.Color.LightGray);
            Chart.Line(Array.zip t (meanVector |> Array.ofSeq), Color=System.Drawing.Color.MediumBlue) 
            |> Chart.WithStyling(BorderWidth=3);
            Chart.Point(plotData, Color = System.Drawing.Color.Black) ]
        |> Chart.Combine
     
    /// Displays a Gaussian process regression curve given a set of data points
    /// Shows a region of +/- 1 standard deviations from the posterior mean.             
    let plot (data:Observation<float> seq) (gp:GaussianProcess<float>) =
        let allLocations = data |> Seq.map (fun d -> d.Locations) |> Array.concat
        let dataMin = Array.min allLocations
        let dataMax = Array.max allLocations
        let range = dataMax - dataMin

        let startLocation = dataMin - 0.15 * range
        let endLocation = dataMax + 0.15 * range

        plotRange (startLocation, endLocation) data gp
