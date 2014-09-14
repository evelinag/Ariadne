(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
#I "../../bin"

(**
Ariadne: F# library for Gaussian process regression
===================

Ariadne is an F# library for fitting Gaussian process regression models.

![Example GP](img/gp.png) 

How to get Ariadne
--------------------------

Ariadne is available from [NuGet](https://nuget.org/packages/Ariadne).
 
What are Gaussian processes?
-------------------------------------
Let's start with an example. Assume we have a set of noisy time series 
observations, like the one below. We would like to model relationship
between the input values and the outputs. 

![Sample data](img/sampleData.png)

When using standard regression models, like linear regression, we 
have to assume there is a specific parametric relation between 
the observed data. 

With Gaussian process regression, we have to use a different types
of assumptions. In this case, we will assume that the underlying
function is quite smooth with some amount of noise. 
What Gaussian processes give us is a flexible regression function 
which takes account of noise in observed data and provides
a measure of uncertainty in the regressed values. 

In the example below,  the blue line represents the predicted mean value of 
the estimated function. 
The grey region shows the distance of +/- two standard deviations from the mean,
which corresponds to 95% confidence interval.
This quantity shows how uncertain the model is about value of the 
function. Note that the grey area is wider in locations where 
there are no observations, and shrinks when a value is observed.

![Sample Gaussian process](img/sampleGP.png)

If you want to learn more about Gaussian processes, the best resource is the
[Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/) book.

Samples & documentation
-----------------------

 * [Tutorial](tutorial.html) shows how to fit a Gaussian process model using Ariadne.

 * [Covariance functions](covarianceFunctions.html) gives an overview of implemented covariance functions.

 * [Optimization](optimization.html) shows implemented methods which can be used to fit hyperparameters 
 of Gaussican processes.

 * [Quick how-to](howto.html) is a very brief overview of functions available in Ariadne.

 * [API Reference](reference/index.html) contains automatically generated documentation for all types, modules
   and functions in the library. 

Information on Gaussian processes
-------------------------
 * [Gaussian Processes for Machine Learning](http://gaussianprocess.org/gpml/) is a great
   book by Carl E. Rasmussen and Christopher K. I. Williams. It is available online for free.

 * [The Kernel Cookbook](http://mlg.eng.cam.ac.uk/duvenaud/cookbook/index.html) is a nice
   overview of different kernel functions that can be used in conjuction with Gaussian processes.
 
Contributing and copyright
--------------------------

The project is hosted on [GitHub][gh] where you can [report issues][issues], fork 
the project and submit pull requests. 
The library is available under Apache 2.0 license, which allows modification and 
redistribution for both commercial and non-commercial purposes. For more information see the 
[License file][license] in the GitHub repository. 

  [content]: https://github.com/evelinag/Ariadne/tree/master/docs/content
  [gh]: https://github.com/evelinag/Ariadne
  [issues]: https://github.com/evelinag/Ariadne/issues
  [license]: https://github.com/evelinag/Ariadne/blob/master/LICENSE
*)
