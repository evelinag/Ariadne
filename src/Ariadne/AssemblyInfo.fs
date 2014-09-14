namespace System
open System.Reflection

[<assembly: AssemblyTitleAttribute("Ariadne")>]
[<assembly: AssemblyProductAttribute("Ariadne")>]
[<assembly: AssemblyDescriptionAttribute("F# package for Gaussian process regression.")>]
[<assembly: AssemblyVersionAttribute("0.1.1")>]
[<assembly: AssemblyFileVersionAttribute("0.1.1")>]
do ()

module internal AssemblyVersionInformation =
    let [<Literal>] Version = "0.1.1"
