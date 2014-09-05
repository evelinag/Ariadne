namespace System
open System.Reflection

[<assembly: AssemblyTitleAttribute("Ariadne")>]
[<assembly: AssemblyProductAttribute("Ariadne")>]
[<assembly: AssemblyDescriptionAttribute("F# package for Gaussian process regression.")>]
[<assembly: AssemblyVersionAttribute("1.0")>]
[<assembly: AssemblyFileVersionAttribute("1.0")>]
do ()

module internal AssemblyVersionInformation =
    let [<Literal>] Version = "1.0"
