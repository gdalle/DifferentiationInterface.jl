```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

```@setup backends
using ADTypes, DifferentiationInterface
using DifferentiationInterfaceTest: backend_string
import Markdown
import Enzyme, FastDifferentiation, FiniteDiff, FiniteDifferences, ForwardDiff, PolyesterForwardDiff, ReverseDiff, Tracker, Zygote

function all_backends()
    return [
        AutoDiffractor(),
        AutoEnzyme(Enzyme.Forward),
        AutoEnzyme(Enzyme.Reverse),
        AutoFastDifferentiation(),
        AutoFiniteDiff(),
        AutoFiniteDifferences(FiniteDifferences.central_fdm(5, 1)),
        AutoForwardDiff(),
        AutoPolyesterForwardDiff(; chunksize=2),
        AutoReverseDiff(),
        AutoTracker(),
        AutoZygote(),
    ]
end

function all_backends_without_enzyme()
    return filter(all_backends()) do b
        !isa(b, AutoEnzyme)
    end
end
```

# Backends

## Types

Most backend choices are defined by [ADTypes.jl](https://github.com/SciML/ADTypes.jl).

!!! warning
    Only the backends listed here are supported by DifferentiationInterface.jl, even though ADTypes.jl defines more.

```@docs
AutoChainRules
AutoDiffractor
AutoEnzyme
AutoForwardDiff
AutoForwardDiff()
AutoFiniteDiff
AutoFiniteDifferences
AutoPolyesterForwardDiff
AutoPolyesterForwardDiff()
AutoReverseDiff
AutoTracker
AutoZygote
```

We also provide a few of our own:

```@docs
AutoFastDifferentiation
```

## Availability

You can use [`check_available`](@ref) to verify whether a given backend is loaded, like we did below:

```@example backends
header = "| backend | available |"  # hide
subheader = "|---|---|"  # hide
rows = map(all_backends()) do backend  # hide
    "| `$(backend_string(backend))` | $(check_available(backend) ? '✅' : '❌') |"  # hide
end  # hide
Markdown.parse(join(vcat(header, subheader, rows...), "\n"))  # hide
```

## Mutation support

All backends are compatible with allocating functions `f(x) = y`.
Only some are compatible with mutating functions `f!(y, x) = nothing`.
You can use [`check_mutation`](@ref) to check that feature, like we did below:

```@example backends
header = "| backend | mutation |"  # hide
subheader = "|---|---|"  # hide
rows = map(all_backends()) do backend  # hide
    "| `$(backend_string(backend))` | $(check_mutation(backend) ? '✅' : '❌') |"  # hide
end  # hide
Markdown.parse(join(vcat(header, subheader, rows...), "\n"))  # hide
```

## Package extensions

```@meta
CurrentModule = DifferentiationInterface
```

Backend-specific extension content is not part of the public API.

```@autodocs
Modules = [
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceChainRulesCoreExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceDiffractorExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceEnzymeExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceFastDifferentiationExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceFiniteDiffExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceFiniteDifferencesExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceForwardDiffExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfacePolyesterForwardDiffExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceReverseDiffExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceTrackerExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceZygoteExt)
]
Filter = t -> !(t isa Type && t <: ADTypes.AbstractADType)
```
