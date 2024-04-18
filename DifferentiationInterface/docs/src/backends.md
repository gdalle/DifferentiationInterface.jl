```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

```@setup backends
using ADTypes
using DifferentiationInterface
using DifferentiationInterface: backend_string
import Markdown
import Diffractor, Enzyme, FastDifferentiation, FiniteDiff, FiniteDifferences, ForwardDiff, PolyesterForwardDiff, ReverseDiff, Tapir, Tracker, Zygote

function all_backends()
    return [
        AutoDiffractor(),
        AutoEnzyme(; mode=Enzyme.Forward),
        AutoEnzyme(; mode=Enzyme.Reverse),
        AutoFastDifferentiation(),
        AutoFiniteDiff(),
        AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1)),
        AutoForwardDiff(),
        AutoPolyesterForwardDiff(; chunksize=1),
        AutoReverseDiff(),
        AutoTapir(),
        AutoTracker(),
        AutoZygote(),
    ]
end
```

# Backends

## Types

We support all dense backend choices from [ADTypes.jl](https://github.com/SciML/ADTypes.jl).

```@docs
AutoChainRules
AutoDiffractor
AutoEnzyme
AutoFastDifferentiation
AutoForwardDiff
AutoFiniteDiff
AutoFiniteDifferences
AutoPolyesterForwardDiff
AutoReverseDiff
AutoSymbolics
AutoTapir
AutoTracker
AutoZygote
```

For sparse backends, only the Jacobian and Hessian operators are implemented differently, the other operators behave the same as for the corresponding dense backend.

```@docs
AutoSparse
```

## Availability

You can use [`check_available`](@ref) to verify whether a given backend is loaded, like we did below:

```@example backends
header = "| backend | available |"  # hide
subheader = "|:---|:---:|"  # hide
rows = map(all_backends()) do backend  # hide
    "| `$(backend_string(backend))` | $(check_available(backend) ? '✅' : '❌') |"  # hide
end  # hide
Markdown.parse(join(vcat(header, subheader, rows...), "\n"))  # hide
```

## Mutation support

All backends are compatible with one-argument functions `f(x) = y`.
Only some are compatible with two-argument functions `f!(y, x) = nothing`.
You can use [`check_twoarg`](@ref) to check that feature, like we did below:

```@example backends
header = "| backend | mutation |"  # hide
subheader = "|:---|:---:|"  # hide
rows = map(all_backends()) do backend  # hide
    "| `$(backend_string(backend))` | $(check_twoarg(backend) ? '✅' : '❌') |"  # hide
end  # hide
Markdown.parse(join(vcat(header, subheader, rows...), "\n"))  # hide
```

## Hessian support

Only some backends are able to compute Hessians.
You can use [`check_hessian`](@ref) to check that feature, like we did below:

```@example backends
header = "| backend | Hessian |"  # hide
subheader = "|:---|:---:|"  # hide
rows = map(all_backends()) do backend  # hide
    "| `$(backend_string(backend))` | $(check_hessian(backend) ? '✅' : '❌') |"  # hide
end  # hide
Markdown.parse(join(vcat(header, subheader, rows...), "\n"))  # hide
```
