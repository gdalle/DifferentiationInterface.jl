```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

```@setup backends
using DifferentiationInterface
using DifferentiationInterface: backend_str
import Markdown

# import Diffractor
import Enzyme
# import FastDifferentiation
import FiniteDiff
import FiniteDifferences
import ForwardDiff
# import PolyesterForwardDiff
import ReverseDiff
# import Symbolics
# import Tapir
import Tracker
import Zygote

const backend_examples = (
    # "AutoDiffractor()",
    "AutoEnzyme(; mode=Enzyme.Forward)",
    "AutoEnzyme(; mode=Enzyme.Reverse)",
    # "AutoFastDifferentiation()",
    "AutoFiniteDiff()",
    "AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1))",
    "AutoForwardDiff()",
    # "AutoPolyesterForwardDiff(; chunksize=1)",
    "AutoReverseDiff()",
    # "AutoSymbolics()",
    # "AutoTapir()",
    "AutoTracker()",
    "AutoZygote()",
)

checkmark(x::Bool) = x ? '✅' : '❌'
unicode_check_available(backend) = checkmark(check_available(backend))
unicode_check_hessian(backend)   = checkmark(check_hessian(backend; verbose=false))
unicode_check_twoarg(backend)    = checkmark(check_twoarg(backend))

io = IOBuffer()

# Table header 
println(io, "| Backend | Availability | Two-argument functions | Hessian support | Example |")
println(io, "|:--------|:------------:|:----------------------:|:---------------:|:--------|")

for example in backend_examples
    b = eval(Meta.parse(example)) # backend
    join(io, [backend_str(b), unicode_check_available(b), unicode_check_twoarg(b), unicode_check_hessian(b), "`$example`"], '|')
    println(io, '|' )
end
backend_table = Markdown.parse(String(take!(io)))
```

# Backends

## Types

We support all dense backend choices from [ADTypes.jl](https://github.com/SciML/ADTypes.jl), as well as their sparse wrapper [`AutoSparse`](@ref).

For sparse backends, only the Jacobian and Hessian operators are implemented differently, the other operators behave the same as for the corresponding dense backend.

```@example backends
backend_table #hide
```

## Checks

### Availability

You can use [`check_available`](@ref) to verify whether a given backend is loaded.

### Support for two-argument functions

All backends are compatible with one-argument functions `f(x) = y`.
Only some are compatible with two-argument functions `f!(y, x) = nothing`.
You can check this compatibility using [`check_twoarg`](@ref).

### Support for Hessian

Only some backends are able to compute Hessians.
You can use [`check_hessian`](@ref) to check this feature (beware that it will try to compute a small Hessian, so it is not instantaneous).

## API reference

!!! warning
    The following documentation has been borrowed from ADTypes.jl.
    Refer to the [ADTypes documentation](https://sciml.github.io/ADTypes.jl/stable/) for more information.

```@docs
ADTypes
ADTypes.AbstractADType
AutoChainRules
AutoDiffractor
AutoEnzyme
AutoFastDifferentiation
AutoFiniteDiff
AutoFiniteDifferences
AutoForwardDiff
AutoPolyesterForwardDiff
AutoReverseDiff
AutoSymbolics
AutoTapir
AutoTracker
AutoZygote
AutoSparse
```
