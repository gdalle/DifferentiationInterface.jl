# Backends

DifferentiationInterface.jl is based on two concepts: **operators** and **backends**.
This page is about the latter, check out [that page](@ref "Operators") to learn about the former.

## List of backends

We support all dense backend choices from [ADTypes.jl](https://github.com/SciML/ADTypes.jl), as well as their sparse wrapper [`AutoSparse`](@extref ADTypes.AutoSparse).

```@setup backends
using ADTypes
using DifferentiationInterface
import Markdown

import Diffractor
import Enzyme
import FastDifferentiation
import FiniteDiff
import FiniteDifferences
import ForwardDiff
import PolyesterForwardDiff
import ReverseDiff
import Symbolics
import Tapir
import Tracker
import Zygote

backend_examples = [
    AutoDiffractor(),
    AutoEnzyme(),
    AutoFastDifferentiation(),
    AutoFiniteDiff(),
    AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1)),
    AutoForwardDiff(),
    AutoPolyesterForwardDiff(; chunksize=1),
    AutoReverseDiff(),
    AutoSymbolics(),
    AutoTapir(; safe_mode=false),
    AutoTracker(),
    AutoZygote(),
]

checkmark(x::Bool) = x ? '✅' : '❌'
unicode_check_available(backend) = checkmark(check_available(backend))
unicode_check_hessian(backend)   = checkmark(check_hessian(backend; verbose=false))
unicode_check_twoarg(backend)    = checkmark(check_twoarg(backend))

io = IOBuffer()

# Table header 
println(io, "| Backend | Availability | Two-argument functions | Hessian support |")
println(io, "|:--------|:------------:|:----------------------:|:---------------:|")

for b in backend_examples
    join(io, ["[`$(nameof(typeof(b)))`](@extref ADTypes)", unicode_check_available(b), unicode_check_twoarg(b), unicode_check_hessian(b)], '|')
    println(io, '|' )
end
backend_table = Markdown.parse(String(take!(io)))
```

```@example backends
backend_table #hide
```

## Compatibility

DifferentiationInterface.jl itself is compatible with Julia 1.6, the Long Term Support (LTS) version of the language.
However, we were only able to test the following backends on Julia 1.6:

- FiniteDifferences.jl
- ForwardDiff.jl
- ReverseDiff.jl
- Tracker.jl
- Zygote.jl

We strongly recommend that users upgrade to Julia 1.10, where all backends are tested.

## Checks

### Availability

You can use [`check_available`](@ref) to verify whether a given backend is loaded.

### Support for two-argument functions

All backends are compatible with one-argument functions `f(x) = y`.
Only some are compatible with two-argument functions `f!(y, x) = nothing`.
You can use [`check_twoarg`](@ref) to verify this compatibility.

### Support for Hessian

Only some backends are able to compute Hessians.
You can use [`check_hessian`](@ref) to verify this feature (beware that it will try to compute a small Hessian, so it is not instantaneous like the other checks).

## Backend switch

The wrapper [`DifferentiateWith`](@ref) allows you to switch between backends.
It takes a function `f` and specifies that `f` should be differentiated with the backend of your choice, instead of whatever other backend the code is trying to use.
In other words, when someone tries to differentiate `dw = DifferentiateWith(f, backend1)` with `backend2`, then `backend1` steps in and `backend2` does nothing.
At the moment, `DifferentiateWith` only works when `backend2` supports [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl).
