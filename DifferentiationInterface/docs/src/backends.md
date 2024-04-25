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

const backend_examples = (
    "AutoDiffractor()",
    "AutoEnzyme(; mode=Enzyme.Forward)",
    "AutoEnzyme(; mode=Enzyme.Reverse)",
    "AutoFastDifferentiation()",
    "AutoFiniteDiff()",
    "AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1))",
    "AutoForwardDiff()",
    "AutoPolyesterForwardDiff(; chunksize=1)",
    "AutoReverseDiff()",
    "AutoTapir()",
    "AutoTracker()",
    "AutoZygote()",
)

checkmark(x::Bool) = x ? '✅' : '❌'
unicode_check_available(backend) = checkmark(check_available(backend))
unicode_check_hessian(backend)   = checkmark(check_hessian(backend))
unicode_check_twoarg(backend)    = checkmark(check_twoarg(backend))

io = IOBuffer()

# Table header 
println(io, "| Backend | Availability | Two-argument functions | Hessian support | Example |")
println(io, "|:--------|:------------:|:----------------------:|:---------------:|:--------|")

for example in backend_examples
    b = eval(Meta.parse(example)) # backend
    join(io, [backend_string(b), unicode_check_available(b), unicode_check_twoarg(b), unicode_check_hessian(b), "`$example`"], '|')
    println(io, '|' )
end
backend_table = Markdown.parse(String(take!(io)))
```

# Backends

## Types

We support all dense backend choices from [ADTypes.jl](https://github.com/SciML/ADTypes.jl), as well as their sparse wrapper `AutoSparse`.

For sparse backends, only the Jacobian and Hessian operators are implemented differently, the other operators behave the same as for the corresponding dense backend.

```@example backends
backend_table #hide
```

## Availability

You can use [`check_available`](@ref) to verify whether a given backend is loaded.

## Support for two-argument functions

All backends are compatible with one-argument functions `f(x) = y`.
Only some are compatible with two-argument functions `f!(y, x) = nothing`.
You can check this compatibility using [`check_twoarg`](@ref).

## Hessian support

Only some backends are able to compute Hessians.
You can use [`check_hessian`](@ref) to check this feature.
