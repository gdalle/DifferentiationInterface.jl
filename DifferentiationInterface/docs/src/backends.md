# Backends

DifferentiationInterface.jl is based on two concepts: **operators** and **backends**.
This page is about the latter, check out [that page](@ref "Operators") to learn about the former.

## List of backends

We support the following dense backend choices from [ADTypes.jl](https://github.com/SciML/ADTypes.jl):

- [`AutoChainRules`](@extref ADTypes.AutoChainRules)
- [`AutoDiffractor`](@extref ADTypes.AutoDiffractor)
- [`AutoEnzyme`](@extref ADTypes.AutoEnzyme)
- [`AutoFastDifferentiation`](@extref ADTypes.AutoFastDifferentiation)
- [`AutoFiniteDiff`](@extref ADTypes.AutoFiniteDiff)
- [`AutoFiniteDifferences`](@extref ADTypes.AutoFiniteDifferences)
- [`AutoForwardDiff`](@extref ADTypes.AutoForwardDiff)
- `AutoGTPSA`
- [`AutoPolyesterForwardDiff`](@extref ADTypes.AutoPolyesterForwardDiff)
- [`AutoReverseDiff`](@extref ADTypes.AutoReverseDiff)
- [`AutoSymbolics`](@extref ADTypes.AutoSymbolics)
- [`AutoTapir`](@extref ADTypes.AutoTapir)
- [`AutoTracker`](@extref ADTypes.AutoTracker)
- [`AutoZygote`](@extref ADTypes.AutoZygote)

We also support the sparse wrapper [`AutoSparse`](@extref ADTypes.AutoSparse).

## Compatibility

DifferentiationInterface.jl itself is compatible with Julia 1.6, the Long Term Support (LTS) version of the language.
However, we were only able to test the following backends on Julia 1.6:

- `AutoFiniteDifferences`
- `AutoForwardDiff`
- `AutoReverseDiff`
- `AutoTracker`
- `AutoZygote`

We strongly recommend that users upgrade to Julia 1.10 or above, where all backends are tested.

## Features

Given a backend object, you can use:

- [`check_available`](@ref) to know whether the required AD package is loaded
- [`check_inplace`](@ref) to know whether the backend supports in-place functions (all backends support out-of-place functions)

```@setup backends
using ADTypes
using DifferentiationInterface
import Markdown

import ChainRulesCore
import Diffractor
import Enzyme
import FastDifferentiation
import FiniteDiff
import FiniteDifferences
import ForwardDiff
import GTPSA
import PolyesterForwardDiff
import ReverseDiff
import Symbolics
import Tapir
import Tracker
import Zygote

backend_examples = [
    AutoChainRules(; ruleconfig=Zygote.ZygoteRuleConfig()),
    AutoDiffractor(),
    AutoEnzyme(),
    AutoFastDifferentiation(),
    AutoFiniteDiff(),
    AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1)),
    AutoForwardDiff(),
    AutoGTPSA(),
    AutoPolyesterForwardDiff(; chunksize=1),
    AutoReverseDiff(),
    AutoSymbolics(),
    AutoTapir(; safe_mode=false),
    AutoTracker(),
    AutoZygote(),
]

checkmark(x::Bool) = x ? '✅' : '❌'
unicode_check_available(backend) = checkmark(check_available(backend))
unicode_check_inplace(backend)    = checkmark(check_inplace(backend))

io = IOBuffer()

# Table header 
println(io, "| Backend | Availability | In-place functions |")
println(io, "|:--------|:------------:|:----------------------:|")

for b in backend_examples
    join(io, ["`$(nameof(typeof(b)))`", unicode_check_available(b), unicode_check_inplace(b)], '|')
    println(io, '|' )
end
backend_table = Markdown.parse(String(take!(io)))
```

```@example backends
backend_table #hide
```

## Backend switch

The wrapper [`DifferentiateWith`](@ref) allows you to switch between backends.
It takes a function `f` and specifies that `f` should be differentiated with the backend of your choice, instead of whatever other backend the code is trying to use.
In other words, when someone tries to differentiate `dw = DifferentiateWith(f, backend1)` with `backend2`, then `backend1` steps in and `backend2` does nothing.
At the moment, `DifferentiateWith` only works when `backend2` supports [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl).
