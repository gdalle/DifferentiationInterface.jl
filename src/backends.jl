"""
$(TYPEDEF)

Performs autodiff with forward mode AD packages based on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl), like [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl).

This must be constructed with an appropriate [`RuleConfig`](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/superpowers/ruleconfig.html) instance.
```julia
using Diffractor, DifferentiationInterface
backend = ChainRulesForwardBackend(Diffractor.DiffractorRuleConfig())
```

# Fields

$(TYPEDFIELDS)
"""
struct ChainRulesForwardBackend{custom,RC} <: AbstractForwardBackend{custom}
    # TODO: check RC<:RuleConfig{>:HasForwardsMode}
    ruleconfig::RC
end

function Base.show(io::IO, backend::ChainRulesForwardBackend{custom}) where {custom}
    return print(
        io,
        "ChainRulesForwardBackend{$(custom ? "custom" : "fallback")}($(backend.ruleconfig))",
    )
end

"""
$(TYPEDEF)

Performs autodiff with reverse mode AD packages based on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl), like [Zygote.jl](https://github.com/FluxML/Zygote.jl).

This must be constructed with an appropriate [`RuleConfig`](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/superpowers/ruleconfig.html) instance:

```julia
using Zygote, DifferentiationInterface
backend = ChainRulesReverseBackend(Zygote.ZygoteRuleConfig())
```

# Fields

$(TYPEDFIELDS)
"""
struct ChainRulesReverseBackend{custom,RC} <: AbstractReverseBackend{custom}
    # TODO: check RC<:RuleConfig{>:HasReverseMode}
    ruleconfig::RC
end

function Base.show(io::IO, backend::ChainRulesReverseBackend{custom}) where {custom}
    return print(
        io,
        "ChainRulesReverseBackend{$(custom ? "custom" : "fallback")}($(backend.ruleconfig))",
    )
end

"""
$(TYPEDEF)

Performs autodiff with [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl).

The type parameter `fdtype` determines the type of finite differences used, it defaults to `Val{:central}`.
"""
struct FiniteDiffBackend{custom,fdtype} <: AbstractForwardBackend{custom} end

function Base.show(io::IO, ::FiniteDiffBackend{custom,fdtype}) where {custom,fdtype}
    return print(io, "FiniteDiffBackend{$(custom ? "custom" : "fallback"),$fdtype}()")
end

"""
$(TYPEDEF)

Performs forward-mode autodiff with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).
"""
struct EnzymeForwardBackend{custom} <: AbstractForwardBackend{custom} end

function Base.show(io::IO, ::EnzymeForwardBackend{custom}) where {custom}
    return print(io, "EnzymeForwardBackend{$(custom ? "custom" : "fallback")}()")
end

"""
$(TYPEDEF)

Performs reverse-mode autodiff with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).

!!! warning
    This backend only works for scalar output.
"""
struct EnzymeReverseBackend{custom} <: AbstractReverseBackend{custom} end

function Base.show(io::IO, ::EnzymeReverseBackend{custom}) where {custom}
    return print(io, "EnzymeReverseBackend{$(custom ? "custom" : "fallback")}()")
end

"""
$(TYPEDEF)

Performs autodiff with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).
"""
struct ForwardDiffBackend{custom} <: AbstractForwardBackend{custom} end

function Base.show(io::IO, ::ForwardDiffBackend{custom}) where {custom}
    return print(io, "ForwardDiffBackend{$(custom ? "custom" : "fallback")}()")
end

"""
$(TYPEDEF)

Performs autodiff with [PolyesterForwardDiff.jl](https://github.com/JuliaDiff/PolyesterForwardDiff.jl), falling back on [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) if needed.

The type parameter `C` is an integer configuring chunk size.

!!! warning
    This backend only works when the arrays are vectors.
"""
struct PolyesterForwardDiffBackend{custom,C} <: AbstractForwardBackend{custom} end

function Base.show(io::IO, ::PolyesterForwardDiffBackend{custom,C}) where {custom,C}
    return print(io, "PolyesterForwardDiffBackend{$(custom ? "custom" : "fallback"),$C}()")
end

"""
$(TYPEDEF)

Performs autodiff with [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl).

!!! warning
    This backend only works for array input.
"""
struct ReverseDiffBackend{custom} <: AbstractReverseBackend{custom} end

function Base.show(io::IO, ::ReverseDiffBackend{custom}) where {custom}
    return print(io, "ReverseDiffBackend{$(custom ? "custom" : "fallback")}()")
end

## Pseudo backends

"""
    ZygoteBackend

Performs autodiff with [Zygote.jl](https://github.com/FluxML/Zygote.jl).

## Note

This is not a type but a function because it actually constructs a [`ChainRulesReverseBackend`](@ref) with `ZygoteRuleConfig()`.
"""
function ZygoteBackend end

## Limitations

handles_input_type(::ReverseDiffBackend, ::Type{<:Number}) = false
handles_output_type(::EnzymeReverseBackend, ::Type{<:AbstractArray}) = false
