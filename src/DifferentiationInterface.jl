"""
    DifferentiationInterface

An experimental redesign for [AbstractDifferentiation.jl]
(https://github.com/JuliaDiff/AbstractDifferentiation.jl).

# Exports

$(EXPORTS)
"""
module DifferentiationInterface

using DocStringExtensions

abstract type AbstractBackend end
abstract type AbstractForwardBackend <: AbstractBackend end
abstract type AbstractReverseBackend <: AbstractBackend end

"""
    ChainRulesReverseBackend{RC}

Performs autodiff with reverse-mode AD packages based on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl), like [Zygote.jl](https://github.com/FluxML/Zygote.jl) or [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl).

This must be constructed with an appropriate [`RuleConfig`](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/superpowers/ruleconfig.html) instance:

```julia
using Zygote, DifferentiationInterface
backend = ChainRulesReverseBackend(Zygote.ZygoteRuleConfig())
```
"""
struct ChainRulesReverseBackend{RC} <: AbstractReverseBackend
    # TODO: check RC<:RuleConfig{>:HasReverseMode}
    ruleconfig::RC
end

"""
    ChainRulesForwardBackend{RC}

Performs autodiff with forward-mode AD packages based on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl).

This must be constructed with an appropriate [`RuleConfig`](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/superpowers/ruleconfig.html) instance.
"""
struct ChainRulesForwardBackend{RC} <: AbstractForwardBackend
    # TODO: check RC<:RuleConfig{>:HasForwardsMode}
    ruleconfig::RC
end

"""
    FiniteDiffBackend

Performs autodiff with [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl).
"""
struct FiniteDiffBackend <: AbstractForwardBackend end

"""
    EnzymeReverseBackend

Performs reverse-mode autodiff with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).
"""
struct EnzymeReverseBackend <: AbstractReverseBackend end

"""
    EnzymeForwardBackend

Performs forward-mode autodiff with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).
"""
struct EnzymeForwardBackend <: AbstractForwardBackend end

"""
    ForwardDiffBackend

Performs autodiff with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).
"""
struct ForwardDiffBackend <: AbstractForwardBackend end

"""
    ReverseDiffBackend

Performs autodiff with [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl).
"""
struct ReverseDiffBackend <: AbstractReverseBackend end

"""
    pushforward!(dy, backend, f, x, dx[, stuff])

Compute a Jacobian-vector product inside `dy` and return it.

# Arguments

- `dy`: cotangent, might be modified
- `backend`: forward-mode autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dx`: tangent
- `stuff`: optional backend-specific storage (cache, config), might be modified
"""
function pushforward! end

"""
    pullback!(dx, backend, f, x, dy[, stuff])

Compute a vector-Jacobian product inside `dx` and return it.

# Arguments

- `dx`: tangent, might be modified
- `backend`: reverse-mode autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dy`: cotangent
- `stuff`: optional backend-specific storage (cache, config), might be modified
"""
function pullback! end

export ChainRulesReverseBackend,
    ChainRulesForwardBackend,
    EnzymeReverseBackend,
    EnzymeForwardBackend,
    FiniteDiffBackend,
    ForwardDiffBackend,
    ReverseDiffBackend
export pushforward!, pullback!

end
