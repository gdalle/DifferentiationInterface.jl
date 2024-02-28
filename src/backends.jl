"""
    ChainRulesReverseBackend{RC}

Performs autodiff with reverse-mode AD packages based on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl), like [Zygote.jl](https://github.com/FluxML/Zygote.jl).

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

function Base.string(backend::ChainRulesReverseBackend)
    return "ChainRulesReverseBackend($(backend.ruleconfig))"
end

"""
    ChainRulesForwardBackend{RC}

Performs autodiff with forward-mode AD packages based on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl), like [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl).

This must be constructed with an appropriate [`RuleConfig`](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/superpowers/ruleconfig.html) instance.
```julia
using Diffractor, DifferentiationInterface
backend = ChainRulesForwardBackend(Diffractor.DiffractorRuleConfig())
```
"""
struct ChainRulesForwardBackend{RC} <: AbstractForwardBackend
    # TODO: check RC<:RuleConfig{>:HasForwardsMode}
    ruleconfig::RC
end

function Base.string(backend::ChainRulesForwardBackend)
    return "ChainRulesForwardBackend($(backend.ruleconfig))"
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
