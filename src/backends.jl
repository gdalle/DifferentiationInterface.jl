## Additional backends

# TODO: remove once https://github.com/SciML/ADTypes.jl/pull/21 is merged and released

"""
    AutoChainRules{RC}

Enables the use of AD libraries based on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl).

# Fields

- `ruleconfig::RC`: a [`RuleConfig`](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/superpowers/ruleconfig.html) object

# Example

```julia
using DifferentiationInterface, Zygote
backend = AutoChainRules(Zygote.ZygoteRuleConfig())
```
"""
struct AutoChainRules{RC} <: AbstractADType
    ruleconfig::RC
end

# TODO: remove this once https://github.com/SciML/ADTypes.jl/issues/27 is solved

"""
    AutoDiffractor

Enables the use of [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl).
"""
struct AutoDiffractor <: AbstractADType end

## Traits and access

"""
    autodiff_mode(backend)

Return `ForwardMode()` or `ReverseMode()` in a statically predictable way.

This function must be overloaded for backends that do not inherit from `ADTypes.AbstractForwardMode` or `ADTypes.AbstractReverseMode` (e.g. because they support both forward and reverse).

We classify `ADTypes.AbstractFiniteDifferencesMode` as forward mode.
"""
autodiff_mode(::AbstractForwardMode) = ForwardMode()
autodiff_mode(::AbstractFiniteDifferencesMode) = ForwardMode()
autodiff_mode(::AbstractReverseMode) = ReverseMode()

"""
    handles_input_type(backend, ::Type{X})

Check if `backend` can differentiate functions with input type `X`.
"""
handles_input_type(::AbstractADType, ::Type{<:Number}) = true
handles_input_type(::AbstractADType, ::Type{<:AbstractArray}) = true

"""
    handles_output_type(backend, ::Type{Y})

Check if `backend` can differentiate functions with output type `Y`.
"""
handles_output_type(::AbstractADType, ::Type{<:Number}) = true
handles_output_type(::AbstractADType, ::Type{<:AbstractArray}) = true

"""
    handles_types(backend, ::Type{X}, ::Type{Y})

Check if `backend` can differentiate functions with input type `X` and output type `Y`.
"""
function handles_types(backend::AbstractADType, ::Type{X}, ::Type{Y}) where {X,Y}
    return handles_input_type(backend, X) && handles_output_type(backend, Y)
end
