## Additional backend

# TODO: remove once https://github.com/SciML/ADTypes.jl/pull/21 is merged

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

ruleconfig(backend::AutoChainRules) = backend.ruleconfig

## Traits and access

"""
    autodiff_mode(backend)

Return `Val(:forward)` or `Val(:reverse)` in a statically predictable way.

This function must be overloaded for backends that do not inherit from `ADTypes.AbstractForwardMode` or `ADTypes.AbstractReverseMode` (e.g. because they support both forward and reverse).

We classify `ADTypes.AbstractFiniteDifferencesMode` as forward mode.
"""
autodiff_mode(::AbstractForwardMode) = Val{:forward}()
autodiff_mode(::AbstractFiniteDifferencesMode) = Val{:forward}()
autodiff_mode(::AbstractReverseMode) = Val{:reverse}()

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
