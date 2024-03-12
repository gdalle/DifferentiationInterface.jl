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
