"""
    prepare_pullback(backend, f, x) -> extras

Create an `extras` object that can be given to pullback operators.
"""
prepare_pullback(backend::AbstractADType, f, x) = nothing

"""
    prepare_pushforward(backend, f, x) -> extras

Create an `extras` object that can be given to pushforward operators.
"""
prepare_pushforward(backend::AbstractADType, f, x) = nothing

"""
    prepare_derivative(backend, f, x) -> extras

Create an `extras` object that can be given to derivative operators.
"""
prepare_derivative(backend::AbstractADType, f, x::Number) = nothing

"""
    prepare_multiderivative(backend, f, x) -> extras

Create an `extras` object that can be given to multiderivative operators.
"""
prepare_multiderivative(backend::AbstractADType, f, x::Number) = nothing

"""
    prepare_gradient(backend, f, x) -> extras

Create an `extras` object that can be given to gradient operators.
"""
prepare_gradient(backend::AbstractADType, f, x::AbstractArray) = nothing

"""
    prepare_jacobian(backend, f, x) -> extras

Create an `extras` object that can be given to jacobian operators.
"""
prepare_jacobian(backend::AbstractADType, f, x::AbstractArray) = nothing
