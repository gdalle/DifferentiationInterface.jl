"""
    prepare_pullback(backend, f, x) -> extras
    prepare_pullback(backend, f!, x, y) -> extras

Create an `extras` object that can be given to pullback operators.
"""
prepare_pullback(::AbstractADType, f, x) = nothing
prepare_pullback(::AbstractADType, f!, x, y) = nothing

"""
    prepare_pushforward(backend, f, x) -> extras
    prepare_pushforward(backend, f!, x, y) -> extras

Create an `extras` object that can be given to pushforward operators.
"""
prepare_pushforward(::AbstractADType, f, x) = nothing
prepare_pushforward(::AbstractADType, f!, x, y) = nothing

"""
    prepare_derivative(backend, f, x) -> extras

Create an `extras` object that can be given to derivative operators.
"""
prepare_derivative(::AbstractADType, f, x::Number) = nothing

"""
    prepare_multiderivative(backend, f, x) -> extras
    prepare_multiderivative(backend, f!, x, y) -> extras

Create an `extras` object that can be given to multiderivative operators.
"""
prepare_multiderivative(::AbstractADType, f, x::Number) = nothing
prepare_multiderivative(::AbstractADType, f!, x::Number, y::AbstractArray) = nothing

"""
    prepare_gradient(backend, f, x) -> extras

Create an `extras` object that can be given to gradient operators.
"""
prepare_gradient(::AbstractADType, f, x::AbstractArray) = nothing

"""
    prepare_jacobian(backend, f, x) -> extras
    prepare_jacobian(backend, f!, x, y) -> extras

Create an `extras` object that can be given to Jacobian operators.
"""
prepare_jacobian(::AbstractADType, f, x::AbstractArray) = nothing
prepare_jacobian(::AbstractADType, f!, x::AbstractArray, y::AbstractArray) = nothing

"""
    prepare_second_derivative(backend, f, x) -> extras

Create an `extras` object that can be given to second derivative operators.
"""
prepare_second_derivative(::AbstractADType, f, x::Number) = nothing

"""
    prepare_hessian(backend, f, x) -> extras

Create an `extras` object that can be given to Hessian operators.
"""
prepare_hessian(::AbstractADType, f, x::AbstractArray) = nothing

"""
    prepare_hessian_vector_product(backend, f, x) -> extras

Create an `extras` object that can be given to Hessian-vector product operators.
"""
prepare_hessian_vector_product(::AbstractADType, f, x::AbstractArray) = nothing
