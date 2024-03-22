"""
    prepare_pullback(f, backend, x) -> extras
    prepare_pullback(f!, backend, y, x) -> extras

Create an `extras` object that can be given to pullback operators.
"""
prepare_pullback(f, ::AbstractADType, x) = nothing
prepare_pullback(f!, ::AbstractADType, y, x) = nothing

"""
    prepare_pushforward(f, backend, x) -> extras
    prepare_pushforward(f!, backend, y, x) -> extras

Create an `extras` object that can be given to pushforward operators.
"""
prepare_pushforward(f, ::AbstractADType, x) = nothing
prepare_pushforward(f!, ::AbstractADType, y, x) = nothing

"""
    prepare_derivative(f, backend, x) -> extras
    prepare_derivative(f!, backend, y, x) -> extras

Create an `extras` object that can be given to derivative operators.
"""
prepare_derivative(f, ::AbstractADType, x) = nothing
prepare_derivative(f!, ::AbstractADType, y, x) = nothing

"""
    prepare_gradient(f, backend, x) -> extras

Create an `extras` object that can be given to gradient operators.
"""
prepare_gradient(f, ::AbstractADType, x) = nothing

"""
    prepare_jacobian(f, backend, x) -> extras
    prepare_jacobian(f!, backend, x, y) -> extras

Create an `extras` object that can be given to Jacobian operators.
"""
prepare_jacobian(f, ::AbstractADType, x) = nothing
prepare_jacobian(f!, ::AbstractADType, y, x) = nothing

"""
    prepare_second_derivative(f, backend, x) -> extras
    prepare_second_derivative(f!, backend, y, x) -> extras

Create an `extras` object that can be given to second derivative operators.
"""
prepare_second_derivative(f, ::AbstractADType, x) = nothing
prepare_second_derivative(f!, ::AbstractADType, y, x) = nothing

"""
    prepare_hessian_vector_product(f, backend, x) -> extras

Create an `extras` object that can be given to Hessian-vector product operators.
"""
prepare_hessian_vector_product(f, ::AbstractADType, x) = nothing

"""
    prepare_hessian(f, backend, x) -> extras

Create an `extras` object that can be given to Hessian operators.
"""
prepare_hessian(f, ::AbstractADType, x) = nothing
