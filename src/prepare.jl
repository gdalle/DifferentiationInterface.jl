"""
    prepare_pullback(backend, f, x) -> extras
    prepare_pullback(backend, f!, x, y) -> extras

Create an `extras` object that can be given to pullback operators.
"""
function prepare_pullback end

prepare_pullback(::AbstractADType, f, x) = nothing
prepare_pullback(::AbstractADType, f!, x, y) = nothing

"""
    prepare_pushforward(backend, f, x) -> extras
    prepare_pushforward(backend, f!, x, y) -> extras

Create an `extras` object that can be given to pushforward operators.
"""
function prepare_pushforward end

prepare_pushforward(::AbstractADType, f, x) = nothing
prepare_pushforward(::AbstractADType, f!, x, y) = nothing

"""
    prepare_derivative(backend, f, x) -> extras

Create an `extras` object that can be given to derivative operators.
"""
function prepare_derivative end

prepare_derivative(::AbstractADType, f, x::Number) = nothing

"""
    prepare_multiderivative(backend, f, x) -> extras
    prepare_multiderivative(backend, f!, x, y) -> extras

Create an `extras` object that can be given to multiderivative operators.
"""
function prepare_multiderivative end

prepare_multiderivative(::AbstractADType, f, x::Number) = nothing
prepare_multiderivative(::AbstractADType, f!, x::Number, y::AbstractArray) = nothing

"""
    prepare_gradient(backend, f, x) -> extras

Create an `extras` object that can be given to gradient operators.
"""
function prepare_gradient end

prepare_gradient(::AbstractADType, f, x::AbstractArray) = nothing

"""
    prepare_jacobian(backend, f, x) -> extras
    prepare_jacobian(backend, f!, x, y) -> extras

Create an `extras` object that can be given to jacobian operators.
"""
function prepare_jacobian end

prepare_jacobian(::AbstractADType, f, x::AbstractArray) = nothing
prepare_jacobian(::AbstractADType, f!, x::AbstractArray, y::AbstractArray) = nothing
