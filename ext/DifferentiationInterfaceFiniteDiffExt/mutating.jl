## Multiderivative

function DI.value_and_multiderivative!(
    y::AbstractArray,
    multider::AbstractArray,
    ::AutoFiniteDiff{fdtype},
    f,
    x::Number,
    extras::Nothing,
) where {fdtype}
    f!(y, x)
    finite_difference_gradient!(multider, f!, x, fdtype, eltype(y), FUNCTION_INPLACE, y)
    return y, multider
end

## Jacobian

function DI.value_and_jacobian!(
    y::AbstractVector,
    jac::AbstractMatrix,
    backend::AutoFiniteDiff,
    f,
    x::AbstractArray,
    extras::Nothing,
)
    f!(y, x)
    finite_difference_jacobian!(jac, f, x, fdtype, eltype(y))
    return y, jac
end
