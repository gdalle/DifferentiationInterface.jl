## Pushforward

function DI.value_and_pushforward!(
    y::AbstractArray,
    dy::AbstractArray,
    ::AutoFiniteDiff{fdtype},
    f!,
    x,
    dx,
    extras::Nothing,
) where {fdtype}
    function step(t::Number)::AbstractArray
        new_y = similar(y)
        f!(new_y, x .+ t .* dx)
        return new_y
    end
    finite_difference_gradient!(
        dy, step, zero(eltype(dx)), fdtype, eltype(y), FUNCTION_NOT_INPLACE, y
    )
    f!(y, x)
    return y, dy
end

## Multiderivative

function DI.value_and_multiderivative!(
    y::AbstractArray,
    multider::AbstractArray,
    ::AutoFiniteDiff{fdtype},
    f!,
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
    ::AutoFiniteDiff{fdtype},
    f!,
    x::AbstractArray,
    extras::Nothing,
) where {fdtype}
    f!(y, x)
    finite_difference_jacobian!(jac, f!, x, fdtype, eltype(y), y)
    return y, jac
end
