## Pushforward

function DI.value_and_pushforward!(
    f::F, _dy::Number, ::AutoFiniteDiff{fdtype}, x, dx
) where {F,fdtype}
    y = f(x)
    step(t::Number)::Number = f(x .+ t .* dx)
    new_dy = finite_difference_derivative(step, zero(eltype(dx)), fdtype, eltype(y), y)
    return y, new_dy
end

function DI.value_and_pushforward!(
    f::F, dy::AbstractArray, ::AutoFiniteDiff{fdtype}, x, dx
) where {F,fdtype}
    y = f(x)
    step(t::Number)::AbstractArray = f(x .+ t .* dx)
    finite_difference_gradient!(
        dy, step, zero(eltype(dx)), fdtype, eltype(y), FUNCTION_NOT_INPLACE, y
    )
    return y, dy
end

function DI.value_and_pushforward(f::F, ::AutoFiniteDiff{fdtype}, x, dx) where {F,fdtype}
    y = f(x)
    step(t::Number) = f(x .+ t .* dx)
    new_dy = finite_difference_derivative(step, zero(eltype(dx)), fdtype, eltype(y), y)
    return y, new_dy
end
