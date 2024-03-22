## Pushforward

function DI.value_and_pushforward!(
    _dy::Number, ::AutoFiniteDiff{fdtype}, f, x, dx
) where {fdtype}
    y = f(x)
    step(t::Number)::Number = f(x .+ t .* dx)
    new_dy = finite_difference_derivative(step, zero(eltype(dx)), fdtype, eltype(y), y)
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::AbstractArray, ::AutoFiniteDiff{fdtype}, f, x, dx
) where {fdtype}
    y = f(x)
    step(t::Number)::AbstractArray = f(x .+ t .* dx)
    finite_difference_gradient!(
        dy, step, zero(eltype(dx)), fdtype, eltype(y), FUNCTION_NOT_INPLACE, y
    )
    return y, dy
end
