## Pushforward

function DI.value_and_pushforward!!(
    f!,
    y::AbstractArray,
    dy::AbstractArray,
    ::AutoFiniteDiff{fdtype},
    x,
    dx,
    extras::Nothing,
) where {F,fdtype}
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
