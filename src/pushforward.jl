## Allocating

"""
    value_and_pushforward(f, backend, x, dx, [extras]) -> (y, dy)

!!! info
    Required primitive for forward mode backends to support allocating functions.
"""
function value_and_pushforward(
    f, backend::AbstractADType, x, dx, extras=prepare_pushforward(f, backend, x)
)
    y = f(x)
    dy = if x isa Number && y isa Number
        dx * pullback(f, backend, x, one(y))
    elseif x isa AbstractArray && y isa Number
        dot(dx, pullback(f, backend, x, one(y)))
    elseif x isa Number && y isa AbstractArray
        map(CartesianIndices(y)) do i
            dx * pullback(f, backend, x, basis(backend, y, i))
        end
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(y)) do i
            dot(dx, pullback(f, backend, x, basis(backend, y, i)))
        end
    end
    return y, dy
end

"""
    value_and_pushforward!!(f, dy, backend, x, dx, [extras]) -> (y, dy)
"""
function value_and_pushforward!!(
    f, dy, backend::AbstractADType, x, dx, extras=prepare_pushforward(f, backend, x)
)
    return value_and_pushforward(f, backend, x, dx, extras)
end

"""
    pushforward(f, backend, x, dx, [extras]) -> (y, dy)
"""
function pushforward(
    f, backend::AbstractADType, x, dx, extras=prepare_pushforward(f, backend, x)
)
    return value_and_pushforward(f, backend, x, dx, extras)[2]
end

"""
    pushforward!!(f, dy, backend, x, dx, [extras]) -> (y, dy)
"""
function pushforward!!(
    f, dy, backend::AbstractADType, x, dx, extras=prepare_pushforward(f, backend, x)
)
    return value_and_pushforward!!(f, dy, backend, x, dx, extras)[2]
end

## Mutating

"""
    value_and_pushforward!!(f!, y, dy, backend, x, dx, [extras]) -> (y, dy)

!!! info
    Required primitive for forward mode backends to support mutating functions.
"""
function value_and_pushforward!!(
    f!, y, dy, backend::AbstractADType, x, dx, extras=prepare_pushforward(f!, backend, y, x)
)
    dy = if x isa Number && y isa AbstractArray
        map(CartesianIndices(y)) do i
            dx * value_and_pullback!!(f!, y, zero(x), backend, x, basis(backend, y, i))[2]
        end
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(y)) do i
            dot(
                dx,
                value_and_pullback!!(f!, y, similar(x), backend, x, basis(backend, y, i))[2],
            )
        end
    end
    return y, dy
end
