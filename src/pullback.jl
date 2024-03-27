## Allocating

"""
    value_and_pullback(f, backend, x, dy, [extras]) -> (y, dx)

!!! info
    Required primitive for reverse mode backends to support allocating functions.
"""
function value_and_pullback(
    f, backend::AbstractADType, x, dy, extras=prepare_pullback(f, backend, x)
)
    y = f(x)
    dx = if x isa Number && y isa Number
        dy * pushforward(f, backend, x, one(x))
    elseif x isa Number && y isa AbstractArray
        dot(dy, pushforward(f, backend, x, one(x)))
    elseif x isa AbstractArray && y isa Number
        map(CartesianIndices(x)) do j
            dy * pushforward(f, backend, x, basis(backend, x, j))
        end
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(x)) do j
            dot(dy, pushforward(f, backend, x, basis(backend, x, j)))
        end
    end
    return y, dx
end

"""
    value_and_pullback!!(f, dx, backend, x, dy, [extras]) -> (y, dx)
"""
function value_and_pullback!!(
    f, dx, backend::AbstractADType, x, dy, extras=prepare_pullback(f, backend, x)
)
    return value_and_pullback(f, backend, x, dy, extras)
end

"""
    pullback(f, backend, x, dy, [extras]) -> dx
"""
function pullback(f, backend::AbstractADType, x, dy, extras=prepare_pullback(f, backend, x))
    return value_and_pullback(f, backend, x, dy, extras)[2]
end

"""
    pullback!!(f, dx, backend, x, dy, [extras]) -> dx
"""
function pullback!!(
    f, dx, backend::AbstractADType, x, dy, extras=prepare_pullback(f, backend, x)
)
    return value_and_pullback!!(f, dx, backend, x, dy, extras)[2]
end

## Mutating

"""
    value_and_pullback!!(f!, y, dx, backend, x, dy, [extras]) -> (y, dx)

!!! info
    Required primitive for reverse mode backends to support mutating functions.
"""
function value_and_pullback!!(
    f!, y, dx, backend::AbstractADType, x, dy, extras=prepare_pullback(f!, backend, y, x)
)
    dx = if x isa Number && y isa AbstractArray
        dot(dy, value_and_pushforward!!(f!, y, similar(y), backend, x, one(x))[2])
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(x)) do j
            dot(
                dy,
                value_and_pushforward!!(
                    f!, y, similar(y), backend, x, basis(backend, x, j)
                )[2],
            )
        end
    end
    return y, dx
end
