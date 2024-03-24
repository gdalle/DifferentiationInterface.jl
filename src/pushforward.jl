## Allocating

"""
    value_and_pushforward(f, backend, x, dx, [extras]) -> (y, dy)

!!! info
    Required primitive for forward mode backends to support allocating functions.
"""
function value_and_pushforward(f::F, backend::AbstractADType, x, dx) where {F}
    extras = prepare_pushforward(f, backend, x)
    return value_and_pushforward(f, backend, x, dx, extras)
end

"""
    value_and_pushforward!!(f, dy, backend, x, dx, [extras]) -> (y, dy)
"""
function value_and_pushforward!!(
    f::F, dy, backend::AbstractADType, x, dx, extras=prepare_pushforward(f, backend, x)
) where {F}
    return value_and_pushforward(f, backend, x, dx, extras)
end

"""
    pushforward(f, backend, x, dx, [extras]) -> (y, dy)
"""
function pushforward(
    f::F, backend::AbstractADType, x, dx, extras=prepare_pushforward(f, backend, x)
) where {F}
    return last(value_and_pushforward(f, backend, x, dx, extras))
end

"""
    pushforward!!(f, dy, backend, x, dx, [extras]) -> (y, dy)
"""
function pushforward!!(
    f::F, dy, backend::AbstractADType, x, dx, extras=prepare_pushforward(f, backend, x)
) where {F}
    return last(value_and_pushforward!!(f, dy, backend, x, dx, extras))
end

## Mutating

"""
    value_and_pushforward!!(f!, y, dy, backend, x, dx, [extras]) -> (y, dy)

!!! info
    Required primitive for forward mode backends to support mutating functions.
"""
function value_and_pushforward!!(f!::F, y, dy, backend::AbstractADType, x, dx) where {F}
    extras = prepare_pushforward(f!, backend, y, x)
    return value_and_pushforward!!(f!, y, dy, backend, x, dx, extras)
end
