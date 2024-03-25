## Allocating

"""
    value_and_pullback(f, backend, x, dy, [extras]) -> (y, dx)

!!! info
    Required primitive for reverse mode backends to support allocating functions.
"""
function value_and_pullback(f, backend::AbstractADType, x, dy)
    extras = prepare_pullback(f, backend, x)
    return value_and_pullback(f, backend, x, dy, extras)
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
    return last(value_and_pullback(f, backend, x, dy, extras))
end

"""
    pullback!!(f, dx, backend, x, dy, [extras]) -> dx
"""
function pullback!!(
    f, dx, backend::AbstractADType, x, dy, extras=prepare_pullback(f, backend, x)
)
    return last(value_and_pullback!!(f, dx, backend, x, dy, extras))
end

## Mutating

"""
    value_and_pullback!!(f!, y, dx, backend, x, dy, [extras]) -> (y, dx)

!!! info
    Required primitive for reverse mode backends to support mutating functions.
"""
function value_and_pullback!!(f!, y, dx, backend::AbstractADType, x, dy)
    extras = prepare_pullback(f!, backend, y, x)
    return value_and_pullback!!(f!, y, dx, backend, x, dy, extras)
end
