"""
    value_and_pullback!!(f, dx, backend, x, dy, [extras]) -> (y, dx)
    value_and_pullback!!(f!, y, dx, backend, x, dy, [extras]) -> (y, dx)
"""
function value_and_pullback!!(f::F, dx, backend, x, dy) where {F}
    extras = prepare_pullback(f, backend, x)
    return value_and_pullback!!(f, dx, backend, x, dy, extras)
end

function value_and_pullback!!(f!::F, y, dx, backend, x, dy) where {F}
    extras = prepare_pullback(f!, backend, y, x)
    return value_and_pullback!!(f!, y, dx, backend, x, dy, extras)
end

"""
    value_and_pullback(f, backend, x, dy, [extras]) -> (y, dx)
"""
function value_and_pullback(f::F, backend, x, dy) where {F}
    extras = prepare_pullback(f, backend, x)
    return value_and_pullback(f, backend, x, dy, extras)
end
