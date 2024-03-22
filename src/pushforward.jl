"""
    value_and_pushforward!(f, dy, backend, x, dx, [extras]) -> (y, dy)
    value_and_pushforward!(f!, y, dy, backend, x, dx, [extras]) -> (y, dy)
"""
function value_and_pushforward!(f::F, dy, backend, x, dx) where {F}
    extras = prepare_pushforward(f, backend, x)
    return value_and_pushforward!(f, dy, backend, x, dx, extras)
end

function value_and_pushforward!(f!::F, y, dy, backend, x, dx) where {F}
    extras = prepare_pushforward(f!, backend, y, x)
    return value_and_pushforward!(f!, y, dy, backend, x, dx, extras)
end

"""
    value_and_pushforward(f, backend, x, dx, [extras]) -> (y, dy)
"""
function value_and_pushforward(f::F, backend, x, dx) where {F}
    extras = prepare_pushforward(f, backend, x)
    return value_and_pushforward(f, backend, x, dx, extras)
end
