"""
    value_and_pushforward!(f, dy, backend, x, dx, [extras]) -> (y, dy)
    value_and_pushforward!(f!, y, dy, backend, x, dx, [extras]) -> (y, dy)
"""
function value_and_pushforward! end

"""
    value_and_pushforward(f, backend, x, dx, [extras]) -> (y, dy)
"""
function value_and_pushforward end
