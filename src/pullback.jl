"""
    value_and_pullback!(f, dx, backend, x, dy, [extras]) -> (y, dx)
    value_and_pullback!(f!, y, dx, backend, x, dy, [extras]) -> (y, dx)
"""
function value_and_pullback! end

"""
    value_and_pullback(f, backend, x, dy, [extras]) -> (y, dx)
"""
function value_and_pullback end
