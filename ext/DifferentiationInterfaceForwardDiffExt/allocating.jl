function DI.value_and_pushforward!(f::F, dy, ::AutoForwardDiff, x, dx) where {F}
    T = tag_type(f, x)
    xdual = make_dual(T, x, dx)
    ydual = f(xdual)
    y = my_value(T, ydual)
    dy = my_derivative!(T, dy, ydual)
    return y, dy
end

function DI.value_and_pushforward(f::F, ::AutoForwardDiff, x, dx) where {F}
    T = tag_type(f, x)
    xdual = make_dual(T, x, dx)
    ydual = f(xdual)
    y = my_value(T, ydual)
    new_dy = my_derivative(T, ydual)
    return y, new_dy
end
