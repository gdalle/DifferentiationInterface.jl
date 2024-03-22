function DI.value_and_pushforward!(
    f!::F, y, dy, ::AutoForwardDiff, x, dx, extras::Nothing
) where {F}
    T = tag_type(f!, x)
    xdual = make_dual(T, x, dx)
    ydual = make_dual(T, y, dy)
    f!(ydual, xdual)
    y = my_value!(T, y, ydual)
    dy = my_derivative!(T, dy, ydual)
    return y, dy
end
