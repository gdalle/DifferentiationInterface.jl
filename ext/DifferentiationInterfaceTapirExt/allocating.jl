function DI.value_and_pullback(f, ::AutoTapir, x, dy, rrule)
    y = f(x)  # TODO: one call too many, just for the conversion
    dy_righttype = convert(typeof(y), dy)
    _, (_, dx) = value_and_pullback!!(rrule, dy_righttype, f, x)
    return y, dx
end

function DI.value_and_pullback!!(f, dx, ::AutoTapir, x, dy, rrule)
    y = f(x)  # TODO: one call too many, just for the conversion
    dy_righttype = convert(typeof(y), dy)
    dx_righttype = zero_sametype!!(dx, x)
    new_y, (_, new_dx) = value_and_pullback!!(
        rrule, dy_righttype, zero_codual(f), CoDual(x, dx_righttype)
    )
    return new_y, new_dx
end

for op in [:pushforward, :pullback, :derivative, :gradient, :jacobian]
    prep_op = Symbol(:prepare_, op)
    @eval function DI.$prep_op(f, ::AutoTapir, x)
        return build_rrule(f, x)
    end
    @eval function DI.$prep_op(rrule, f, ::AutoTapir, x)
        return rrule
    end
end
