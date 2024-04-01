function DI.value_and_pullback!!(f!, y, dx, ::AutoTapir, x, dy, extras::Nothing)
    rrule = build_rrule(f!, y, x)
    dy_righttype = convert(typeof(y), dy)
    dx_righttype = zero_sametype!!(dx, x)
    dz = nothing  # f!(y, x) = nothing
    _, (_, _, new_dx) = value_and_pullback!!(
        rrule, dz, zero_codual(f!), CoDual(y, dy_righttype), CoDual(x, dx_righttype)
    )
    return y, new_dx
end
