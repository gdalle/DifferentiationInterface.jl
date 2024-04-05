struct TapirAllocatingPullbackExtras{R} <: PullbackExtras
    rrule::R
end

DI.prepare_pullback(f, ::AutoTapir, x) = TapirAllocatingPullbackExtras(build_rrule(f, x))

function DI.value_and_pullback(f, ::AutoTapir, x, dy, extras::TapirAllocatingPullbackExtras)
    y = f(x)  # TODO: one call too many, just for the conversion
    dy_righttype = convert(tangent_type(typeof(y)), dy)
    new_y, (new_df, new_dx) = value_and_pullback!!(extras.rrule, dy_righttype, f, x)
    return new_y, new_dx
end

function DI.value_and_pullback!!(
    f, dx, ::AutoTapir, x, dy, extras::TapirAllocatingPullbackExtras
)
    y = f(x)  # TODO: one call too many, just for the conversion
    dy_righttype = convert(tangent_type(typeof(y)), dy)
    dx_righttype = convert(tangent_type(typeof(x)), dx)
    dx_righttype = set_to_zero!!(dx_righttype)
    new_y, (new_df, new_dx) = value_and_pullback!!(
        extras.rrule, dy_righttype, zero_codual(f), CoDual(x, dx_righttype)
    )
    return new_y, new_dx
end

function DI.pullback(f, backend::AutoTapir, x, dy, extras::TapirAllocatingPullbackExtras)
    return DI.value_and_pullback(f, backend, x, dy, extras)[2]
end

function DI.pullback!!(
    f, dx, backend::AutoTapir, x, dy, extras::TapirAllocatingPullbackExtras
)
    return DI.value_and_pullback!!(f, dx, backend, x, dy, extras)[2]
end

#=
# First try

function DI.value_and_pullback_split(f, ::AutoTapir, x, extras::TapirAllocatingPullbackExtras)
    tf = zero_tangent(f)
    tx = zero_tangent(x)
    out, pb!! = extras.rrule(CoDual(f, tf), CoDual(x, tx))
    y = copy(primal(out))
    function pullbackfunc(dy)
        dy_righttype = convert(tangent_type(typeof(y)), copy(dy))
        ty = increment!!(tangent(out), dy_righttype)
        res = pb!!(ty, tf, tx)
        extras.rrule(CoDual(f, tf), CoDual(x, tx))
        return last(res)
    end
    return y, pullbackfunc
end
=#

function DI.value_and_pullback_split(
    f, backend::AutoTapir, x, extras::TapirAllocatingPullbackExtras
)
    y = f(x)
    pullbackfunc(dy) = DI.pullback(f, backend, x, dy, extras)
    return y, pullbackfunc
end

function DI.value_and_pullback!!_split(
    f, backend::AutoTapir, x, extras::TapirAllocatingPullbackExtras
)
    y = f(x)
    pullbackfunc!!(dx, dy) = DI.pullback!!(f, dx, backend, x, dy, extras)
    return y, pullbackfunc!!
end
