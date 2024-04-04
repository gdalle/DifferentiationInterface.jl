module DifferentiationInterfaceTapirExt

using ADTypes: ADTypes
import DifferentiationInterface as DI
using DifferentiationInterface: AutoTapir
using DifferentiationInterface: PullbackExtras
using Tapir: CoDual, build_rrule, value_and_pullback!!, zero_codual

DI.supports_mutation(::AutoTapir) = DI.MutationNotSupported()

function zero_sametype!!(x_target, x::Number)
    return zero(x)
end

function zero_sametype!!(x_target, x::AbstractArray)
    x_sametype = convert(typeof(x), x_target)
    x_sametype .= zero(eltype(x))
    return x_sametype
end

## Pullback

struct TapirPullbackExtras{R} <: PullbackExtras
    rrule::R
end

DI.prepare_pullback(f, ::AutoTapir, x) = TapirPullbackExtras(build_rrule(f, x))

function DI.value_and_pullback(f, ::AutoTapir, x, dy, extras::TapirPullbackExtras)
    y = f(x)
    dy_righttype = convert(typeof(y), dy)
    _, (_, dx) = value_and_pullback!!(extras.rrule, dy_righttype, f, x)
    return y, dx
end

function DI.value_and_pullback!!(f, dx, ::AutoTapir, x, dy, extras::TapirPullbackExtras)
    y = f(x)
    dy_righttype = convert(typeof(y), dy)
    dx_righttype = zero_sametype!!(dx, x)
    new_y, (_, new_dx) = value_and_pullback!!(
        extras.rrule, dy_righttype, zero_codual(f), CoDual(x, dx_righttype)
    )
    return new_y, new_dx
end

end
