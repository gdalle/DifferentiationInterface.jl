module DifferentiationInterfaceTapirExt

using ADTypes: ADTypes
using DifferentiationInterface: AutoTapir
import DifferentiationInterface as DI
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

function DI.value_and_pullback(f, ::AutoTapir, x, dy, rrule)
    y = f(x)
    dy_righttype = convert(typeof(y), dy)
    _, (_, dx) = value_and_pullback!!(rrule, dy_righttype, f, x)
    return y, dx
end

function DI.value_and_pullback!!(f, dx, ::AutoTapir, x, dy, rrule)
    y = f(x)
    dy_righttype = convert(typeof(y), dy)
    dx_righttype = zero_sametype!!(dx, x)
    new_y, (_, new_dx) = value_and_pullback!!(
        rrule, dy_righttype, zero_codual(f), CoDual(x, dx_righttype)
    )
    return new_y, new_dx
end

for op in [:pushforward, :pullback, :derivative, :gradient, :jacobian]
    prep_op = Symbol(:prepare_, op)
    @eval function DI.$prep_op(f, backend::AutoTapir, x)
        return build_rrule(f, x)
    end
    @eval function DI.$prep_op(rrule, f, backend::AutoTapir, x)
        return rrule
    end
end

end
