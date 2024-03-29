module DifferentiationInterfaceTapirExt

using ADTypes: ADTypes
using DifferentiationInterface: AutoTapir
import DifferentiationInterface as DI
using Tapir: CoDual, build_rrule, value_and_pullback!!, zero_codual

DI.supports_mutation(::AutoTapir) = DI.MutationNotSupported()

function DI.value_and_pullback(f, ::AutoTapir, x, dy, rrule)
    y = f(x)
    dy_righttype = convert(typeof(y), dy)
    _, (_, dx) = value_and_pullback!!(rrule, dy_righttype, f, x)
    return y, dx
end

for op in [
    :pushforward,
    :pullback,
    :derivative,
    :gradient,
    :jacobian,
    :second_derivative,
    :hvp,
    :hessian,
]
    prep_op = Symbol(:prepare_, op)
    @eval function DI.$prep_op(f, backend::AutoTapir, x)
        return build_rrule(f, x)
    end
    @eval function DI.$prep_op(rrule, f, backend::AutoTapir, x)
        return rrule
    end
end

end
