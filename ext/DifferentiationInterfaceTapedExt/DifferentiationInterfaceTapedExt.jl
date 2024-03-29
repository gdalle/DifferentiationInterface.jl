module DifferentiationInterfaceTapedExt

using ADTypes: ADTypes
using DifferentiationInterface: AutoTaped
import DifferentiationInterface as DI
using Taped: build_rrule, value_and_pullback!!

DI.supports_mutation(::AutoTaped) = DI.MutationNotSupported()

function DI.value_and_pullback(f, ::AutoTaped, x, dy, extras::Nothing)
    rrule = build_rrule(f, x)
    y = f(x)
    dy_righttype = convert(typeof(y), dy)
    _, (_, dx) = value_and_pullback!!(rrule, dy_righttype, f, x)
    return y, dx
end

end
