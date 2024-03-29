module DifferentiationInterfaceTapirExt

using ADTypes: ADTypes
using DifferentiationInterface: AutoTapir
import DifferentiationInterface as DI
using Tapir: build_rrule, value_and_pullback!!

DI.supports_mutation(::AutoTapir) = DI.MutationNotSupported()

function DI.value_and_pullback(f, ::AutoTapir, x, dy, extras::Nothing)
    rrule = build_rrule(f, x)
    y = f(x)
    dy_righttype = convert(typeof(y), dy)
    _, (_, dx) = value_and_pullback!!(rrule, dy_righttype, f, x)
    return y, dx
end

end
