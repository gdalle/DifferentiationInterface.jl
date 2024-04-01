module DifferentiationInterfaceTapirExt

using ADTypes: ADTypes
using DifferentiationInterface: AutoTapir
import DifferentiationInterface as DI
using Tapir: CoDual, build_rrule, value_and_pullback!!, zero_codual

function zero_sametype!!(x_target, x::Number)
    return zero(x)
end

function zero_sametype!!(x_target, x::AbstractArray)
    x_sametype = convert(typeof(x), x_target)
    x_sametype .= zero(eltype(x))
    return x_sametype
end

include("allocating.jl")
include("mutating.jl")

end
