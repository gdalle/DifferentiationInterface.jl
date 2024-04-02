module DifferentiationInterfaceTapirExt

using ADTypes: ADTypes
import DifferentiationInterface as DI
using DifferentiationInterface: AutoTapir
using DifferentiationInterface: PullbackExtras
using Tapir: CoDual, NoTangent, build_rrule, value_and_pullback!!, tangent_type, zero_codual

zero!!(x::Number) = x
function zero!!(x::AbstractArray)
    x .= zero(eltype(x))
    return x
end

include("allocating.jl")
include("mutating.jl")

end
