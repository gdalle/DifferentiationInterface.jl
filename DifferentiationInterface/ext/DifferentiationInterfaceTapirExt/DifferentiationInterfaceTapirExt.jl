module DifferentiationInterfaceTapirExt

using ADTypes: ADTypes
import DifferentiationInterface as DI
using DifferentiationInterface: AutoTapir
using DifferentiationInterface: PullbackExtras
using Tapir:
    CoDual,
    NoTangent,
    build_rrule,
    increment!!,
    set_to_zero!!,
    tangent,
    tangent_type,
    value_and_pullback!!,
    zero_codual,
    zero_tangent

include("allocating.jl")
include("mutating.jl")

end
