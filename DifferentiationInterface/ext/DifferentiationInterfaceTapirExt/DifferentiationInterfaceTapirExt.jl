module DifferentiationInterfaceTapirExt

using ADTypes: ADTypes, AutoTapir
import DifferentiationInterface as DI
using DifferentiationInterface: PullbackPrep, Tangents
using Tapir:
    CoDual,
    NoTangent,
    build_rrule,
    increment!!,
    primal,
    set_to_zero!!,
    tangent,
    tangent_type,
    value_and_pullback!!,
    zero_codual,
    zero_tangent,
    NoRData,
    fdata,
    rdata,
    __value_and_pullback!!,
    get_tapir_interpreter

DI.check_available(::AutoTapir) = true

include("onearg.jl")
include("twoarg.jl")

end
