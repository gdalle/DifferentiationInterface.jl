module DifferentiationInterfaceMooncakeExt

using ADTypes: ADTypes, AutoMooncake
import DifferentiationInterface as DI
using Mooncake:
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
    get_interpreter

DI.check_available(::AutoMooncake) = true

include("onearg.jl")
include("twoarg.jl")

end
