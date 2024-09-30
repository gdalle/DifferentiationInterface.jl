module DifferentiationInterfaceMooncakeExt

using ADTypes: ADTypes, AutoMooncake
import DifferentiationInterface as DI
using DifferentiationInterface: Context, PullbackPrep, unwrap
using Mooncake:
    CoDual,
    Config,
    NoRData,
    NoTangent,
    build_rrule,
    fdata,
    get_interpreter,
    increment!!,
    primal,
    rdata,
    set_to_zero!!,
    tangent,
    tangent_type,
    value_and_pullback!!,
    zero_codual,
    zero_fcodual,
    zero_tangent,
    __value_and_pullback!!

DI.check_available(::AutoMooncake) = true

get_config(::AutoMooncake{Nothing}) = Config()
get_config(backend::AutoMooncake{<:Config}) = backend.config

include("onearg.jl")
include("twoarg.jl")

end
