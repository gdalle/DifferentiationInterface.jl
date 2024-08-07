module DifferentiationInterfaceZygoteExt

using ADTypes: AutoForwardDiff, AutoZygote
import DifferentiationInterface as DI
using DifferentiationInterface:
    Batch,
    HVPExtras,
    NoGradientExtras,
    NoHessianExtras,
    NoJacobianExtras,
    NoPullbackExtras,
    PullbackExtras
using DocStringExtensions
using ForwardDiff: ForwardDiff
using Zygote:
    Buffer,
    ZygoteRuleConfig,
    gradient,
    hessian,
    jacobian,
    pullback,
    withgradient,
    withjacobian
using Compat

DI.check_available(::AutoZygote) = true
DI.twoarg_support(::AutoZygote) = DI.TwoArgSupported()

include("onearg.jl")
include("twoarg.jl")

end
