using ADTypes
using BenchmarkTools
using DifferentiationInterface
using LinearAlgebra

using Diffractor: Diffractor
using Enzyme: Enzyme
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote, ZygoteRuleConfig

include("utils.jl")

SUITE = make_suite(;
    backends=[
        AutoChainRules(ZygoteRuleConfig()),
        AutoDiffractor(),
        AutoEnzyme(Enzyme.Forward),
        AutoEnzyme(Enzyme.Reverse),
        AutoFiniteDiff(),
        AutoForwardDiff(; chunksize=2),
        AutoPolyesterForwardDiff(; chunksize=2),
        AutoReverseDiff(),
        AutoReverseDiff(; compile=true),
        AutoZygote(),
    ],
    included=[:derivative, :multiderivative, :gradient, :jacobian],
)
