module DifferentiationInterfaceEnzymeExt

using ADTypes: ADTypes, AutoEnzyme
using Base: Fix1
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    JacobianExtras,
    HVPExtras,
    PullbackExtras,
    PushforwardExtras,
    NoDerivativeExtras,
    NoGradientExtras,
    NoHVPExtras,
    NoJacobianExtras,
    NoPullbackExtras,
    NoPushforwardExtras,
    Tangents,
    pick_batchsize
using Enzyme:
    Active,
    Annotation,
    BatchDuplicated,
    Const,
    Duplicated,
    DuplicatedNoNeed,
    EnzymeCore,
    Forward,
    ForwardMode,
    ForwardWithPrimal,
    MixedDuplicated,
    Mode,
    Reverse,
    ReverseMode,
    ReverseModeSplit,
    ReverseSplitWithPrimal,
    ReverseWithPrimal,
    autodiff,
    autodiff_thunk,
    create_shadows,
    gradient,
    gradient!,
    guess_activity,
    hvp,
    hvp!,
    jacobian,
    make_zero,
    make_zero!,
    onehot

DI.check_available(::AutoEnzyme) = true

include("utils.jl")

include("forward_onearg.jl")
include("forward_twoarg.jl")

include("reverse_onearg.jl")
include("reverse_twoarg.jl")

# include("second_order.jl")

end # module
