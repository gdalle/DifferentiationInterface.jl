module DifferentiationInterfaceEnzymeExt

using ADTypes: ADTypes, AutoEnzyme
using Base: Fix1
import DifferentiationInterface as DI
using Enzyme:
    Active,
    Annotation,
    BatchDuplicated,
    BatchMixedDuplicated,
    Const,
    Duplicated,
    DuplicatedNoNeed,
    EnzymeCore,
    Forward,
    ForwardMode,
    ForwardWithPrimal,
    MixedDuplicated,
    Mode,
    NoPrimal,
    Reverse,
    ReverseMode,
    ReverseModeSplit,
    ReverseSplitNoPrimal,
    ReverseSplitWidth,
    ReverseSplitWithPrimal,
    ReverseWithPrimal,
    WithPrimal,
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

end # module
