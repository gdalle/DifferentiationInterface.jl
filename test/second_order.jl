using ADTypes
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest

using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote

using JET: JET
using Test

cross_backends = [
    # forward over reverse
    SecondOrder(AutoZygote(), AutoForwardDiff()),
    SecondOrder(AutoZygote(), AutoEnzyme(Enzyme.Forward)),
    SecondOrder(AutoReverseDiff(), AutoForwardDiff()),
    # reverse over forward
    SecondOrder(AutoForwardDiff(), AutoEnzyme(Enzyme.Reverse)),
]

@testset "$(backend_string(backend))" for backend in cross_backends
    test_operators(backend; first_order=false, mutating=false, type_stability=false)
end;
