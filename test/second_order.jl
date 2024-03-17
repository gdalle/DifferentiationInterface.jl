using ADTypes
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Test

using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote

cross_backends = [
    # forward over reverse
    SecondOrder(AutoZygote(), AutoForwardDiff()),
    SecondOrder(AutoZygote(), AutoEnzyme(Enzyme.Forward)),
    SecondOrder(AutoReverseDiff(), AutoForwardDiff()),
    # reverse over forward
    SecondOrder(AutoForwardDiff(), AutoEnzyme(Enzyme.Reverse)),
]

@testset "$(typeof(b.outer)) over $(typeof(b.inner))" for b in cross_backends
    test_second_order_operators_allocating(b; type_stability=false)
end;
