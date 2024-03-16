using ADTypes
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Test

using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote

second_order_backends = [
    SecondOrder(AutoZygote(), AutoForwardDiff()),
    SecondOrder(AutoReverseDiff(), AutoForwardDiff()),
    SecondOrder(AutoZygote(), AutoEnzyme(Enzyme.Forward)),
]

@testset "$(typeof(backend.second)) over $(typeof(backend.first))" for backend in
                                                                       second_order_backends
    test_operators_allocating(
        backend; input_type=AbstractVector, included=[:hessian], type_stability=false
    )
end;
