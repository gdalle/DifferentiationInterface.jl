using DifferentiationInterface, DifferentiationInterfaceTest
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using Test

backends = [
    SecondOrder(AutoForwardDiff(), AutoEnzyme(; mode=Enzyme.Forward)),
    SecondOrder(AutoEnzyme(; mode=Enzyme.Reverse), AutoForwardDiff()),
]

for backend in backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(backends; first_order=false, logging=LOGGING);
