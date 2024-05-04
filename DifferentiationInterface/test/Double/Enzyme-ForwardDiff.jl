using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using Zygote: Zygote

backends = [
    SecondOrder(AutoForwardDiff(), AutoEnzyme(Enzyme.Forward)),
    SecondOrder(AutoEnzyme(Enzyme.Reverse), AutoForwardDiff()),
]

for backend in backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(backends; first_order=false, logging=LOGGING);
