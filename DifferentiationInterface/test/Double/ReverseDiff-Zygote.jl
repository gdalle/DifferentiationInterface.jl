using DifferentiationInterface, DifferentiationInterfaceTest
using ReverseDiff: ReverseDiff
using Test
using Zygote: Zygote

backends = [SecondOrder(AutoReverseDiff(), AutoZygote())]

for backend in backends
    @test check_available(backend)
    @test !check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(backends; first_order=false, logging=LOGGING);
