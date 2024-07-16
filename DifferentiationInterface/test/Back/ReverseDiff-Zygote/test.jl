using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
using ReverseDiff: ReverseDiff
using Zygote: Zygote
using Test

backends = [SecondOrder(AutoReverseDiff(), AutoZygote())]

for backend in backends
    @test check_available(backend)
    @test !check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(backends; first_order=false, logging=LOGGING);
