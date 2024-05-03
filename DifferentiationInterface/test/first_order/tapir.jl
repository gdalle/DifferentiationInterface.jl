using DifferentiationInterface, DifferentiationInterfaceTest
using Tapir: Tapir

backends = [AutoTapir()]

for backend in backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test !check_hessian(backend; verbose=false)
end

test_differentiation(backends; second_order=false, logging=LOGGING);
