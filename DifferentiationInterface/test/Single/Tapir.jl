using DifferentiationInterface, DifferentiationInterfaceTest
using Tapir: Tapir

for backend in [AutoTapir()]
    @test check_available(backend)
    @test check_twoarg(backend)
    @test !check_hessian(backend; verbose=false)
end

test_differentiation(AutoTapir(); second_order=false, logging=LOGGING);
