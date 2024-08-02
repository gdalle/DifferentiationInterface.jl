using Pkg
Pkg.add("Tapir")

using DifferentiationInterface, DifferentiationInterfaceTest
using Tapir: Tapir
using Test

for backend in [AutoTapir(; safe_mode=false)]
    @test check_available(backend)
    @test check_twoarg(backend)
    @test !check_hessian(backend; verbose=false)
end

# Safe mode switched off to avoid polluting the test suite with 
test_differentiation(AutoTapir(; safe_mode=false); second_order=false, logging=LOGGING);
