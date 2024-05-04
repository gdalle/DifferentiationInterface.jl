using DifferentiationInterface, DifferentiationInterfaceTest
using Diffractor: Diffractor

for backend in [AutoDiffractor()]
    @test check_available(backend)
    @test !check_twoarg(backend)
    @test !check_hessian(backend; verbose=false)
end

test_differentiation(AutoDiffractor(); second_order=false, logging=LOGGING);
