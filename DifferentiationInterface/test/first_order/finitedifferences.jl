using DifferentiationInterface, DifferentiationInterfaceTest
using FiniteDifferences: FiniteDifferences

backends = [AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1))]

for backend in backends
    @test check_available(backend)
    @test !check_twoarg(backend)
    @test_broken !check_hessian(backend; verbose=false)
end

test_differentiation(backends; second_order=false, logging=LOGGING);
