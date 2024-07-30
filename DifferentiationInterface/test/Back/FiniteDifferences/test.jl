using Pkg
Pkg.add("FiniteDifferences")

using DifferentiationInterface, DifferentiationInterfaceTest
using FiniteDifferences: FiniteDifferences
using Test

for backend in [AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1))]
    @test check_available(backend)
    @test !check_twoarg(backend)
    @test_broken !check_hessian(backend; verbose=false)
end

test_differentiation(
    AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1));
    second_order=false,
    logging=LOGGING,
);
