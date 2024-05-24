using DifferentiationInterface, DifferentiationInterfaceTest
using PolyesterForwardDiff: PolyesterForwardDiff
using Test

for backend in [AutoPolyesterForwardDiff(; chunksize=1)]
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(AutoPolyesterForwardDiff(; chunksize=1); logging=LOGGING);
