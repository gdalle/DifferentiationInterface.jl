using DifferentiationInterface, DifferentiationInterfaceTest
using PolyesterForwardDiff: PolyesterForwardDiff

backends = [AutoPolyesterForwardDiff(; chunksize=1)]

for backend in backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(backends; logging=LOGGING);
