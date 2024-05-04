using DifferentiationInterface, DifferentiationInterfaceTest
using ReverseDiff: ReverseDiff

backends = [AutoReverseDiff(; compile=false), AutoReverseDiff(; compile=true)]

for backend in backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(backends; logging=LOGGING);
