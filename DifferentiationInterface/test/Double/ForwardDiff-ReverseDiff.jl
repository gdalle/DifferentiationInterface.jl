using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using SparseConnectivityTracer, SparseMatrixColorings
using Test

dense_backends = [SecondOrder(AutoForwardDiff(), AutoReverseDiff())]

for backend in dense_backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

## Dense backends

test_differentiation(dense_backends; first_order=false, logging=LOGGING);
