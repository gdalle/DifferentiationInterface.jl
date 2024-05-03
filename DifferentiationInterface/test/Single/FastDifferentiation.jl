using DifferentiationInterface, DifferentiationInterfaceTest
using FastDifferentiation: FastDifferentiation

backends = [AutoFastDifferentiation(), AutoSparse(AutoFastDifferentiation())]

for backend in backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(AutoFastDifferentiation(); logging=LOGGING);
test_differentiation(
    AutoSparse(AutoFastDifferentiation());
    excluded=[JacobianScenario, HessianScenario],
    logging=LOGGING,
);

test_differentiation(
    AutoSparse(AutoFastDifferentiation()),
    sparse_scenarios();
    sparsity=true,
    logging=LOGGING,
);
