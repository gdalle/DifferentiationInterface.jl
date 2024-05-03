using DifferentiationInterface, DifferentiationInterfaceTest
using Symbolics: Symbolics

backends = [AutoSymbolics(), AutoSparse(AutoSymbolics())]

for backend in backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(AutoSymbolics(); logging=LOGGING);
test_differentiation(
    AutoSparse(AutoSymbolics());
    excluded=[JacobianScenario, HessianScenario],
    logging=LOGGING,
);

test_differentiation(
    AutoSparse(AutoSymbolics()), sparse_scenarios(); sparsity=true, logging=LOGGING
);
