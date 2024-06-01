using DifferentiationInterface, DifferentiationInterfaceTest
using Symbolics: Symbolics
using Test

for backend in [AutoSymbolics(), AutoSparse(AutoSymbolics())]
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(AutoSymbolics(); logging=LOGGING);

test_differentiation(
    AutoSparse(AutoSymbolics()), sparse_scenarios(); sparsity=true, logging=LOGGING
);
