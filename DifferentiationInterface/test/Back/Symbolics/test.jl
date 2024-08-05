using Pkg
Pkg.add("Symbolics")

using DifferentiationInterface, DifferentiationInterfaceTest
using Symbolics: Symbolics
using Test

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoSymbolics(), AutoSparse(AutoSymbolics())]
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(AutoSymbolics(); logging=LOGGING);

test_differentiation(
    AutoSparse(AutoSymbolics()), sparse_scenarios(); sparsity=true, logging=LOGGING
);
