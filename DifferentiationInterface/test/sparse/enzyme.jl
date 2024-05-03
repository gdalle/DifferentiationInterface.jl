using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
using Enzyme: Enzyme
using Symbolics: Symbolics

coloring_algorithm = DI.GreedyColoringAlgorithm()
sparsity_detector = DI.SymbolicsSparsityDetector()

backends = [AutoSparse(AutoEnzyme(Enzyme.Reverse); sparsity_detector, coloring_algorithm)]

for backend in backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test !check_hessian(backend)
end

test_differentiation(
    backends, sparse_scenarios(); second_order=false, sparsity=true, logging=LOGGING
);
