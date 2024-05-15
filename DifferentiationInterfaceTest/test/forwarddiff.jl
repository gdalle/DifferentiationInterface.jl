using ADTypes
using DifferentiationInterface
using DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer

function MyAutoSparse(backend::AbstractADType)
    coloring_algorithm = GreedyColoringAlgorithm()
    sparsity_detector = TracerSparsityDetector()
    return AutoSparse(backend; sparsity_detector, coloring_algorithm)
end

test_differentiation(AutoForwardDiff(); logging=get(ENV, "CI", "false") == "false")

test_differentiation(
    MyAutoSparse(AutoForwardDiff()),
    sparse_scenarios();
    sparsity=true,
    logging=get(ENV, "CI", "false") == "false",
)

test_differentiation(
    AutoForwardDiff(),
    component_scenarios();
    excluded=[HessianScenario],
    logging=get(ENV, "CI", "false") == "false",
)
