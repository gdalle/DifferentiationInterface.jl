using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer
using SparseMatrixColorings
using Test

backends = [
    AutoForwardDiff(),
    AutoForwardDiff(; chunksize=2, tag=:hello),
    AutoSparse(
        AutoForwardDiff();
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    ),
]

for backend in backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(backends; logging=LOGGING);

test_differentiation(backends[3], sparse_scenarios(); sparsity=true, logging=LOGGING);

test_differentiation(
    backends[1:2];
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=LOGGING,
);

test_differentiation(
    backends[1],
    # ForwardDiff access individual indices
    vcat(component_scenarios(), static_scenarios());
    # jacobian is super slow for some reason
    excluded=[JacobianScenario],
    second_order=false,
    logging=LOGGING,
);

if VERSION >= v"1.10"
    include("ForwardDiff/efficiency.jl")
end
