using Pkg
Pkg.add("Zygote")

using ComponentArrays: ComponentArrays
using DifferentiationInterface, DifferentiationInterfaceTest
using JLArrays: JLArrays
using SparseConnectivityTracer, SparseMatrixColorings
using StaticArrays: StaticArrays
using Test
using Zygote: Zygote

LOGGING = get(ENV, "CI", "false") == "false"

dense_backends = [AutoZygote()]

sparse_backends = [
    AutoSparse(
        AutoZygote();
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    ),
]

for backend in vcat(dense_backends, sparse_backends)
    @test check_available(backend)
    @test !check_inplace(backend)
end

## Dense backends

test_differentiation(AutoZygote(); excluded=[:second_derivative], logging=LOGGING);

if VERSION >= v"1.10"
    test_differentiation(
        AutoZygote(),
        vcat(component_scenarios(), gpu_scenarios(), static_scenarios());
        second_order=false,
        logging=LOGGING,
    )
end

## Sparse backends

test_differentiation(
    sparse_backends,
    default_scenarios();
    excluded=[:derivative, :gradient, :hvp, :pullback, :pushforward, :second_derivative],
    logging=LOGGING,
);

test_differentiation(sparse_backends, sparse_scenarios(); sparsity=true, logging=LOGGING)
