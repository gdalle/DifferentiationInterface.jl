using DifferentiationInterface, DifferentiationInterfaceTest
using DifferentiationInterface: AutoForwardFromPrimitive, AutoReverseFromPrimitive
using DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer, SparseMatrixColorings
using Test

LOGGING = get(ENV, "CI", "false") == "false"

fromprimitive_backends = [ #
    AutoForwardFromPrimitive(AutoForwardDiff(; chunksize=5)),
    AutoReverseFromPrimitive(AutoForwardDiff(; chunksize=5)),
]

fromprimitive_sparse_backends =
    AutoSparse.(
        fromprimitive_backends,
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    )

fromprimitive_secondorder_backends = [ #
    SecondOrder(
        AutoForwardFromPrimitive(AutoForwardDiff(; chunksize=5)),
        AutoReverseFromPrimitive(AutoForwardDiff(; chunksize=5)),
    ),
    SecondOrder(
        AutoReverseFromPrimitive(AutoForwardDiff(; chunksize=5)),
        AutoForwardFromPrimitive(AutoForwardDiff(; chunksize=5)),
    ),
]

for backend in vcat(fromprimitive_backends)
    @test check_available(backend)
    @test check_inplace(backend)
end

## Dense scenarios

test_differentiation(
    fromprimitive_backends, default_scenarios(; include_constantified=true); logging=LOGGING
);

test_differentiation(
    fromprimitive_secondorder_backends,
    default_scenarios(; include_constantified=true);
    first_order=false,
    logging=LOGGING,
);

## Sparse scenarios

test_differentiation(
    fromprimitive_sparse_backends,
    default_scenarios(; include_constantified=true);
    excluded=[:derivative, :gradient, :pullback, :pushforward, :second_derivative, :hvp],
    logging=LOGGING,
);

test_differentiation(
    fromprimitive_sparse_backends,
    sparse_scenarios(; include_constantified=true);
    sparsity=true,
    logging=LOGGING,
);
