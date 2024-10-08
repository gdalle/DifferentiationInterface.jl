using Pkg
Pkg.add("Enzyme")

using ADTypes: ADTypes
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Enzyme: Enzyme
using SparseConnectivityTracer, SparseMatrixColorings
using Test

LOGGING = get(ENV, "CI", "false") == "false"

dense_backends = [
    AutoEnzyme(; mode=nothing),
    AutoEnzyme(; mode=Enzyme.Forward),
    AutoEnzyme(; mode=Enzyme.Reverse),
    AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Const),
    AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const),
]

duplicated_function_backends = [
    AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Duplicated),
    AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Duplicated),
]

sparse_backends =
    AutoSparse.(
        dense_backends[1:3],
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    )

second_order_backends = [
    SecondOrder(AutoEnzyme(; mode=Reverse), AutoEnzyme(; mode=Enzyme.Forward)),
    SecondOrder(AutoEnzyme(; mode=Forward), AutoEnzyme(; mode=Enzyme.Revers)),
]

@testset "Checks" begin
    @testset "Check $(typeof(backend))" for backend in
                                            vcat(dense_backends, second_order_backends)
        @test check_available(backend)
        @test check_inplace(backend)
    end
end;

## Dense backends

test_differentiation(
    dense_backends, default_scenarios(); second_order=false, logging=LOGGING
);

test_differentiation(
    dense_backends[1:3],
    default_scenarios(; include_normal=false, include_constantified=true);
    second_order=false,
    logging=LOGGING,
);

test_differentiation(
    duplicated_function_backends,
    default_scenarios(; include_normal=false, include_closurified=true);
    second_order=false,
    logging=LOGGING,
);

#=
# TODO: reactivate type stability tests

test_differentiation(
    AutoEnzyme(; mode=Enzyme.Forward),  # TODO: add more
    default_scenarios(; include_batchified=false);
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=LOGGING,
);
=#

## Second order

test_differentiation(
    [AutoEnzyme(; mode=nothing), AutoEnzyme(; mode=Enzyme.Reverse)];
    first_order=false,
    excluded=[:second_derivative],
    logging=LOGGING,
);

test_differentiation(
    [AutoEnzyme(; mode=nothing), AutoEnzyme(; mode=Enzyme.Forward)];
    first_order=false,
    excluded=[:hessian, :hvp],
    logging=LOGGING,
);

test_differentiation(second_order_backends; first_order=false, logging=LOGGING);

## Sparse backends

test_differentiation(
    sparse_backends,
    default_scenarios();
    excluded=[:derivative, :gradient, :pullback, :pushforward, :second_derivative, :hvp],
    second_order=false,  # TODO: make true
    logging=LOGGING,
);

test_differentiation(
    sparse_backends,
    sparse_scenarios();
    sparsity=true,
    second_order=false,  # TODO: make true
    logging=LOGGING,
);
