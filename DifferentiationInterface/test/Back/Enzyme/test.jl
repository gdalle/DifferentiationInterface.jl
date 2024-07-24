using ADTypes: ADTypes
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Enzyme: Enzyme
using SparseConnectivityTracer, SparseMatrixColorings
using StableRNGs
using Test

dense_backends = [
    AutoEnzyme(; mode=nothing, constant_function=true),
    AutoEnzyme(; mode=Enzyme.Forward, constant_function=true),
    AutoEnzyme(; mode=Enzyme.Reverse, constant_function=true),
]

nested_dense_backends = [
    DifferentiationInterface.nested(
        AutoEnzyme(; mode=Enzyme.Forward, constant_function=true)
    ),
    DifferentiationInterface.nested(
        AutoEnzyme(; mode=Enzyme.Reverse, constant_function=true)
    ),
]

sparse_backends =
    AutoSparse.(
        dense_backends,
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    )

@testset "Checks" begin
    @testset "Check $(typeof(backend))" for backend in vcat(dense_backends, sparse_backends)
        @test check_available(backend)
        @test check_twoarg(backend)
        @test check_hessian(backend; verbose=false)
    end
end

## Dense backends

test_differentiation(
    vcat(dense_backends, nested_dense_backends),
    default_scenarios();
    second_order=false,
    logging=LOGGING,
);

test_differentiation(
    [
        AutoEnzyme(; mode=Enzyme.Forward, constant_function=false),
        AutoEnzyme(; mode=Enzyme.Reverse, constant_function=false),
    ],
    DIT.make_closure.(default_scenarios());
    second_order=false,
    logging=LOGGING,
);

test_differentiation(
    [
        AutoEnzyme(; mode=nothing, constant_function=true),
        AutoEnzyme(; mode=Enzyme.Reverse, constant_function=true),
        SecondOrder(
            AutoEnzyme(; mode=Enzyme.Reverse, constant_function=true),
            AutoEnzyme(; mode=Enzyme.Reverse, constant_function=true),
        ),
        SecondOrder(
            AutoEnzyme(; mode=Enzyme.Forward, constant_function=true),
            AutoEnzyme(; mode=Enzyme.Reverse, constant_function=true),
        ),
    ];
    first_order=false,
    excluded=[:second_derivative],
    logging=LOGGING,
);

test_differentiation(
    [
        AutoEnzyme(; mode=nothing, constant_function=true),
        AutoEnzyme(; mode=Enzyme.Forward, constant_function=true),
    ];
    first_order=false,
    excluded=[:hessian, :hvp],
    logging=LOGGING,
);

test_differentiation(
    AutoEnzyme(; mode=Enzyme.Forward, constant_function=true);  # TODO: add more
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=LOGGING,
);

## Sparse backends

test_differentiation(
    sparse_backends,
    default_scenarios();
    excluded=[:derivative, :gradient, :pullback, :pushforward],
    second_order=false,
    logging=LOGGING,
);

test_differentiation(
    sparse_backends, sparse_scenarios(); second_order=false, sparsity=true, logging=LOGGING
);

## Activity analysis

Ext = Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceEnzymeExt)

function make_closure(data)
    function f(x)
        data
        return x
    end
    return f
end

backend = AutoEnzyme(; mode=Enzyme.Reverse, constant_function=false)

@test Ext.get_f_and_df(make_closure([1.0]), backend) isa Enzyme.Duplicated
