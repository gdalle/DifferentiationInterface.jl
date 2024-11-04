using Pkg
Pkg.add("Enzyme")

using ADTypes: ADTypes
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Enzyme: Enzyme
using StaticArrays
using Test

LOGGING = true # get(ENV, "CI", "false") == "false"

backends = [
    AutoEnzyme(; mode=nothing),
    AutoEnzyme(; mode=Enzyme.Forward),
    AutoEnzyme(; mode=Enzyme.Reverse),
    AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Const),
    AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const),
]

duplicated_backends = [
    AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Duplicated),
    AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Duplicated),
]

@testset "Checks" begin
    @testset "Check $(typeof(backend))" for backend in backends
        @test check_available(backend)
        @test check_inplace(backend)
    end
end;

## First order

test_differentiation(backends, default_scenarios(); excluded=SECOND_ORDER, logging=LOGGING);

test_differentiation(
    backends[1:3],
    default_scenarios(; include_normal=false, include_constantified=true);
    excluded=SECOND_ORDER,
    logging=LOGGING,
);

test_differentiation(
    duplicated_backends,
    default_scenarios(; include_normal=false, include_closurified=true);
    excluded=SECOND_ORDER,
    logging=LOGGING,
);

#=
# TODO: reactivate type stability tests

test_differentiation(
    AutoEnzyme(; mode=Enzyme.Forward),  # TODO: add more
    default_scenarios(; include_batchified=false);
    correctness=false,
    type_stability=:prepared,
    excluded=SECOND_ORDER,
    logging=LOGGING,
);
=#

## Second order

test_differentiation(
    AutoEnzyme(),
    default_scenarios(; include_constantified=true);
    excluded=FIRST_ORDER,
    logging=LOGGING,
);

test_differentiation(
    AutoEnzyme(; mode=Enzyme.Forward);
    excluded=vcat(FIRST_ORDER, [:hessian, :hvp]),
    logging=LOGGING,
);

test_differentiation(
    AutoEnzyme(; mode=Enzyme.Reverse);
    excluded=vcat(FIRST_ORDER, [:second_derivative]),
    logging=LOGGING,
);

test_differentiation(
    SecondOrder(AutoEnzyme(; mode=Enzyme.Reverse), AutoEnzyme(; mode=Enzyme.Forward));
    logging=LOGGING,
);

## Sparse

test_differentiation(
    MyAutoSparse.(AutoEnzyme(; function_annotation=Enzyme.Const)),
    sparse_scenarios();
    sparsity=true,
    logging=LOGGING,
);

##

filtered_static_scenarios = filter(static_scenarios()) do s
    DIT.operator_place(s) == :out && DIT.function_place(s) == :out
end

test_differentiation(
    [AutoEnzyme(; mode=Enzyme.Forward), AutoEnzyme(; mode=Enzyme.Reverse)],
    filtered_static_scenarios;
    excluded=SECOND_ORDER,
    logging=LOGGING,
)
