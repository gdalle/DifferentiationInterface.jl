using Pkg
Pkg.add("Enzyme")

using ADTypes: ADTypes
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Enzyme: Enzyme
using Test

LOGGING = true || get(ENV, "CI", "false") == "false"

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

test_differentiation(backends, default_scenarios(); second_order=false, logging=LOGGING);

test_differentiation(
    backends[1:3],
    default_scenarios(; include_normal=false, include_constantified=true);
    second_order=false,
    logging=LOGGING,
);

test_differentiation(
    duplicated_backends,
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
    AutoEnzyme(),
    default_scenarios(; include_constantified=true);
    first_order=false,
    logging=LOGGING,
);

test_differentiation(
    AutoEnzyme(; mode=Enzyme.Forward);
    first_order=false,
    excluded=[:hessian, :hvp],
    logging=LOGGING,
);

test_differentiation(
    AutoEnzyme(; mode=Enzyme.Reverse);
    first_order=false,
    excluded=[:second_derivative],
    logging=LOGGING,
);

test_differentiation(
    SecondOrder(AutoEnzyme(; mode=Enzyme.Reverse), AutoEnzyme(; mode=Enzyme.Forward));
    first_order=false,
    logging=LOGGING,
);

## Sparse

test_differentiation(
    MyAutoSparse.(AutoEnzyme()), sparse_scenarios(); sparsity=true, logging=LOGGING
);
