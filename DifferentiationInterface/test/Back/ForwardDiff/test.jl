using Pkg
Pkg.add("ForwardDiff")

using ComponentArrays: ComponentArrays
using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using StaticArrays: StaticArrays
using Test

LOGGING = get(ENV, "CI", "false") == "false"

backends = [
    AutoForwardDiff(), AutoForwardDiff(; tag=:hello), AutoForwardDiff(; chunksize=5)
]

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
end

## Dense

test_differentiation(
    backends, default_scenarios(; include_constantified=true); logging=LOGGING
);

test_differentiation(
    AutoForwardDiff(); correctness=false, type_stability=:prepared, logging=LOGGING
);

test_differentiation(
    AutoForwardDiff(; chunksize=5);
    correctness=false,
    type_stability=:full,
    excluded=[:hessian],
    logging=LOGGING,
);

test_differentiation(
    backends,
    vcat(component_scenarios(), static_scenarios()); # FD accesses individual indices
    excluded=vcat(SECOND_ORDER, [:jacobian]),  # jacobian is super slow for some reason
    logging=LOGGING,
);

## Sparse

test_differentiation(MyAutoSparse.(backends), default_scenarios(); logging=LOGGING);

test_differentiation(
    MyAutoSparse.(backends),
    sparse_scenarios(; include_constantified=true);
    sparsity=true,
    logging=LOGGING,
);

## Static

test_differentiation(AutoForwardDiff(), static_scenarios(); logging=LOGGING)
