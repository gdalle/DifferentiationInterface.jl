using Pkg
Pkg.add("ForwardDiff")

using ComponentArrays: ComponentArrays
using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using StaticArrays: StaticArrays
using Test

LOGGING = get(ENV, "CI", "false") == "false"

backends = [AutoForwardDiff(; tag=:hello), AutoForwardDiff(; chunksize=5)]

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
end

## Dense

test_differentiation(
    backends, default_scenarios(; include_constantified=true); logging=LOGGING
);

test_differentiation(
    AutoForwardDiff(); correctness=false, type_stability=true, logging=LOGGING
);

test_differentiation(
    AutoForwardDiff(; chunksize=5);
    correctness=false,
    type_stability=(; preparation=true, prepared_op=true, unprepared_op=false),
    logging=LOGGING,
);

test_differentiation(
    backends,
    vcat(component_scenarios(), static_scenarios()); # FD accesses individual indices
    excluded=[:jacobian],  # jacobian is super slow for some reason
    second_order=false,
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
