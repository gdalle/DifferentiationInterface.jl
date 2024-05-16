using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using DataFrames: DataFrame

for backend in [AutoForwardDiff(), AutoSparse(AutoForwardDiff())]
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(
    [
        AutoForwardDiff(),
        AutoForwardDiff(; chunksize=2, tag=:hello),
        AutoSparse(AutoForwardDiff()),
    ];
    logging=LOGGING,
);

test_differentiation(
    MyAutoSparse(AutoForwardDiff()), sparse_scenarios(); sparsity=true, logging=LOGGING
);

test_differentiation(
    AutoForwardDiff(),
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
