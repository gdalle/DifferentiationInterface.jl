include("test_imports.jl")

using ForwardDiff: ForwardDiff
using Zygote: Zygote

test_differentiation(
    AutoForwardDiff(; chunksize=2),
    all_operators(),
    # ForwardDiff access individual indices
    weird_array_scenarios(; static=true, component=true, gpu=false);
    # jacobian is super long for some reason
    excluded=[jacobian],
    second_order=false,
);

test_differentiation(
    AutoZygote(),
    all_operators(),
    weird_array_scenarios(; static=true, component=true, gpu=true);
    second_order=false,
);
