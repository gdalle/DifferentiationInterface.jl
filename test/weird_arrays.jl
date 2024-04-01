using ForwardDiff: ForwardDiff
using Zygote: Zygote

test_differentiation(
    AutoForwardDiff(; chunksize=2),
    all_operators(),
    # ForwardDiff access individual indices
    vcat(component_scenarios(), static_scenarios());
    # jacobian is super slow for some reason
    excluded=[jacobian],
    second_order=false,
    logging=true,
);

test_differentiation(
    AutoZygote(),
    all_operators(),
    vcat(component_scenarios(), gpu_scenarios(), static_scenarios());
    second_order=false,
    logging=true,
);
