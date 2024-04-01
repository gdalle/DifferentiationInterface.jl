using ForwardDiff: ForwardDiff
using Zygote: Zygote

test_differentiation(
    AutoForwardDiff(; chunksize=2),
    # ForwardDiff access individual indices
    vcat(component_scenarios(), static_scenarios());
    # jacobian is super slow for some reason
    excluded=[JacobianScenario],
    second_order=false,
    logging=true,
);

test_differentiation(
    AutoZygote(),
    vcat(component_scenarios(), gpu_scenarios(), static_scenarios());
    second_order=false,
    logging=true,
);
