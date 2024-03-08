using DifferentiationInterface
using Diffractor: Diffractor

# see https://github.com/JuliaDiff/Diffractor.jl/issues/277

@test_skip test_pushforward(
    ChainRulesForwardBackend(Diffractor.DiffractorRuleConfig()),
    scenarios;
    type_stability=false,
);
@test_skip test_jacobian_and_friends(
    ChainRulesForwardBackend(Diffractor.DiffractorRuleConfig()),
    scenarios;
    type_stability=false,
);
