using Diffractor
using DifferentiationInterface

# see https://github.com/JuliaDiff/Diffractor.jl/issues/277

@test_skip test_pushforward(
    ChainRulesForwardBackend(Diffractor.DiffractorRuleConfig()); type_stability=false
);
@test_skip test_jacobian_and_friends(
    ChainRulesForwardBackend(Diffractor.DiffractorRuleConfig()); type_stability=false
);
