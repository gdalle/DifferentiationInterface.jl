using Diffractor
using DifferentiationInterface

test_pushforward(
    ChainRulesForwardBackend(Diffractor.DiffractorRuleConfig()); type_stability=false
);
test_jacobian_and_friends(
    ChainRulesForwardBackend(Diffractor.DiffractorRuleConfig()); type_stability=false
);
