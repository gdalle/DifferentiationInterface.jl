using Diffractor
using DifferentiationInterface

test_pushforward(
    ChainRulesForwardBackend(Diffractor.DiffractorRuleConfig()); type_stability=false
)
