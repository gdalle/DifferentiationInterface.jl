using Diffractor
using DifferentiationInterface

test_pushforward(ChainRulesBackend(Diffractor.DiffractorRuleConfig()); type_stability=false)
