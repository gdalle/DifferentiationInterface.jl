using DifferentiationInterface
using Zygote

test_pullback(
    ChainRulesBackend(Zygote.ZygoteRuleConfig()); output_type=Array, type_stability=false
)
