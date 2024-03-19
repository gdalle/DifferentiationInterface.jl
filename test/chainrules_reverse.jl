using ADTypes: AutoChainRules
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Test
using Zygote: ZygoteRuleConfig

@test available(AutoChainRules(ZygoteRuleConfig()))

test_operators(AutoChainRules(ZygoteRuleConfig()); second_order=false, type_stability=false);
