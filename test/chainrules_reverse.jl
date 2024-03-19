using ADTypes: AutoChainRules
using Zygote: ZygoteRuleConfig
using DifferentiationInterface.DifferentiationTest
using Test

@test available(AutoChainRules(ZygoteRuleConfig()))

test_operators(AutoChainRules(ZygoteRuleConfig()); second_order=false, type_stability=false);
