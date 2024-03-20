using ADTypes: AutoChainRules
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Zygote: ZygoteRuleConfig

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test available(AutoChainRules(ZygoteRuleConfig()))
@test !supports_mutation(AutoChainRules(ZygoteRuleConfig()))
@test supports_hessian(AutoChainRules(ZygoteRuleConfig()))

test_operators(AutoChainRules(ZygoteRuleConfig()); type_stability=false);
