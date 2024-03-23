using ADTypes: AutoChainRules
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Zygote: ZygoteRuleConfig

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test check_available(AutoChainRules(ZygoteRuleConfig()))
@test !check_mutation(AutoChainRules(ZygoteRuleConfig()))
@test_broken !check_hessian(AutoChainRules(ZygoteRuleConfig()))

test_operators(AutoChainRules(ZygoteRuleConfig()); second_order=false, type_stability=false);
