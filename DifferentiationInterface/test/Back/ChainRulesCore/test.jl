using Pkg
Pkg.add(["ChainRulesCore", "Zygote"])

using ChainRulesCore
using DifferentiationInterface, DifferentiationInterfaceTest
using Test
using Zygote: ZygoteRuleConfig

for backend in [AutoChainRules(ZygoteRuleConfig())]
    @test check_available(backend)
    @test !check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(
    AutoChainRules(ZygoteRuleConfig());
    excluded=[:second_derivative],
    second_order=VERSION >= v"1.10",
    logging=LOGGING,
);
