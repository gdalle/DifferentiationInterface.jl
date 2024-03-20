using ADTypes: AutoZygote
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Zygote: Zygote

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test available(AutoZygote())
@test !supports_mutation(AutoZygote())
@test supports_hessian(AutoZygote())

test_operators(AutoZygote(); type_stability=false);
