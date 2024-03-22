using ADTypes: AutoZygote
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Zygote: Zygote

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test check_available(AutoZygote())
@test !check_mutation(AutoZygote())

test_operators(AutoZygote(); type_stability=false);
