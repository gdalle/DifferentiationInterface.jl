using ADTypes: AutoZygote
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Test
using Zygote: Zygote

@test available(AutoZygote())

test_operators(AutoZygote(); second_order=false, type_stability=false);
