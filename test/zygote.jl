using ADTypes: AutoZygote
using Zygote: Zygote
using DifferentiationInterface.DifferentiationTest
using Test

@test available(AutoZygote())

test_operators(AutoZygote(); second_order=false, type_stability=false);
