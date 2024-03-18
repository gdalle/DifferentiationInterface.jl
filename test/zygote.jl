using ADTypes: AutoZygote
using Zygote: Zygote
using DifferentiationInterface.DifferentiationTest

test_operators(AutoZygote(); second_order=false, type_stability=false);
