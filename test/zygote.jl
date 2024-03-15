using ADTypes: AutoZygote
using Zygote: Zygote
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoZygote(); type_stability=false);
