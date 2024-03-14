using ADTypes: AutoZygote
using Zygote: Zygote
using DifferentiationInterface.DifferentiationTest

test_all_operators(AutoZygote(), default_scenarios(); type_stability=false);
