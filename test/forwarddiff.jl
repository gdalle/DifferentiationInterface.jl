using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using DifferentiationInterface.DifferentiationTest

test_all_operators(AutoForwardDiff(; chunksize=2), default_scenarios(); type_stability=true);

test_all_operators_mutating(
    AutoForwardDiff(; chunksize=2), default_scenarios(); type_stability=true
);
