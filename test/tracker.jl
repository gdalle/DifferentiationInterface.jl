using ADTypes: AutoTracker
using Tracker: Tracker
using DifferentiationInterface.DifferentiationTest
using Test

@test available(AutoTracker())

test_operators(
    AutoTracker();
    output_type=Union{Number,AbstractVector},
    second_order=false,
    mutating=false,
    type_stability=false,
);
