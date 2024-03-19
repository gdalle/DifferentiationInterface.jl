using ADTypes: AutoTracker
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Test
using Tracker: Tracker

@test available(AutoTracker())

test_operators(
    AutoTracker();
    output_type=Union{Number,AbstractVector},
    second_order=false,
    mutating=false,
    type_stability=false,
);
