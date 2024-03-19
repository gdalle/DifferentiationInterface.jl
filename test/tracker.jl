using ADTypes: AutoTracker
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Tracker: Tracker

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test available(AutoTracker())

test_operators(
    AutoTracker();
    output_type=Union{Number,AbstractVector},
    second_order=false,
    mutating=false,
    type_stability=false,
);
