using ADTypes: AutoTracker
using Tracker: Tracker
using DifferentiationInterface.DifferentiationTest

test_operators(
    AutoTracker();
    input_type=AbstractVector,
    output_type=Number,
    second_order=false,
    type_stability=false,
);
