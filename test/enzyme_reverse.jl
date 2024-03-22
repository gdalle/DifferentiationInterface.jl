using ADTypes: AutoEnzyme
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Enzyme: Enzyme

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test check_available(AutoEnzyme(Enzyme.Reverse))
@test check_mutation(AutoEnzyme(Enzyme.Reverse))

test_operators(
    AutoEnzyme(Enzyme.Reverse);
    output_type=Number,
    allocating=true,
    mutating=false,
    second_order=false,
);

test_operators(
    AutoEnzyme(Enzyme.Reverse); mutating=true, allocating=false, second_order=false
);
