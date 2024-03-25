include("test_imports.jl")

using Enzyme: Enzyme

@test check_available(AutoEnzyme(Enzyme.Reverse))
@test check_mutation(AutoEnzyme(Enzyme.Reverse))

test_differentiation(
    AutoEnzyme(Enzyme.Reverse);
    output_type=Number,
    type_stability=true,
    allocating=true,
    mutating=false,
    second_order=false,
);

test_differentiation(
    AutoEnzyme(Enzyme.Reverse); mutating=true, allocating=false, second_order=false
);
