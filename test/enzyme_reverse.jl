include("test_imports.jl")

using Enzyme: Enzyme

@test check_available(AutoEnzyme(Enzyme.Reverse))
@test check_mutation(AutoEnzyme(Enzyme.Reverse))

test_differentiation(AutoEnzyme(Enzyme.Reverse); second_order=false);
