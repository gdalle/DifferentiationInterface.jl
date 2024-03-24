using Enzyme: Enzyme

@test check_available(AutoEnzyme(Enzyme.Forward))
@test check_mutation(AutoEnzyme(Enzyme.Forward))

test_differentiation(AutoEnzyme(Enzyme.Forward); type_stability=true, second_order=false);
