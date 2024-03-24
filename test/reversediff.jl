using ADTypes: AutoReverseDiff
using ReverseDiff: ReverseDiff

@test_broken check_available(AutoReverseDiff())
@test check_mutation(AutoReverseDiff())
@test check_hessian(AutoReverseDiff())

test_differentiation(AutoReverseDiff(); input_type=AbstractArray);
