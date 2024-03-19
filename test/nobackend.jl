using ADTypes
using DifferentiationInterface
using Test

struct AutoNothingForward <: ADTypes.AbstractForwardMode end
struct AutoNothingReverse <: ADTypes.AbstractReverseMode end

@test_throws ArgumentError jacobian(AutoNothingForward(), copy, zeros(1))
@test_throws ArgumentError jacobian(AutoNothingReverse(), copy, zeros(1))

@test_throws ArgumentError value_and_jacobian!(
    zeros(1), zeros(1, 1), AutoNothingForward(), copyto!, zeros(1)
)
@test_throws ArgumentError value_and_jacobian!(
    zeros(1), zeros(1, 1), AutoNothingReverse(), copyto!, zeros(1)
)
