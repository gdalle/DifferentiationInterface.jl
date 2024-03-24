using ADTypes: ADTypes

struct AutoNothingForward <: ADTypes.AbstractForwardMode end
struct AutoNothingReverse <: ADTypes.AbstractReverseMode end

@test !check_available(AutoNothingForward())
@test !check_available(AutoNothingReverse())
