include("test_imports.jl")

using Tracker: Tracker

@test check_available(AutoTracker())
@test !check_mutation(AutoTracker())
@test !check_hessian(AutoTracker())

test_differentiation(
    AutoTracker(); output_type=Union{Number,AbstractVector}, second_order=false
);
