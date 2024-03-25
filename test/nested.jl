include("test_imports.jl")

using ForwardDiff: ForwardDiff
using Tracker: Tracker
using Zygote: Zygote

test_differentiation(AutoZygote(), [gradient], nested_scenarios(); detailed=true);

test_differentiation(
    AutoTracker(), [gradient], nested_scenarios(; immutables=false); detailed=true
);

test_differentiation(
    AutoForwardDiff(),
    [derivative],
    nested_scenarios();
    detailed=true,
    correctness=false,
    error_free=true,
);
