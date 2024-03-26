include("test_imports.jl")

using ForwardDiff: ForwardDiff
using Tracker: Tracker
using Zygote: Zygote

## Reverse mode

test_differentiation(AutoZygote(), [gradient], nested_scenarios(); detailed=true);

## Forward mode

test_differentiation(AutoForwardDiff(), [derivative], nested_scenarios(); detailed=true);
