using DifferentiationInterface
using Enzyme

#=
Spotted Enzyme issues influencing tests:
- https://github.com/EnzymeAD/Enzyme.jl/issues/1330
- https://github.com/EnzymeAD/Enzyme.jl/issues/1332
=#

test_pushforward(EnzymeForwardBackend(); type_stability=true);
test_jacobian_and_friends(
    EnzymeForwardBackend(; custom=true);
    output_type=Union{Number,AbstractVector},
    type_stability=true,
);
test_jacobian_and_friends(EnzymeForwardBackend(; custom=false); type_stability=true);
