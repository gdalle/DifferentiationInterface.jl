using DifferentiationInterface
using Zygote

test_pullback(ZygoteBackend(); type_stability=false);
test_jacobian_and_friends(ZygoteBackend(; custom=true); type_stability=false);
test_jacobian_and_friends(ZygoteBackend(; custom=false); type_stability=false);
