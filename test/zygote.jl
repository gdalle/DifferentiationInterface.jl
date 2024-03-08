using DifferentiationInterface
using Zygote: Zygote

test_pullback(ZygoteBackend(), scenarios; type_stability=false);
test_jacobian_and_friends(ZygoteBackend(; custom=true), scenarios; type_stability=false);
test_jacobian_and_friends(ZygoteBackend(; custom=false), scenarios; type_stability=false);
