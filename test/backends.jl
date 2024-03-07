using DifferentiationInterface
using DifferentiationInterface: is_custom
using Test

backend = ChainRulesForwardBackend{false,Nothing}(nothing)
@test !is_custom(backend)
@test ad_mode(backend) == :forward

backend = ChainRulesReverseBackend{true,Nothing}(nothing)
@test is_custom(backend)
@test ad_mode(backend) == :reverse
