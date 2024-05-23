using ADTypes
using DifferentiationInterface
using Test

@test ADTypes.mode(SecondOrder(AutoForwardDiff(), AutoZygote())) isa ADTypes.ForwardMode

@test startswith(string(SecondOrder(AutoForwardDiff(), AutoZygote())), "SecondOrder")
