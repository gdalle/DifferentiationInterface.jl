using ADTypes
using DifferentiationInterface
using Test

backend = SecondOrder(AutoForwardDiff(), AutoZygote())

@test ADTypes.mode(backend) isa ADTypes.ForwardMode
@test DifferentiationInterface.outer(backend) isa AutoForwardDiff
@test DifferentiationInterface.inner(backend) isa AutoZygote
