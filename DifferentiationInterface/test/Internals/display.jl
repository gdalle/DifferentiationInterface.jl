using ADTypes
using DifferentiationInterface
using Test

backend = SecondOrder(AutoForwardDiff(), AutoZygote())
@test startswith(string(backend), "SecondOrder(")
@test endswith(string(backend), ")")

detector = DenseSparsityDetector(AutoForwardDiff(); atol=1e-23)
@test startswith(string(detector), "DenseSparsityDetector(")
@test endswith(string(detector), ")")

diffwith = DifferentiateWith(exp, AutoForwardDiff())
@test startswith(string(diffwith), "DifferentiateWith(")
@test endswith(string(diffwith), ")")

@test DifferentiationInterface.package_name(AutoForwardDiff()) == "ForwardDiff"
@test DifferentiationInterface.package_name(AutoZygote()) == "Zygote"
@test DifferentiationInterface.package_name(AutoSparse(AutoForwardDiff())) == "ForwardDiff"
@test DifferentiationInterface.package_name(SecondOrder(AutoForwardDiff(), AutoZygote())) ==
    "ForwardDiff, Zygote"
