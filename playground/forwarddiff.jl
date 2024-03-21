using ADTypes
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
import ForwardDiff

b = AutoForwardDiff()
value_and_derivative(x -> [1, x, 2x^2], b, 3)
value_and_derivative!(x -> [1, x, 2x^2],[0, 0, 0.],  b, 3)
value_and_jacobian(copy, b, [1, 2])
