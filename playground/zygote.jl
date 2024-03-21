using ADTypes
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
import Zygote

b = AutoZygote()
value_and_gradient(sum, b, [3.0])
value_and_jacobian(copy,  b, [3.0])
