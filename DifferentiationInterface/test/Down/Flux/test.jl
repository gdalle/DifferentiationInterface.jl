using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using FiniteDifferences: FiniteDifferences
using Flux: Flux
using Enzyme: Enzyme
using Zygote: Zygote
using Test

test_differentiation([AutoZygote()], flux_scenarios())
