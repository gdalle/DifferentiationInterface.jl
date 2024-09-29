using Pkg
Pkg.add(["Enzyme", "Reactant"])

using DifferentiationInterface
using DifferentiationInterface: ReactantBackend
using DifferentiationInterfaceTest
using Enzyme: Enzyme
using Reactant: Reactant
using Test

LOGGING = get(ENV, "CI", "false") == "false"

rebackend = ReactantBackend(AutoEnzyme())

test_differentiation(
    ReactantBackend(AutoEnzyme());
    excluded=[
        :derivative, :jacobian, :hessian, :hvp, :pullback, :pushforward, :second_derivative
    ],
    logging=LOGGING,
)
