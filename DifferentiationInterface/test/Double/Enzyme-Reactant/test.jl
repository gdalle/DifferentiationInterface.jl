using DifferentiationInterface
using DifferentiationInterface: ReactantBackend
using DifferentiationInterfaceTest
using Enzyme: Enzyme
using Reactant: Reactant
using Test

rebackend = ReactantBackend(AutoEnzyme())

test_differentiation(
    ReactantBackend(AutoEnzyme());
    excluded=[
        :derivative, :jacobian, :hessian, :hvp, :pullback, :pushforward, :second_derivative
    ],
    logging=LOGGING,
)
