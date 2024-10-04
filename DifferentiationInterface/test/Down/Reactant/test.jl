using Pkg
Pkg.add("Enzyme")
Pkg.add(; url="https://github.com/EnzymeAD/Reactant.jl")

using DifferentiationInterface
using DifferentiationInterface: ReactantBackend
using DifferentiationInterfaceTest
using Enzyme: Enzyme
using LinearAlgebra
using Reactant: Reactant
using Test

@assert !isnothing(
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceReactantExt)
)

LOGGING = get(ENV, "CI", "false") == "false"

rebackend = ReactantBackend(AutoEnzyme())
x = rand(3)
xr = Reactant.to_rarray(x)
prep = prepare_gradient(sum, rebackend, x)

test_differentiation(
    ReactantBackend(AutoEnzyme()),
    default_scenarios(; linalg=true);
    excluded=[
        :derivative, :jacobian, :hessian, :hvp, :pullback, :pushforward, :second_derivative
    ],
    logging=LOGGING,
)
