using DifferentiationInterface
using Enzyme: Enzyme
using Reactant: Reactant
using Test

ReactantBackend =
    Base.get_extension(
        DifferentiationInterface, :DifferentiationInterfaceReactantExt
    ).ReactantBackend

rebackend = ReactantBackend(AutoEnzyme())

@test gradient(sum, rebackend, rand(3)) ≈ ones(3)
