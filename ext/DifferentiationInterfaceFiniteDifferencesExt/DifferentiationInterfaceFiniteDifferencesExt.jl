module DifferentiationInterfaceFiniteDifferencesExt

using ADTypes: AutoFiniteDifferences
import DifferentiationInterface as DI
using FillArrays: OneElement
using FiniteDifferences: FiniteDifferences, jvp, j′vp
using LinearAlgebra: dot

DI.supports_mutation(::AutoFiniteDifferences) = DI.MutationNotSupported()

function FiniteDifferences.to_vec(a::OneElement)  # TODO: remove type piracy (https://github.com/JuliaDiff/FiniteDifferences.jl/issues/141)
    return FiniteDifferences.to_vec(collect(a))
end

function DI.value_and_pushforward(
    f, backend::AutoFiniteDifferences{fdm}, x, dx, extras::Nothing
) where {fdm}
    y = f(x)
    return y, jvp(backend.fdm, f, (x, dx))
end

#=
# TODO: why does this fail?

function DI.value_and_pullback(
    f, backend::AutoFiniteDifferences{fdm}, x, dy, extras::Nothing
) where {fdm}
    y = f(x)
    return y, j′vp(backend.fdm, f, x, dy)[1]
end
=#

end
