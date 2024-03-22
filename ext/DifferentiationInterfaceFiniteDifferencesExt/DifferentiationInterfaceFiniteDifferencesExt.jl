module DifferentiationInterfaceFiniteDifferencesExt

using ADTypes: AutoFiniteDifferences
import DifferentiationInterface as DI
using FiniteDifferences
using LinearAlgebra: dot

function DI.value_and_pushforward(
    f::F, backend::AutoFiniteDifferences{fdm}, x::Number, dx::Number
) where {F,fdm}
    return backend.fdm(f, x) .* dx
end

end
