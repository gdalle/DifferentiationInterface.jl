module DifferentiationInterfaceStaticArraysExt

using ADTypes: AutoForwardDiff, AutoEnzyme
import DifferentiationInterface as DI
using StaticArrays: SArray, StaticArray

function DI.stack_vec_col(t::NTuple{B,<:SArray}) where {B}
    return hcat(map(vec, t)...)
end

DI.adaptive_batchsize(::AutoForwardDiff{nothing}, a::StaticArray) = Val(length(a))
DI.adaptive_batchsize(::AutoEnzyme, a::StaticArray) = Val(length(a))

end
