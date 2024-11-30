module DifferentiationInterfaceStaticArraysExt

using ADTypes: AutoForwardDiff, AutoEnzyme
import DifferentiationInterface as DI
using StaticArrays: SArray, StaticArray

function DI.stack_vec_col(t::NTuple{B,<:StaticArray}) where {B}
    return hcat(map(vec, t)...)
end

function DI.stack_vec_row(t::NTuple{B,<:StaticArray}) where {B}
    return vcat(transpose.(map(vec, t))...)
end

DI.ismutable_array(::Type{<:SArray}) = false

function DI.BatchSizeSettings(::AutoForwardDiff{nothing}, x::StaticArray)
    return BatchSizeSettings{length(x),true,true}(length(x))
end

function DI.BatchSizeSettings(::AutoEnzyme, x::StaticArray)
    return BatchSizeSettings{length(x),true,true}(length(x))
end

end
