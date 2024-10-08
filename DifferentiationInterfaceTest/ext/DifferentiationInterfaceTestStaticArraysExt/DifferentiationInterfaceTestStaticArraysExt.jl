module DifferentiationInterfaceTestStaticArraysExt

import DifferentiationInterface as DI
using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Random: AbstractRNG, default_rng
using SparseArrays: SparseArrays, SparseMatrixCSC, nnz, spdiagm
using StaticArrays: MArray, MMatrix, MVector, SArray, SMatrix, SVector

mySArray(f::Function) = f
myMArray(f::Function) = f

mySArray(::DIT.NumToArr{A}) where {T,A<:AbstractVector{T}} = DIT.NumToArr(SVector{6,T})
myMArray(::DIT.NumToArr{A}) where {T,A<:AbstractVector{T}} = DIT.NumToArr(MVector{6,T})

mySArray(::DIT.NumToArr{A}) where {T,A<:AbstractMatrix{T}} = DIT.NumToArr(SMatrix{2,3,T,6})
myMArray(::DIT.NumToArr{A}) where {T,A<:AbstractMatrix{T}} = DIT.NumToArr(MMatrix{2,3,T,6})

mySArray(f::DIT.MultiplyByConstant) = f
myMArray(f::DIT.MultiplyByConstant) = f

mySArray(f::DIT.WritableClosure) = f
myMArray(f::DIT.WritableClosure) = f

mySArray(x::Number) = x
myMArray(x::Number) = x

mySArray(x::AbstractVector{T}) where {T} = convert(SVector{length(x),T}, x)
myMArray(x::AbstractVector{T}) where {T} = convert(MVector{length(x),T}, x)

function mySArray(x::AbstractMatrix{T}) where {T}
    return convert(SMatrix{size(x, 1),size(x, 2),T,length(x)}, x)
end
function myMArray(x::AbstractMatrix{T}) where {T}
    return convert(MMatrix{size(x, 1),size(x, 2),T,length(x)}, x)
end

mySArray(x::Tuple) = map(mySArray, x)
myMArray(x::Tuple) = map(myMArray, x)

mySArray(x::DI.Constant) = DI.Constant(mySArray(DI.unwrap(x)))
myMArray(x::DI.Constant) = DI.Constant(myMArray(DI.unwrap(x)))

mySArray(::Nothing) = nothing
myMArray(::Nothing) = nothing

function mySArray(scen::Scenario{op,pl_op,pl_fun}) where {op,pl_op,pl_fun}
    (; f, x, y, tang, contexts, res1, res2) = scen
    return Scenario{op,pl_op,pl_fun}(
        mySArray(f);
        x=mySArray(x),
        y=pl_fun == :in ? myMArray(y) : mySArray(y),
        tang=mySArray(tang),
        contexts=mySArray(contexts),
        res1=mySArray(res1),
        res2=mySArray(res2),
    )
end

function myMArray(scen::Scenario{op,pl_op,pl_fun}) where {op,pl_op,pl_fun}
    (; f, x, y, tang, contexts, res1, res2) = scen
    return Scenario{op,pl_op,pl_fun}(
        myMArray(f);
        x=myMArray(x),
        y=pl_fun == :in ? myMArray(y) : myMArray(y),
        tang=myMArray(tang),
        contexts=myMArray(contexts),
        res1=myMArray(res1),
        res2=myMArray(res2),
    )
end

function DIT.static_scenarios(args...; kwargs...)
    scens = DIT.default_scenarios(args...; kwargs...)
    return vcat(mySArray.(scens), myMArray.(scens))
end

end
