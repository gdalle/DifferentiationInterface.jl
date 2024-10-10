module DifferentiationInterfaceStaticArraysExt

import DifferentiationInterface as DI
using StaticArrays: SArray

function DI.stack_vec_col(t::NTuple{B,<:SArray}) where {B}
    return hcat(map(vec, t)...)
end

end
