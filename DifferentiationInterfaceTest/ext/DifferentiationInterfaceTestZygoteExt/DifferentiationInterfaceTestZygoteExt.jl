module DifferentiationInterfaceTestZygoteExt

using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Zygote

DIT.multiplicator(::Type{Zygote.Buffer{T,A}}) where {T,A} = DIT.multiplicator(A)

end
