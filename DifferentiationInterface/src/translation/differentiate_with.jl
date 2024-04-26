"""
    DifferentiateWith

Callable function wrapper that enforces differentiation with a specified (inner) backend.

This works by defining new rules overriding the behavior of the outer backend that would normally be used.

!!! warning
    This is an experimental functionality, whose API cannot yet be considered stable.
    At the moment, it only supports one-argument functions, and rules are only defined for [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)-compatible outer backends.

# Fields

- `f`: the function in question
- `backend::AbstractADType`: the inner backend to use for differentiation

# Constructor

    DifferentiateWith(f, backend)

# Example

```@repl
using DifferentiationInterface
import ForwardDiff, Zygote

function f(x)
    a = Vector{eltype(x)}(undef, 1)
    a[1] = sum(x)  # mutation that breaks Zygote
    return a[1]
end

dw = DifferentiateWith(f, AutoForwardDiff());

gradient(dw, AutoZygote(), [1.0, 2.0])  # works because it calls ForwardDiff instead
gradient(f, AutoZygote(), [1.0, 2.0])  # fails
```
"""
struct DifferentiateWith{F,B<:AbstractADType}
    f::F
    backend::B
end

"""
    (dw::DifferentiateWith)(x)

Call the underlying function `dw.f` of a [`DifferentiateWith`](@ref) wrapper.
"""
(dw::DifferentiateWith)(x) = dw.f(x)

function Base.show(io::IO, dw::DifferentiateWith)
    (; f, backend) = dw
    return print(io, "$f differentiated with $(backend_str(backend))")
end
