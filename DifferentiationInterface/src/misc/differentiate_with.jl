"""
    DifferentiateWith

Function wrapper that enforces differentiation with a substitute AD backend, different from the true AD backend that the user intends to call.

For instance, suppose `f` is not differentiable with Zygote because it involves mutation, but you know that it is differentiable with Enzyme.
Then `g = DifferentiateWith(f, AutoEnzyme())` is differentiable with Zygote thanks to a chain rule which calls Enzyme under the hood.
Moreover, any composition involving `g` will also be differentiable.

!!! warning
    `DifferentiateWith` only supports out-of-place functions `y = f(x)` without additional context arguments.
    It only makes these functions differentiable if the true backend is either [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) or anything [ChainRules](https://github.com/JuliaDiff/ChainRules.jl)-compatible.

# Fields

- `f`: the function in question, with signature `f(x)`
- `backend::AbstractADType`: the substitute backend to use for differentiation

!!! note
    For the substitute AD backend to be called under the hood, its package needs to be loaded in addition to the package of the true AD backend.

# Constructor

    DifferentiateWith(f, backend)

# Example

```jldoctest
julia> using DifferentiationInterface

julia> import Enzyme, ForwardDiff, Zygote

julia> function f(x::Vector{Float64})
           a = Vector{Float64}(undef, 1)  # type constraint breaks ForwardDiff
           a[1] = sum(abs2, x)  # mutation breaks Zygote
           return a[1]
       end;

julia> g = DifferentiateWith(f, AutoEnzyme());

julia> h(x) = 7 * g(x);

julia> ForwardDiff.gradient(h, [3.0, 5.0])
2-element Vector{Float64}:
 42.0
 70.0

julia> Zygote.gradient(h, [3.0, 5.0])[1]
2-element Vector{Float64}:
 42.0
 70.0
```
"""
struct DifferentiateWith{F,B<:AbstractADType}
    f::F
    backend::B
end

(dw::DifferentiateWith)(x) = dw.f(x)

function Base.show(io::IO, dw::DifferentiateWith)
    @compat (; f, backend) = dw
    return print(
        io,
        DifferentiateWith,
        "(",
        repr(f; context=io),
        ", ",
        repr(backend; context=io),
        ")",
    )
end
