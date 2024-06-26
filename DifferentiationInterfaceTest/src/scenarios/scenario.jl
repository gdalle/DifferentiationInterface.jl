"""
    Scenario{op,args,pl}

Store a testing scenario composed of a function and its input + output for a given operator.

This generic type should never be used directly: use the specific constructor corresponding to the operator you want to test, or a predefined list of scenarios.

# Constructors

- [`PushforwardScenario`](@ref)
- [`PullbackScenario`](@ref)
- [`DerivativeScenario`](@ref)
- [`GradientScenario`](@ref)
- [`JacobianScenario`](@ref)
- [`SecondDerivativeScenario`](@ref)
- [`HVPScenario`](@ref)
- [`HessianScenario`](@ref)

# Type parameters

- `op`: one  of `:pushforward`, `:pullback`, `:derivative`, `:gradient`, `:jacobian`,`:second_derivative`, `:hvp`, `:hessian`
- `args`: either `1` (for `f(x) = y`) or `2` (for `f!(y, x) = nothing`)
- `pl`: either `:inplace` or `:outofplace`

# Fields

$(TYPEDFIELDS)
"""
struct Scenario{op,args,pl,F,X,Y,D,R1,R2}
    "function `f` (if `args==1`) or `f!` (if `args==2`) to apply"
    f::F
    "primal input"
    x::X
    "primal output"
    y::Y
    "seed for pushforward, pullback or HVP"
    seed::D
    "first-order result"
    res1::R1
    "second-order result"
    res2::R2

    function Scenario{op,args,pl}(
        f::F; x::X, y::Y, seed::D, res1::R1, res2::R2
    ) where {op,args,pl,F,X,Y,D,R1,R2}
        return new{op,args,pl,F,X,Y,D,R1,R2}(f, x, y, seed, res1, res2)
    end
end

operator(::Scenario{op}) where {op} = op
nb_args(::Scenario{op,args}) where {op,args} = args
place(::Scenario{op,args,pl}) where {op,args,pl} = pl

function order(
    ::Union{
        Scenario{:pushforward},
        Scenario{:pullback},
        Scenario{:derivative},
        Scenario{:gradient},
        Scenario{:jacobian},
    },
)
    return 1
end

function order(::Union{Scenario{:second_derivative},Scenario{:hvp},Scenario{:hessian}})
    return 2
end

function change_function(scen::Scenario{op,args,pl}, new_f) where {op,args,pl}
    return Scenario{op,args,pl}(
        new_f; x=scen.x, y=scen.y, seed=scen.seed, res1=scen.res1, res2=scen.res2
    )
end

maybe_zero(x::Number) = zero(x)
maybe_zero(x::AbstractArray) = zero(x)
maybe_zero(::Nothing) = nothing

function scenario_to_zero(scen::Scenario{op,args,pl}) where {op,args,pl}
    return Scenario{op,args,pl}(
        scen.f;
        x=scen.x,
        y=scen.y,
        seed=scen.seed,
        res1=maybe_zero(scen.res1),
        res2=maybe_zero(scen.res2),
    )
end

function compatible(backend::AbstractADType, scen::Scenario)
    if nb_args(scen) == 2
        return Bool(twoarg_support(backend))
    end
    return true
end

function group_by_operator(scenarios::AbstractVector{<:Scenario})
    return Dict(
        op => filter(s -> operator(s) == op, scenarios) for
        op in unique(operator.(scenarios))
    )
end

function Base.show(
    io::IO, scen::S
) where {op,args,pl,F,X,Y,D,S<:Scenario{op,args,pl,F,X,Y,D}}
    if D <: Nothing
        print(
            io,
            "Scenario{$(repr(op)),$(repr(args)),$(repr(pl))} with function $(repr(scen.f)) : $X -> $Y",
        )
    else
        print(
            io,
            "Scenario{$(repr(op)),$(repr(args)),$(repr(pl))} with function $(repr(scen.f)) : $X -> $Y and seed $D",
        )
    end
end

"""
$(SIGNATURES)
"""
function PushforwardScenario(f; x, y, dx, dy=nothing, nb_args, place=:inplace)
    return Scenario{:pushforward,nb_args,place}(f; x, y, seed=dx, res1=dy, res2=nothing)
end

"""
$(SIGNATURES)
"""
function PullbackScenario(f; x, y, dy, dx=nothing, nb_args, place=:inplace)
    return Scenario{:pullback,nb_args,place}(f; x, y, seed=dy, res1=dx, res2=nothing)
end

"""
$(SIGNATURES)
"""
function DerivativeScenario(f; x, y, der=nothing, nb_args, place=:inplace)
    return Scenario{:derivative,nb_args,place}(
        f; x, y, seed=nothing, res1=der, res2=nothing
    )
end

"""
$(SIGNATURES)
"""
function GradientScenario(f; x, y, grad=nothing, nb_args, place=:inplace)
    return Scenario{:gradient,nb_args,place}(f; x, y, seed=nothing, res1=grad, res2=nothing)
end

"""
$(SIGNATURES)
"""
function JacobianScenario(f; x, y, jac=nothing, nb_args, place=:inplace)
    return Scenario{:jacobian,nb_args,place}(f; x, y, seed=nothing, res1=jac, res2=nothing)
end

"""
$(SIGNATURES)
"""
function SecondDerivativeScenario(
    f; x, y, der=nothing, der2=nothing, nb_args, place=:inplace
)
    return Scenario{:second_derivative,nb_args,place}(
        f; x, y, seed=nothing, res1=der, res2=der2
    )
end

"""
$(SIGNATURES)
"""
function HVPScenario(f; x, y, dx, grad=nothing, dg=nothing, nb_args, place=:inplace)
    return Scenario{:hvp,nb_args,place}(f; x, y, seed=dx, res1=grad, res2=dg)
end

"""
$(SIGNATURES)
"""
function HessianScenario(f; x, y, grad=nothing, hess=nothing, nb_args, place=:inplace)
    return Scenario{:hessian,nb_args,place}(f; x, y, seed=nothing, res1=grad, res2=hess)
end
