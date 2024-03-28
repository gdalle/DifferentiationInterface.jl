abstract type AbstractScenario end

"""
    PushforwardScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct PushforwardScenario{mutating,F,X,Y,DX,DY} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "pushforward seed"
    dx::DX
    "Reference pushforward to compare against"
    pushforward::DY
end

function PushforwardScenario{mutating}(
    f::F, x::X, y::Y, dx::DX, pushforward::DY
) where {mutating,F,X,Y,DX,DY}
    return PushforwardScenario{mutating,F,X,Y,DX,DY}(f, x, y, dx, pushforward)
end

"""
    PullbackScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct PullbackScenario{mutating,F,X,Y,DX,DY} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "pullback seed"
    dy::DY
    "Reference pullback to compare against"
    pullback::DX
end

function PullbackScenario{mutating}(
    f::F, x::X, y::Y, dy::DY, pullback::DX
) where {mutating,F,X,Y,DX,DY}
    return PullbackScenario{mutating,F,X,Y,DX,DY}(f, x, y, dy, pullback)
end

"""
    DerivativeScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct DerivativeScenario{mutating,F,X<:Number,Y} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Reference derivative to compare against."
    derivative::Y
end

function DerivativeScenario{mutating}(
    f::F, x::X, y::Y, derivative::Y
) where {mutating,F,X,Y}
    return DerivativeScenario{mutating,F,X,Y}(f, x, y, derivative)
end

"""
    GradientScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct GradientScenario{mutating,F,X,Y<:Number} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Reference gradient to compare against."
    gradient::X
end

function GradientScenario{mutating}(f::F, x::X, y::Y, gradient::X) where {mutating,F,X,Y}
    return GradientScenario{mutating,F,X,Y}(f, x, y, gradient)
end

"""
    JacobianScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct JacobianScenario{mutating,F,X<:AbstractArray,Y<:AbstractArray,J<:AbstractMatrix} <:
       AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Reference Jacobian to compare against."
    jacobian::J
end

function JacobianScenario{mutating}(f::F, x::X, y::Y, jacobian::J) where {mutating,F,X,Y,J}
    return JacobianScenario{mutating,F,X,Y,J}(f, x, y, jacobian)
end

"""
    SecondDerivativeScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct SecondDerivativeScenario{mutating,F,X,Y} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Reference second-derivative to compare against."
    second_derivative::Y
end

function SecondDerivativeScenario{mutating}(
    f::F, x::X, y::Y, second_derivative::Y
) where {mutating,F,X,Y}
    return SecondDerivativeScenario{mutating,F,X,Y}(f, x, y, second_derivative)
end

"""
    HVPScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct HVPScenario{mutating,F,X,Y<:Number} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Reference Hessian-vector product to compare against."
    hvp::X
end

function HVPScenario{mutating}(f::F, x::X, y::Y, hvp::X) where {mutating,F,X,Y}
    return HVPScenario{mutating,F,X,Y}(f, x, y, hvp)
end

"""
    HessianScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct HessianScenario{mutating,F,X<:AbstractArray,Y<:Number,H<:AbstractMatrix} <:
       AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Reference Hessian to compare against."
    hessian::H
end

function HessianScenario{mutating}(f::F, x::X, y::Y, hessian::H) where {mutating,F,X,Y,H}
    return HessianScenario{mutating,F,X,Y,H}(f, x, y, hessian)
end

## Utilities & Pretty-printing

for S in (
    :PushforwardScenario,
    :PullbackScenario,
    :DerivativeScenario,
    :GradientScenario,
    :JacobianScenario,
    :SecondDerivativeScenario,
    :HVPScenario,
    :HessianScenario,
)
    @eval begin
        is_mutating(::($S){mutating}) where {mutating} = mutating

        function Base.string(scen::($S){mutating,F,X,Y}) where {mutating,F,X,Y}
            return "$S on $(string(scen.f)): $X -> $Y"
        end
    end
end

# function change_ref(scen::Scenario{mutating}, new_ref::AbstractADType) where {mutating}
#     return Scenario{mutating}(scen.f, scen.x, scen.y, scen.dx, scen.dy, new_ref)
# end

# function PushforwardScenario(f; x, y=nothing, pushforward=nothing)
#     if isnothing(y)
#         y = f(x)
#         dx = mysimilar_random(x)
#         dy = mysimilar_random(y)
#         return PushforwardScenario{false}(f, x, y, dx, dy, ref)
#     else
#         f! = f
#         f!(y, x)
#         dx = mysimilar_random(x)
#         dy = mysimilar_random(y)
#         return PushforwardScenario{true}(f!, x, y, dx, dy, ref)
#     end
# end
