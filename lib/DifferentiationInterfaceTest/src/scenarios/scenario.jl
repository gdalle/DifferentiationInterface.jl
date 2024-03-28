abstract type AbstractScenario end
# Subtypes are expected to be parametric types `MyScenario{mutating, F, X, Y, Result, ...}`

"""
    PushforwardScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct PushforwardScenario{mutating,F,X,Y,R,DX} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "pushforward seed"
    dx::DX
    "Reference pushforward to compare against"
    pushforward::R
end

function PushforwardScenario{mutating}(
    f::F, x::X; y::Y=nothing, dx::DX=nothing, pushforward::R=nothing
) where {mutating,F,X,Y,DX,R}
    return PushforwardScenario{mutating,F,X,Y,R,DX}(f, x, y, dx, pushforward)
end

"""
    PullbackScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct PullbackScenario{mutating,F,X,Y,R,DY} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "pullback seed"
    dy::DY
    "Reference pullback to compare against"
    pullback::R
end

function PullbackScenario{mutating}(
    f::F, x::X; y::Y=nothing, dy::DY=nothing, pullback::R=nothing
) where {mutating,F,X,Y,DX,R}
    return PullbackScenario{mutating,F,X,Y,R,DX}(f, x, y, dy, pullback)
end

"""
    DerivativeScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct DerivativeScenario{mutating,F,X,Y,R} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Reference derivative to compare against."
    derivative::R
end

function DerivativeScenario{mutating}(
    f::F, x::X; y::Y=nothing, derivative::R=nothing
) where {mutating,F,X,Y,R}
    return DerivativeScenario{mutating,F,X,Y,R}(f, x, y, derivative)
end

"""
    GradientScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct GradientScenario{mutating,F,X,Y,R} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Reference gradient to compare against."
    gradient::R
end

function GradientScenario{mutating}(
    f::F, x::X; y::Y=nothing, gradient::R=nothing
) where {mutating,F,X,Y,R}
    return GradientScenario{mutating,F,X,Y,R}(f, x, y, gradient)
end

"""
    JacobianScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct JacobianScenario{mutating,F,X<:AbstractArray,Y,R} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Reference Jacobian to compare against."
    jacobian::R
end

function JacobianScenario{mutating}(
    f::F, x::X; y::Y=nothing, jacobian::R=nothing
) where {mutating,F,X,Y,R}
    return JacobianScenario{mutating,F,X,Y,J}(f, x, y, jacobian)
end

"""
    SecondDerivativeScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct SecondDerivativeScenario{mutating,F,X,Y,R} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Reference second-derivative to compare against."
    second_derivative::R
end

function SecondDerivativeScenario{mutating}(
    f::F, x::X; y::Y=nothing, second_derivative::R=nothing
) where {mutating,F,X,Y,R}
    return SecondDerivativeScenario{mutating,F,X,Y,R}(f, x, y, second_derivative)
end

"""
    HVPScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct HVPScenario{mutating,F,X,Y,R} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Reference Hessian-vector product to compare against."
    hvp::R
end

function HVPScenario{mutating}(
    f::F, x::X; y::Y=nothing, hvp::R=nothing
) where {mutating,F,X,Y,R}
    return HVPScenario{mutating,F,X,Y,R}(f, x, y, hvp)
end

"""
    HessianScenario{mutating}

Store a testing scenario composed of a function and its input + primal output + result.

# Fields

$(TYPEDFIELDS)
"""
struct HessianScenario{mutating,F,X<:AbstractArray,Y,R} <: AbstractScenario
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Reference Hessian to compare against."
    hessian::R
end

function HessianScenario{mutating}(
    f::F, x::X; y::Y=nothing, hessian::H=nothing
) where {mutating,F,X,Y,H}
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
