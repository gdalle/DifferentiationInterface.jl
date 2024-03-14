
in_type(::Scenario{F,X}) where {F,X} = X
out_type(::Scenario{F,X,Y}) where {F,X,Y} = Y

function DT.make_scenario(rng::AbstractRNG, f, x)
    y = f(x)
    return make_scenario(rng, f, x, y)
end

## Auto fill

function DT.make_scenario(rng::AbstractRNG, f, x::Number, y::Number)
    dx = randn(rng, typeof(x))
    dy = randn(rng, typeof(y))
    der_true = ForwardDiff.derivative(f, x)
    dx_true = der_true * dy
    dy_true = der_true * dx
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, der_true)
end

function DT.make_scenario(rng::AbstractRNG, f, x::Number, y::AbstractArray)
    dx = randn(rng, typeof(x))
    dy = similar(y)
    randn!(rng, dy)
    multider_true = ForwardDiff.derivative(f, x)
    dx_true = dot(multider_true, dy)
    dy_true = multider_true .* dx
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, multider_true)
end

function DT.make_scenario(rng::AbstractRNG, f, x::AbstractArray, y::Number)
    dx = similar(x)
    randn!(rng, dx)
    dy = randn(rng, typeof(y))
    grad_true = ForwardDiff.gradient(f, x)
    dx_true = grad_true .* dy
    dy_true = dot(grad_true, dx)
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, grad_true)
end

function DT.make_scenario(rng::AbstractRNG, f, x::AbstractArray, y::AbstractArray)
    dx = similar(x)
    randn!(rng, dx)
    dy = similar(y)
    randn!(rng, dy)
    jac_true = ForwardDiff.jacobian(f, x)
    dx_true = reshape(transpose(jac_true) * vec(dy), size(x))
    dy_true = reshape(jac_true * vec(dx), size(y))
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, jac_true)
end

## Defaults

f_scalar_scalar(x::Number)::Number = sin(x)

f_scalar_vector(x::Number)::AbstractVector = [sin(x), sin(2x)]
f_scalar_matrix(x::Number)::AbstractMatrix = hcat([sin(x) cos(x)], [sin(2x) cos(2x)])

f_vector_scalar(x::AbstractVector)::Number = sum(sin, x)
f_matrix_scalar(x::AbstractMatrix)::Number = sum(sin, x)

f_vector_vector(x::AbstractVector)::AbstractVector = vcat(sin.(x), cos.(x))
f_vector_matrix(x::AbstractVector)::AbstractMatrix = hcat(sin.(x), cos.(x))

f_matrix_vector(x::AbstractMatrix)::AbstractVector = vcat(vec(sin.(x)), vec(cos.(x)))
f_matrix_matrix(x::AbstractMatrix)::AbstractMatrix = hcat(vec(sin.(x)), vec(cos.(x)))

function DT.default_scenarios(rng::AbstractRNG)
    scenarios = [
        make_scenario(rng, f_scalar_scalar, 1.0),
        make_scenario(rng, f_scalar_vector, 1.0),
        make_scenario(rng, f_scalar_matrix, 1.0),
        make_scenario(rng, f_vector_scalar, [1.0, 2.0]),
        make_scenario(rng, f_matrix_scalar, [1.0 2.0; 3.0 4.0]),
        make_scenario(rng, f_vector_vector, [1.0, 2.0]),
        make_scenario(rng, f_vector_matrix, [1.0, 2.0]),
        make_scenario(rng, f_matrix_vector, [1.0 2.0; 3.0 4.0]),
        make_scenario(rng, f_matrix_matrix, [1.0 2.0; 3.0 4.0]),
    ]
    return scenarios
end

DT.default_scenarios() = default_scenarios(default_rng())
