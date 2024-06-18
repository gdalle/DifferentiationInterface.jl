#=
Constraints on the scenarios:
- non-allocating whenever possible
- type-stable
- GPU-compatible (no scalar indexing)
- vary shapes to be tricky
=#

first_half(v::AbstractVector) = @view v[1:(length(v) ÷ 2)]
second_half(v::AbstractVector) = @view v[(length(v) ÷ 2 + 1):end]

## Scalar to scalar

num_to_num(x::Number)::Number = sin(x)

num_to_num_derivative(x) = cos(x)
num_to_num_second_derivative(x) = -sin(x)
num_to_num_pushforward(x, dx) = num_to_num_derivative(x) * dx
num_to_num_pullback(x, dy) = num_to_num_derivative(x) * dy

function num_to_num_scenarios_onearg(x::Number; dx::Number, dy::Number)
    nb_args = 1
    place = :outofplace
    f = num_to_num
    y = f(x)
    dy_from_dx = num_to_num_pushforward(x, dx)
    dx_from_dy = num_to_num_pullback(x, dy)
    der = num_to_num_derivative(x)
    der2 = num_to_num_second_derivative(x)

    # everyone out of place
    return [
        PushforwardScenario(f; x, y, dx, dy=dy_from_dx, nb_args, place),
        PullbackScenario(f; x, y, dy, dx=dx_from_dy, nb_args, place),
        DerivativeScenario(f; x, y, der, nb_args, place),
        SecondDerivativeScenario(f; x, y, der, der2, nb_args, place),
    ]
end

## Scalar to array

num_to_arr_aux(x::Number, a::AbstractArray)::AbstractArray = sin.(a .* x)

function num_to_arr_aux!(y::AbstractArray, x::Number, a::AbstractArray)::Nothing
    y .= sin.(a .* x)
    return nothing
end

function _num_to_arr(a::AbstractArray)
    num_to_arr(x::Number) = num_to_arr_aux(x, a)
    return num_to_arr
end

function _num_to_arr!(a::AbstractArray)
    num_to_arr!(y::AbstractArray, x::Number) = num_to_arr_aux!(y, x, a)
    return num_to_arr!
end

function _num_to_arr_derivative(a)
    return (x) -> a .* cos.(a .* x)
end

function _num_to_arr_second_derivative(a)
    return (x) -> -(a .^ 2) .* sin.(a .* x)
end

function _num_to_arr_pushforward(a)
    return (x, dx) -> a .* cos.(a .* x) .* dx
end

function _num_to_arr_pullback(a)
    return (x, dy) -> dot(a .* cos.(a .* x), dy)
end

function num_to_arr_scenarios_onearg(
    x::Number, a::AbstractArray; dx::Number, dy::AbstractArray
)
    nb_args = 1
    f = _num_to_arr(a)
    y = f(x)
    der = _num_to_arr_derivative(a)(x)
    der2 = _num_to_arr_second_derivative(a)(x)
    dy_from_dx = _num_to_arr_pushforward(a)(x, dx)
    dx_from_dy = _num_to_arr_pullback(a)(x, dy)

    # pullback stays out of place
    scens = Scenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(f; x, y, dx, dy=dy_from_dx, nb_args, place),
                DerivativeScenario(f; x, y, der, nb_args, place),
                SecondDerivativeScenario(f; x, y, der, der2, nb_args, place),
            ],
        )
    end
    for place in (:outofplace,)
        append!(scens, [PullbackScenario(f; x, y, dy, dx=dx_from_dy, nb_args, place)])
    end
    return scens
end

function num_to_arr_scenarios_twoarg(
    x::Number, a::AbstractArray; dx::Number, dy::AbstractArray
)
    nb_args = 2
    f! = _num_to_arr!(a)
    y = similar(float.(a))
    f!(y, x)
    dy_from_dx = _num_to_arr_pushforward(a)(x, dx)
    dx_from_dy = _num_to_arr_pullback(a)(x, dy)
    der = _num_to_arr_derivative(a)(x)

    # pullback stays out of place
    scens = Scenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(f!; x, y, dx, dy=dy_from_dx, nb_args, place),
                DerivativeScenario(f!; x, y, der, nb_args, place),
            ],
        )
    end
    for place in (:outofplace,)
        append!(scens, [PullbackScenario(f!; x, y, dy, dx=dx_from_dy, nb_args, place)])
    end
    return scens
end

## Array to scalar

const DEFAULT_α = 4
const DEFAULT_β = 6

arr_to_num_aux_linalg(x; α, β) = sum(vec(x .^ α) .* transpose(vec(x .^ β)))

function arr_to_num_aux_no_linalg(x; α, β)
    n = length(x)
    s = zero(eltype(x))
    for i in 1:n, j in 1:n
        s += x[i]^α * x[j]^β
    end
    return s
end

function arr_to_num_aux_gradient(x; α, β)
    x = Array(x)  # GPU arrays don't like indexing
    g = similar(x)
    for k in eachindex(g, x)
        g[k] = (
            α * x[k]^(α - 1) * sum(x[j]^β for j in eachindex(x) if j != k) +
            β * x[k]^(β - 1) * sum(x[i]^α for i in eachindex(x) if i != k) +
            (α + β) * x[k]^(α + β - 1)
        )
    end
    return g
end

function arr_to_num_aux_hessian(x; α, β)
    x = Array(x)  # GPU arrays don't like indexing
    H = similar(x, length(x), length(x))
    for k in axes(H, 1), l in axes(H, 2)
        if k == l
            H[k, k] = (
                α * (α - 1) * x[k]^(α - 2) * sum(x[j]^β for j in eachindex(x) if j != k) +
                β * (β - 1) * x[k]^(β - 2) * sum(x[i]^α for i in eachindex(x) if i != k) +
                (α + β) * (α + β - 1) * x[k]^(α + β - 2)
            )
        else
            H[k, l] = α * β * (x[k]^(α - 1) * x[l]^(β - 1) + x[k]^(β - 1) * x[l]^(α - 1))
        end
    end
    return H
end

arr_to_num_linalg(x::AbstractArray)::Number =
    arr_to_num_aux_linalg(x; α=DEFAULT_α, β=DEFAULT_β)
arr_to_num_no_linalg(x::AbstractArray)::Number =
    arr_to_num_aux_no_linalg(x; α=DEFAULT_α, β=DEFAULT_β)

arr_to_num_gradient(x) = arr_to_num_aux_gradient(x; α=DEFAULT_α, β=DEFAULT_β)
arr_to_num_hessian(x) = arr_to_num_aux_hessian(x; α=DEFAULT_α, β=DEFAULT_β)
arr_to_num_pushforward(x, dx) = dot(arr_to_num_gradient(x), dx)
arr_to_num_pullback(x, dy) = arr_to_num_gradient(x) .* dy
arr_to_num_hvp(x, dx) = reshape(arr_to_num_hessian(x) * vec(dx), size(x))

function arr_to_num_scenarios_onearg(
    x::AbstractArray; dx::AbstractArray, dy::Number, linalg=true
)
    nb_args = 1
    f = linalg ? arr_to_num_linalg : arr_to_num_no_linalg
    y = f(x)
    dy_from_dx = arr_to_num_pushforward(x, dx)
    dx_from_dy = arr_to_num_pullback(x, dy)
    grad = arr_to_num_gradient(x)
    dg = arr_to_num_hvp(x, dx)
    hess = arr_to_num_hessian(x)

    # pushforward stays out of place
    scens = Scenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PullbackScenario(f; x, y, dy, dx=dx_from_dy, nb_args, place),
                GradientScenario(f; x, y, grad, nb_args, place),
                HVPScenario(f; x, y, dx, grad, dg, nb_args, place),
                HessianScenario(f; x, y, grad, hess, nb_args, place),
            ],
        )
    end
    for place in (:outofplace,)
        append!(scens, [PushforwardScenario(f; x, y, dx, dy=dy_from_dx, nb_args, place)])
    end
    return scens
end

## Array to array

function all_array_to_array_scenarios(f; x, y, dx, dy, dy_from_dx, dx_from_dy, jac, nb_args)
    scens = Scenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(f; x, y, dx, dy=dy_from_dx, nb_args, place),
                PullbackScenario(f; x, y, dy, dx=dx_from_dy, nb_args, place),
                JacobianScenario(f; x, y, jac, nb_args, place),
            ],
        )
    end
    return scens
end

vec_to_vec(x::AbstractVector)::AbstractVector = vcat(sin.(x), cos.(x))

function vec_to_vec!(y::AbstractVector, x::AbstractVector)
    y[1:length(x)] .= sin.(x)
    y[(length(x) + 1):(2length(x))] .= cos.(x)
    return nothing
end

vec_to_vec_pushforward(x, dx) = vcat(cos.(x) .* dx, -sin.(x) .* dx)
vec_to_vec_pullback(x, dy) = cos.(x) .* first_half(dy) .- sin.(x) .* second_half(dy)
vec_to_vec_jacobian(x) = vcat(Diagonal(cos.(x)), Diagonal(-sin.(x)))

function vec_to_vec_scenarios_onearg(
    x::AbstractVector; dx::AbstractVector, dy::AbstractVector
)
    nb_args = 1
    f = vec_to_vec
    y = f(x)
    dy_from_dx = vec_to_vec_pushforward(x, dx)
    dx_from_dy = vec_to_vec_pullback(x, dy)
    jac = vec_to_vec_jacobian(x)

    return all_array_to_array_scenarios(
        f; x, y, dx, dy, dy_from_dx, dx_from_dy, jac, nb_args
    )
end

function vec_to_vec_scenarios_twoarg(
    x::AbstractVector; dx::AbstractVector, dy::AbstractVector
)
    nb_args = 2
    f! = vec_to_vec!
    y = similar(vec_to_vec(x))
    f!(y, x)
    dy_from_dx = vec_to_vec_pushforward(x, dx)
    dx_from_dy = vec_to_vec_pullback(x, dy)
    jac = vec_to_vec_jacobian(x)

    return all_array_to_array_scenarios(
        f!; x, y, dx, dy, dy_from_dx, dx_from_dy, jac, nb_args
    )
end

vec_to_mat(x::AbstractVector)::AbstractMatrix = hcat(sin.(x), cos.(x))

function vec_to_mat!(y::AbstractMatrix, x::AbstractVector)
    y[:, 1] .= sin.(x)
    y[:, 2] .= cos.(x)
    return nothing
end

vec_to_mat_pushforward(x, dx) = hcat(cos.(x) .* dx, -sin.(x) .* dx)
vec_to_mat_pullback(x, dy) = cos.(x) .* dy[:, 1] .- sin.(x) .* dy[:, 2]
vec_to_mat_jacobian(x) = vcat(Diagonal(cos.(x)), Diagonal(-sin.(x)))

function vec_to_mat_scenarios_onearg(
    x::AbstractVector; dx::AbstractVector, dy::AbstractMatrix
)
    nb_args = 1
    f = vec_to_mat
    y = f(x)
    dy_from_dx = vec_to_mat_pushforward(x, dx)
    dx_from_dy = vec_to_mat_pullback(x, dy)
    jac = vec_to_mat_jacobian(x)

    return all_array_to_array_scenarios(
        f; x, y, dx, dy, dy_from_dx, dx_from_dy, jac, nb_args
    )
end

function vec_to_mat_scenarios_twoarg(
    x::AbstractVector; dx::AbstractVector, dy::AbstractMatrix
)
    nb_args = 2
    f! = vec_to_mat!
    y = similar(vec_to_mat(x))
    f!(y, x)
    dy_from_dx = vec_to_mat_pushforward(x, dx)
    dx_from_dy = vec_to_mat_pullback(x, dy)
    jac = vec_to_mat_jacobian(x)

    return all_array_to_array_scenarios(
        f!; x, y, dx, dy, dy_from_dx, dx_from_dy, jac, nb_args
    )
end

mat_to_vec(x::AbstractMatrix)::AbstractVector = vcat(vec(sin.(x)), vec(cos.(x)))

function mat_to_vec!(y::AbstractVector, x::AbstractMatrix)
    n = length(x)
    y[1:n] .= sin.(getindex.(Ref(x), 1:n))
    y[(n + 1):(2n)] .= cos.(getindex.(Ref(x), 1:n))
    return nothing
end

function mat_to_vec_pushforward(x, dx)
    return vcat(vec(cos.(x) .* dx), vec(-sin.(x) .* dx))
end

function mat_to_vec_pullback(x, dy)
    return cos.(x) .* reshape(first_half(dy), size(x)) .-
           sin.(x) .* reshape(second_half(dy), size(x))
end

mat_to_vec_jacobian(x) = vcat(Diagonal(vec(cos.(x))), Diagonal(vec(-sin.(x))))

function mat_to_vec_scenarios_onearg(
    x::AbstractMatrix; dx::AbstractMatrix, dy::AbstractVector
)
    nb_args = 1
    f = mat_to_vec
    y = f(x)
    dy_from_dx = mat_to_vec_pushforward(x, dx)
    dx_from_dy = mat_to_vec_pullback(x, dy)
    jac = mat_to_vec_jacobian(x)

    return all_array_to_array_scenarios(
        f; x, y, dx, dy, dy_from_dx, dx_from_dy, jac, nb_args
    )
end

function mat_to_vec_scenarios_twoarg(
    x::AbstractMatrix; dx::AbstractMatrix, dy::AbstractVector
)
    nb_args = 2
    f! = mat_to_vec!
    y = similar(mat_to_vec(x))
    f!(y, x)
    dy_from_dx = mat_to_vec_pushforward(x, dx)
    dx_from_dy = mat_to_vec_pullback(x, dy)
    jac = mat_to_vec_jacobian(x)

    return all_array_to_array_scenarios(
        f!; x, y, dx, dy, dy_from_dx, dx_from_dy, jac, nb_args
    )
end

mat_to_mat(x::AbstractMatrix)::AbstractMatrix = hcat(vec(sin.(x)), vec(cos.(x)))

function mat_to_mat!(y::AbstractMatrix, x::AbstractMatrix)
    n = length(x)
    y[:, 1] .= sin.(getindex.(Ref(x), 1:n))
    y[:, 2] .= cos.(getindex.(Ref(x), 1:n))
    return nothing
end

function mat_to_mat_pushforward(x, dx)
    return hcat(vec(cos.(x) .* dx), vec(-sin.(x) .* dx))
end

function mat_to_mat_pullback(x, dy)
    return cos.(x) .* reshape(dy[:, 1], size(x)) .- sin.(x) .* reshape(dy[:, 2], size(x))
end

mat_to_mat_jacobian(x) = vcat(Diagonal(vec(cos.(x))), Diagonal(vec(-sin.(x))))

function mat_to_mat_scenarios_onearg(
    x::AbstractMatrix; dx::AbstractMatrix, dy::AbstractMatrix
)
    nb_args = 1
    f = mat_to_mat
    y = f(x)
    dy_from_dx = mat_to_mat_pushforward(x, dx)
    dx_from_dy = mat_to_mat_pullback(x, dy)
    jac = mat_to_mat_jacobian(x)

    return all_array_to_array_scenarios(
        f; x, y, dx, dy, dy_from_dx, dx_from_dy, jac, nb_args
    )
end

function mat_to_mat_scenarios_twoarg(
    x::AbstractMatrix; dx::AbstractMatrix, dy::AbstractMatrix
)
    nb_args = 2
    f! = mat_to_mat!
    y = similar(mat_to_mat(x))
    f!(y, x)
    dy_from_dx = mat_to_mat_pushforward(x, dx)
    dx_from_dy = mat_to_mat_pullback(x, dy)
    jac = mat_to_mat_jacobian(x)

    return all_array_to_array_scenarios(
        f!; x, y, dx, dy, dy_from_dx, dx_from_dy, jac, nb_args
    )
end

## Gather

"""
    default_scenarios(rng=Random.default_rng(); linalg::Bool=true)

Create a vector of [`Scenario`](@ref)s with standard array types.

The option `linalg` controls whether scenarios are allowed to use functions from the LinearAlgebra standard library.
"""
function default_scenarios(rng::AbstractRNG=default_rng(); linalg=true)
    v = float.(Vector(1:6))
    m = float.(Matrix((1:2) .* transpose(1:3)))

    x_ = rand(rng)
    dx_ = rand(rng)
    dy_ = rand(rng)

    x_6 = rand(rng, 6)
    dx_6 = rand(rng, 6)

    x_2_3 = rand(rng, 2, 3)
    dx_2_3 = rand(rng, 2, 3)

    dy_12 = rand(rng, 12)
    dy_6_2 = rand(rng, 6, 2)

    dy_v = mycopy_random(rng, v)
    dy_m = mycopy_random(rng, m)

    return vcat(
        # one argument
        num_to_num_scenarios_onearg(x_; dx=dx_, dy=dy_),
        num_to_arr_scenarios_onearg(x_, v; dx=dx_, dy=dy_v),
        num_to_arr_scenarios_onearg(x_, m; dx=dx_, dy=dy_m),
        arr_to_num_scenarios_onearg(x_6; dx=dx_6, dy=dy_, linalg),
        arr_to_num_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_, linalg),
        vec_to_vec_scenarios_onearg(x_6; dx=dx_6, dy=dy_12),
        vec_to_mat_scenarios_onearg(x_6; dx=dx_6, dy=dy_6_2),
        mat_to_vec_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_12),
        mat_to_mat_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_6_2),
        # two arguments
        num_to_arr_scenarios_twoarg(x_, v; dx=dx_, dy=dy_v),
        num_to_arr_scenarios_twoarg(x_, m; dx=dx_, dy=dy_m),
        vec_to_vec_scenarios_twoarg(x_6; dx=dx_6, dy=dy_12),
        vec_to_mat_scenarios_twoarg(x_6; dx=dx_6, dy=dy_6_2),
        mat_to_vec_scenarios_twoarg(x_2_3; dx=dx_2_3, dy=dy_12),
        mat_to_mat_scenarios_twoarg(x_2_3; dx=dx_2_3, dy=dy_6_2),
    )
end
