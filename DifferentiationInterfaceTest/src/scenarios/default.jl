#=
Constraints on the scenarios:
- type-stable
- GPU-compatible (no scalar indexing)
- vary shapes to be tricky
=#

first_half(v::AbstractVector) = @view v[1:(length(v) ÷ 2)]
second_half(v::AbstractVector) = @view v[(length(v) ÷ 2 + 1):end]

## Number to number

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

## Number to array

multiplicator(::Type{V}) where {V<:AbstractVector} = convert(V, float.(1:6))
multiplicator(::Type{M}) where {M<:AbstractMatrix} = convert(M, float.(reshape(1:6, 2, 3)))

function num_to_arr(x::Number, ::Type{A}) where {A<:AbstractArray}
    a = multiplicator(A)
    return sin.(x .* a)
end

num_to_arr_vector(x) = num_to_arr(x, Vector{Float64})
num_to_arr_svector(x) = num_to_arr(x, SVector{6,Float64})
num_to_arr_jlvector(x) = num_to_arr(x, JLArray{Float64,1})

num_to_arr_matrix(x) = num_to_arr(x, Matrix{Float64})
num_to_arr_smatrix(x) = num_to_arr(x, SMatrix{2,3,Float64,6})
num_to_arr_jlmatrix(x) = num_to_arr(x, JLArray{Float64,2})

function pick_num_to_arr(::Type{A}) where {A<:AbstractArray}
    if A <: Vector
        return num_to_arr_vector
    elseif A <: SVector
        return num_to_arr_svector
    elseif A <: JLArray{<:Any,1}
        return num_to_arr_jlvector
    elseif A <: Matrix
        return num_to_arr_matrix
    elseif A <: SMatrix
        return num_to_arr_smatrix
    elseif A <: JLArray{<:Any,2}
        return num_to_arr_jlmatrix
    else
        throw(ArgumentError("Array type $A not supported"))
    end
end

function num_to_arr!(y::AbstractArray, x::Number)::Nothing
    a = multiplicator(typeof(y))
    y .= sin.(x .* a)
    return nothing
end

function num_to_arr_derivative(x, ::Type{A}) where {A}
    a = multiplicator(A)
    return a .* cos.(a .* x)
end

function num_to_arr_second_derivative(x, ::Type{A}) where {A}
    a = multiplicator(A)
    return -(a .^ 2) .* sin.(a .* x)
end

function num_to_arr_pushforward(x, dx, ::Type{A}) where {A}
    a = multiplicator(A)
    return a .* cos.(a .* x) .* dx
end

function num_to_arr_pullback(x, dy, ::Type{A}) where {A}
    a = multiplicator(A)
    return dot(a .* cos.(a .* x), dy)
end

function num_to_arr_scenarios_onearg(
    x::Number, ::Type{A}; dx::Number, dy::AbstractArray
) where {A<:AbstractArray}
    nb_args = 1
    f = pick_num_to_arr(A)
    y = f(x)
    dy_from_dx = num_to_arr_pushforward(x, dx, A)
    dx_from_dy = num_to_arr_pullback(x, dy, A)
    der = num_to_arr_derivative(x, A)
    der2 = num_to_arr_second_derivative(x, A)

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
    x::Number, ::Type{A}; dx::Number, dy::AbstractArray
) where {A<:AbstractArray}
    nb_args = 2
    f! = num_to_arr!
    y = similar(num_to_arr(x, A))
    f!(y, x)
    dy_from_dx = num_to_arr_pushforward(x, dx, A)
    dx_from_dy = num_to_arr_pullback(x, dy, A)
    der = num_to_arr_derivative(x, A)

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

### Number to matrix

## Array to number

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

const DEFAULT_α = 4
const DEFAULT_β = 6

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

### Vector to vector

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

### Vector to matrix

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

### Matrix to vector

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

### Matrix to matrix

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
    default_scenarios(rng=Random.default_rng())

Create a vector of [`Scenario`](@ref)s with standard array types.
"""
function default_scenarios(rng::AbstractRNG=default_rng(); linalg=true)
    x_ = rand(rng)
    dx_ = rand(rng)
    dy_ = rand(rng)

    x_6 = rand(rng, 6)
    dx_6 = rand(rng, 6)

    x_2_3 = rand(rng, 2, 3)
    dx_2_3 = rand(rng, 2, 3)

    dy_6 = rand(rng, 6)
    dy_12 = rand(rng, 12)
    dy_2_3 = rand(rng, 2, 3)
    dy_6_2 = rand(rng, 6, 2)

    V = Vector{Float64}
    M = Matrix{Float64}

    scens = vcat(
        # one argument
        num_to_num_scenarios_onearg(x_; dx=dx_, dy=dy_),
        num_to_arr_scenarios_onearg(x_, V; dx=dx_, dy=dy_6),
        num_to_arr_scenarios_onearg(x_, M; dx=dx_, dy=dy_2_3),
        arr_to_num_scenarios_onearg(x_6; dx=dx_6, dy=dy_, linalg),
        arr_to_num_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_, linalg),
        vec_to_vec_scenarios_onearg(x_6; dx=dx_6, dy=dy_12),
        vec_to_mat_scenarios_onearg(x_6; dx=dx_6, dy=dy_6_2),
        mat_to_vec_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_12),
        mat_to_mat_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_6_2),
        # two arguments
        num_to_arr_scenarios_twoarg(x_, V; dx=dx_, dy=dy_6),
        num_to_arr_scenarios_twoarg(x_, M; dx=dx_, dy=dy_2_3),
        vec_to_vec_scenarios_twoarg(x_6; dx=dx_6, dy=dy_12),
        vec_to_mat_scenarios_twoarg(x_6; dx=dx_6, dy=dy_6_2),
        mat_to_vec_scenarios_twoarg(x_2_3; dx=dx_2_3, dy=dy_12),
        mat_to_mat_scenarios_twoarg(x_2_3; dx=dx_2_3, dy=dy_6_2),
    )
    return scens
end
