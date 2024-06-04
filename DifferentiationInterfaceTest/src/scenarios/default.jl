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

function num_to_num_scenarios_onearg(x::Number)
    # everyone out of place
    return [
        PushforwardScenario(num_to_num; x=x, ref=num_to_num_pushforward, place=:outofplace),
        PullbackScenario(num_to_num; x=x, ref=num_to_num_pullback, place=:outofplace),
        DerivativeScenario(num_to_num; x=x, ref=num_to_num_derivative, place=:outofplace),
        SecondDerivativeScenario(
            num_to_num; x=x, ref=num_to_num_second_derivative, place=:outofplace
        ),
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

function num_to_arr_scenarios_onearg(x::Number, a::AbstractArray)
    # pullback stays out of place
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    _num_to_arr(a); x=x, ref=_num_to_arr_pushforward(a), place=place
                ),
                DerivativeScenario(
                    _num_to_arr(a); x=x, ref=_num_to_arr_derivative(a), place=place
                ),
                SecondDerivativeScenario(
                    _num_to_arr(a);
                    x=x,
                    ref=_num_to_arr_second_derivative(a),
                    first_order_ref=_num_to_num_derivative(a),
                    place=place,
                ),
            ],
        )
    end
    for place in (:outofplace,)
        append!(
            scens,
            [
                PullbackScenario(
                    _num_to_arr(a); x=x, ref=_num_to_arr_pullback(a), place=place
                ),
            ],
        )
    end
    return scens
end

function num_to_arr_scenarios_twoarg(x::Number, a::AbstractArray)
    # pullback stays out of place
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    _num_to_arr!(a);
                    x=x,
                    y=similar(float.(a)),
                    ref=_num_to_arr_pushforward(a),
                    place=place,
                ),
                DerivativeScenario(
                    _num_to_arr!(a);
                    x=x,
                    y=similar(float.(a)),
                    ref=_num_to_arr_derivative(a),
                    place=place,
                ),
            ],
        )
    end
    for place in (:outofplace,)
        append!(
            scens,
            [
                PullbackScenario(
                    _num_to_arr!(a);
                    x=x,
                    y=similar(float.(a)),
                    ref=_num_to_arr_pullback(a),
                    place=place,
                ),
            ],
        )
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
arr_to_num_hvp(x, v) = reshape(arr_to_num_hessian(x) * vec(v), size(x))

function arr_to_num_scenarios_onearg(x::AbstractArray; linalg=true)
    arr_to_num = linalg ? arr_to_num_linalg : arr_to_num_no_linalg
    # pushforward stays out of place
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PullbackScenario(arr_to_num; x=x, ref=arr_to_num_pullback, place=place),
                GradientScenario(arr_to_num; x=x, ref=arr_to_num_gradient, place=place),
                HVPScenario(
                    arr_to_num;
                    x=x,
                    ref=arr_to_num_hvp,
                    first_order_ref=arr_to_num_gradient,
                    place=place,
                ),
                HessianScenario(
                    arr_to_num;
                    x=x,
                    ref=arr_to_num_hessian,
                    first_order_ref=arr_to_num_gradient,
                    place=place,
                ),
            ],
        )
    end
    for place in (:outofplace,)
        append!(
            scens,
            [PushforwardScenario(arr_to_num; x=x, ref=arr_to_num_pushforward, place=place)],
        )
    end
    return scens
end

## Array to array

vec_to_vec(x::AbstractVector)::AbstractVector = vcat(sin.(x), cos.(x))

function vec_to_vec!(y::AbstractVector, x::AbstractVector)
    y[1:length(x)] .= sin.(x)
    y[(length(x) + 1):(2length(x))] .= cos.(x)
    return nothing
end

vec_to_vec_pushforward(x, dx) = vcat(cos.(x) .* dx, -sin.(x) .* dx)
vec_to_vec_pullback(x, dy) = cos.(x) .* first_half(dy) .- sin.(x) .* second_half(dy)
vec_to_vec_jacobian(x) = vcat(Diagonal(cos.(x)), Diagonal(-sin.(x)))

function vec_to_vec_scenarios_onearg(x::AbstractVector)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    vec_to_vec; x=x, ref=vec_to_vec_pushforward, place=place
                ),
                PullbackScenario(vec_to_vec; x=x, ref=vec_to_vec_pullback, place=place),
                JacobianScenario(vec_to_vec; x=x, ref=vec_to_vec_jacobian, place=place),
            ],
        )
    end
    return scens
end

function vec_to_vec_scenarios_twoarg(x::AbstractVector)
    n = length(x)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    vec_to_vec!;
                    x=x,
                    y=similar(x, 2n),
                    ref=vec_to_vec_pushforward,
                    place=place,
                ),
                PullbackScenario(
                    vec_to_vec!; x=x, y=similar(x, 2n), ref=vec_to_vec_pullback, place=place
                ),
                JacobianScenario(
                    vec_to_vec!; x=x, y=similar(x, 2n), ref=vec_to_vec_jacobian, place=place
                ),
            ],
        )
    end
    return scens
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

function vec_to_mat_scenarios_onearg(x::AbstractVector)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    vec_to_mat; x=x, ref=vec_to_mat_pushforward, place=place
                ),
                PullbackScenario(vec_to_mat; x=x, ref=vec_to_mat_pullback, place=place),
                JacobianScenario(vec_to_mat; x=x, ref=vec_to_mat_jacobian, place=place),
            ],
        )
    end
    return scens
end

function vec_to_mat_scenarios_twoarg(x::AbstractVector)
    n = length(x)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    vec_to_mat!;
                    x=x,
                    y=similar(x, n, 2),
                    ref=vec_to_mat_pushforward,
                    place=place,
                ),
                PullbackScenario(
                    vec_to_mat!;
                    x=x,
                    y=similar(x, n, 2),
                    ref=vec_to_mat_pullback,
                    place=place,
                ),
                JacobianScenario(
                    vec_to_mat!;
                    x=x,
                    y=similar(x, n, 2),
                    ref=vec_to_mat_jacobian,
                    place=place,
                ),
            ],
        )
    end
    return scens
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

function mat_to_vec_scenarios_onearg(x::AbstractMatrix)
    m, n = size(x)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    mat_to_vec; x=x, ref=mat_to_vec_pushforward, place=place
                ),
                PullbackScenario(
                    mat_to_vec; x=randn(m, n), ref=mat_to_vec_pullback, place=place
                ),
                JacobianScenario(
                    mat_to_vec; x=randn(m, n), ref=mat_to_vec_jacobian, place=place
                ),
            ],
        )
    end
    return scens
end

function mat_to_vec_scenarios_twoarg(x::AbstractMatrix)
    m, n = size(x)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    mat_to_vec!;
                    x=x,
                    y=similar(x, m * n * 2),
                    ref=mat_to_vec_pushforward,
                    place=place,
                ),
                PullbackScenario(
                    mat_to_vec!;
                    x=x,
                    y=similar(x, m * n * 2),
                    ref=mat_to_vec_pullback,
                    place=place,
                ),
                JacobianScenario(
                    mat_to_vec!;
                    x=x,
                    y=similar(x, m * n * 2),
                    ref=mat_to_vec_jacobian,
                    place=place,
                ),
            ],
        )
    end
    return scens
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

function mat_to_mat_scenarios_onearg(x::AbstractMatrix)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    mat_to_mat; x=x, ref=mat_to_mat_pushforward, place=place
                ),
                PullbackScenario(mat_to_mat; x=x, ref=mat_to_mat_pullback, place=place),
                JacobianScenario(mat_to_mat; x=x, ref=mat_to_mat_jacobian, place=place),
            ],
        )
    end
    return scens
end

function mat_to_mat_scenarios_twoarg(x::AbstractMatrix)
    m, n = size(x)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    mat_to_mat!;
                    x=x,
                    y=similar(x, m * n, 2),
                    ref=mat_to_mat_pushforward,
                    place=place,
                ),
                PullbackScenario(
                    mat_to_mat!;
                    x=x,
                    y=similar(x, m * n, 2),
                    ref=mat_to_mat_pullback,
                    place=place,
                ),
                JacobianScenario(
                    mat_to_mat!;
                    x=x,
                    y=similar(x, m * n, 2),
                    ref=mat_to_mat_jacobian,
                    place=place,
                ),
            ],
        )
    end
    return scens
end

## Gather

const IVEC = Vector(1:6)
const IMAT = Matrix((1:2) .* transpose(1:3))

"""
    default_scenarios(rng=Random.default_rng())

Create a vector of [`AbstractScenario`](@ref)s with standard array types.
"""
function default_scenarios(rng::AbstractRNG=default_rng(); linalg=true)
    return vcat(
        # one argument
        num_to_num_scenarios_onearg(rand(rng)),
        num_to_arr_scenarios_onearg(rand(rng), IVEC),
        num_to_arr_scenarios_onearg(rand(rng), IMAT),
        arr_to_num_scenarios_onearg(rand(rng, 6); linalg),
        arr_to_num_scenarios_onearg(rand(rng, 2, 3); linalg),
        vec_to_vec_scenarios_onearg(rand(rng, 6)),
        vec_to_mat_scenarios_onearg(rand(rng, 6)),
        mat_to_vec_scenarios_onearg(rand(rng, 2, 3)),
        mat_to_mat_scenarios_onearg(rand(rng, 2, 3)),
        # two arguments
        num_to_arr_scenarios_twoarg(rand(rng), IVEC),
        num_to_arr_scenarios_twoarg(rand(rng), IMAT),
        vec_to_vec_scenarios_twoarg(rand(rng, 6)),
        vec_to_mat_scenarios_twoarg(rand(rng, 6)),
        mat_to_vec_scenarios_twoarg(rand(rng, 2, 3)),
        mat_to_mat_scenarios_twoarg(rand(rng, 2, 3)),
    )
end
