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
        PushforwardScenario(
            num_to_num; x=x, ref=num_to_num_pushforward, operator=:outofplace
        ),
        PullbackScenario(num_to_num; x=x, ref=num_to_num_pullback, operator=:outofplace),
        DerivativeScenario(
            num_to_num; x=x, ref=num_to_num_derivative, operator=:outofplace
        ),
        SecondDerivativeScenario(
            num_to_num; x=x, ref=num_to_num_second_derivative, operator=:outofplace
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
    for op in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    _num_to_arr(a); x=x, ref=_num_to_arr_pushforward(a), operator=op
                ),
                DerivativeScenario(
                    _num_to_arr(a); x=x, ref=_num_to_arr_derivative(a), operator=op
                ),
                SecondDerivativeScenario(
                    _num_to_arr(a); x=x, ref=_num_to_arr_second_derivative(a), operator=op
                ),
            ],
        )
    end
    for op in (:outofplace,)
        append!(
            scens,
            [
                PullbackScenario(
                    _num_to_arr(a); x=x, ref=_num_to_arr_pullback(a), operator=op
                ),
            ],
        )
    end
    return scens
end

function num_to_arr_scenarios_twoarg(x::Number, a::AbstractArray)
    # pullback stays out of place
    scens = AbstractScenario[]
    for op in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    _num_to_arr!(a);
                    x=x,
                    y=similar(float.(a)),
                    ref=_num_to_arr_pushforward(a),
                    operator=op,
                ),
                DerivativeScenario(
                    _num_to_arr!(a);
                    x=x,
                    y=similar(float.(a)),
                    ref=_num_to_arr_derivative(a),
                    operator=op,
                ),
            ],
        )
    end
    for op in (:outofplace,)
        append!(
            scens,
            [
                PullbackScenario(
                    _num_to_arr!(a);
                    x=x,
                    y=similar(float.(a)),
                    ref=_num_to_arr_pullback(a),
                    operator=op,
                ),
            ],
        )
    end
    return scens
end

## Array to scalar

arr_to_num(x::AbstractArray)::Number = sum(sin, x)

arr_to_num_gradient(x) = cos.(x)
arr_to_num_hvp(x, v) = -sin.(x) .* v
arr_to_num_pushforward(x, dx) = dot(arr_to_num_gradient(x), dx)
arr_to_num_pullback(x, dy) = arr_to_num_gradient(x) .* dy
arr_to_num_hessian(x) = Matrix(Diagonal(-sin.(vec(x))))

function arr_to_num_scenarios_onearg(x::AbstractArray)
    # pushforward stays out of place
    scens = AbstractScenario[]
    for op in (:outofplace, :inplace)
        append!(
            scens,
            [
                PullbackScenario(arr_to_num; x=x, ref=arr_to_num_pullback, operator=op),
                GradientScenario(arr_to_num; x=x, ref=arr_to_num_gradient, operator=op),
                GradientScenario(arr_to_num; x=x, ref=arr_to_num_gradient, operator=op),
                HVPScenario(arr_to_num; x=x, ref=arr_to_num_hvp, operator=op),
                HessianScenario(arr_to_num; x=x, ref=arr_to_num_hessian, operator=op),
            ],
        )
    end
    for op in (:outofplace,)
        append!(
            scens,
            [PushforwardScenario(arr_to_num; x=x, ref=arr_to_num_pushforward, operator=op)],
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
    for op in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    vec_to_vec; x=x, ref=vec_to_vec_pushforward, operator=op
                ),
                PullbackScenario(vec_to_vec; x=x, ref=vec_to_vec_pullback, operator=op),
                JacobianScenario(vec_to_vec; x=x, ref=vec_to_vec_jacobian, operator=op),
            ],
        )
    end
    return scens
end

function vec_to_vec_scenarios_twoarg(x::AbstractVector)
    n = length(x)
    scens = AbstractScenario[]
    for op in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    vec_to_vec!;
                    x=x,
                    y=similar(x, 2n),
                    ref=vec_to_vec_pushforward,
                    operator=op,
                ),
                PullbackScenario(
                    vec_to_vec!; x=x, y=similar(x, 2n), ref=vec_to_vec_pullback, operator=op
                ),
                JacobianScenario(
                    vec_to_vec!; x=x, y=similar(x, 2n), ref=vec_to_vec_jacobian, operator=op
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
    for op in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    vec_to_mat; x=x, ref=vec_to_mat_pushforward, operator=op
                ),
                PullbackScenario(vec_to_mat; x=x, ref=vec_to_mat_pullback, operator=op),
                JacobianScenario(vec_to_mat; x=x, ref=vec_to_mat_jacobian, operator=op),
            ],
        )
    end
    return scens
end

function vec_to_mat_scenarios_twoarg(x::AbstractVector)
    n = length(x)
    scens = AbstractScenario[]
    for op in (:outofplace, :inplace)
        return [
            PushforwardScenario(
                vec_to_mat!;
                x=x,
                y=similar(x, n, 2),
                ref=vec_to_mat_pushforward,
                operator=op,
            ),
            PullbackScenario(
                vec_to_mat!; x=x, y=similar(x, n, 2), ref=vec_to_mat_pullback, operator=op
            ),
            JacobianScenario(
                vec_to_mat!; x=x, y=similar(x, n, 2), ref=vec_to_mat_jacobian, operator=op
            ),
        ]
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
    for op in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    mat_to_vec; x=x, ref=mat_to_vec_pushforward, operator=op
                ),
                PullbackScenario(
                    mat_to_vec; x=randn(m, n), ref=mat_to_vec_pullback, operator=op
                ),
                JacobianScenario(
                    mat_to_vec; x=randn(m, n), ref=mat_to_vec_jacobian, operator=op
                ),
            ],
        )
    end
    return scens
end

function mat_to_vec_scenarios_twoarg(x::AbstractMatrix)
    m, n = size(x)
    scens = AbstractScenario[]
    for op in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    mat_to_vec!;
                    x=x,
                    y=similar(x, m * n * 2),
                    ref=mat_to_vec_pushforward,
                    operator=op,
                ),
                PullbackScenario(
                    mat_to_vec!;
                    x=x,
                    y=similar(x, m * n * 2),
                    ref=mat_to_vec_pullback,
                    operator=op,
                ),
                JacobianScenario(
                    mat_to_vec!;
                    x=x,
                    y=similar(x, m * n * 2),
                    ref=mat_to_vec_jacobian,
                    operator=op,
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
    for op in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    mat_to_mat; x=x, ref=mat_to_mat_pushforward, operator=op
                ),
                PullbackScenario(mat_to_mat; x=x, ref=mat_to_mat_pullback, operator=op),
                JacobianScenario(mat_to_mat; x=x, ref=mat_to_mat_jacobian, operator=op),
            ],
        )
    end
    return scens
end

function mat_to_mat_scenarios_twoarg(x::AbstractMatrix)
    m, n = size(x)
    scens = AbstractScenario[]
    for op in (:outofplace, :inplace)
        append!(
            scens,
            [
                PushforwardScenario(
                    mat_to_mat!;
                    x=x,
                    y=similar(x, m * n, 2),
                    ref=mat_to_mat_pushforward,
                    operator=op,
                ),
                PullbackScenario(
                    mat_to_mat!;
                    x=x,
                    y=similar(x, m * n, 2),
                    ref=mat_to_mat_pullback,
                    operator=op,
                ),
                JacobianScenario(
                    mat_to_mat!;
                    x=x,
                    y=similar(x, m * n, 2),
                    ref=mat_to_mat_jacobian,
                    operator=op,
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
    default_scenarios()

Create a vector of [`AbstractScenario`](@ref)s with standard array types.
"""
function default_scenarios()
    return vcat(
        # one argument
        num_to_num_scenarios_onearg(randn()),
        num_to_arr_scenarios_onearg(randn(), IVEC),
        num_to_arr_scenarios_onearg(randn(), IMAT),
        arr_to_num_scenarios_onearg(randn(6)),
        arr_to_num_scenarios_onearg(randn(2, 3)),
        vec_to_vec_scenarios_onearg(randn(6)),
        vec_to_mat_scenarios_onearg(randn(6)),
        mat_to_vec_scenarios_onearg(randn(2, 3)),
        mat_to_mat_scenarios_onearg(randn(2, 3)),
        # two arguments
        num_to_arr_scenarios_twoarg(randn(), IVEC),
        num_to_arr_scenarios_twoarg(randn(), IMAT),
        vec_to_vec_scenarios_twoarg(randn(6)),
        vec_to_mat_scenarios_twoarg(randn(6)),
        mat_to_vec_scenarios_twoarg(randn(2, 3)),
        mat_to_mat_scenarios_twoarg(randn(2, 3)),
    )
end
