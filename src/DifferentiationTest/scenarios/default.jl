#=
Constraints on the scenarios:
- non-allocating whenever possible
- type-stable
- GPU-compatible (no scalar indexing)
- vary shapes to be tricky
=#

first_half(v::AbstractVector) = @view v[1:(length(v) ÷ 2)]
second_half(v::AbstractVector) = @view v[(length(v) ÷ 2 + 1):end]

top_half(M::AbstractMatrix) = @view M[1:(size(M, 2) ÷ 2), :]
bottom_half(M::AbstractMatrix) = @view M[(size(M, 2) ÷ 2 + 1):end, :]

left_half(M::AbstractMatrix) = @view M[:, 1:(size(M, 2) ÷ 2)]
right_half(M::AbstractMatrix) = @view M[:, (size(M, 2) ÷ 2 + 1):end]

## Scalar to scalar

scalar_to_scalar(x::Number)::Number = sin(x)

scalar_to_scalar_derivative(x) = cos(x)
scalar_to_scalar_second_derivative(x) = -sin(x)
scalar_to_scalar_pushforward(x, dx) = scalar_to_scalar_derivative(x) * dx
scalar_to_scalar_pullback(x, dy) = scalar_to_scalar_derivative(x) * dy
scalar_to_scalar_gradient(x) = scalar_to_scalar_derivative(x)
scalar_to_scalar_hvp(x, v) = scalar_to_scalar_second_derivative(x) * v

function scalar_to_scalar_ref()
    return Reference(;
        pushforward=scalar_to_scalar_pushforward,
        pullback=scalar_to_scalar_pullback,
        derivative=scalar_to_scalar_derivative,
        gradient=scalar_to_scalar_gradient,
        second_derivative=scalar_to_scalar_second_derivative,
        hvp=scalar_to_scalar_hvp,
    )
end

## Scalar to array

_scalar_to_array(x::Number, scaling::AbstractArray)::AbstractArray = sin.(scaling .* x)

function _scalar_to_array!(y::AbstractArray, x::Number, scaling::AbstractArray)::Nothing
    y .= sin.(scaling .* x)
    return nothing
end

function make_scalar_to_array(scaling::AbstractArray)
    scalar_to_array(x::Number) = _scalar_to_array(x, scaling)
    return scalar_to_array
end

function make_scalar_to_array!(scaling::AbstractArray)
    scalar_to_array!(y::AbstractArray, x::Number) = _scalar_to_array!(y, x, scaling)
    return scalar_to_array!
end

scalar_to_array_derivative(x, scaling) = scaling .* cos.(scaling .* x)
scalar_to_array_second_derivative(x, scaling) = -(scaling .^ 2) .* sin.(scaling .* x)
scalar_to_array_pushforward(x, dx, scaling) = scalar_to_array_derivative(x, scaling) .* dx
scalar_to_array_pullback(x, dy, scaling) = dot(scalar_to_array_derivative(x, scaling), dy)

function scalar_to_array_ref(scaling)
    return Reference(;
        pushforward=(args...) -> scalar_to_array_pushforward(args..., scaling),
        pullback=(args...) -> scalar_to_array_pullback(args..., scaling),
        derivative=(args...) -> scalar_to_array_derivative(args..., scaling),
        second_derivative=(args...) -> scalar_to_array_second_derivative(args..., scaling),
    )
end

## Array to scalar

array_to_scalar(x::AbstractArray)::Number = sum(sin, x)

array_to_scalar_gradient(x) = cos.(x)
array_to_scalar_hvp(x, v) = -sin.(x) .* v
array_to_scalar_pushforward(x, dx) = dot(array_to_scalar_gradient(x), dx)
array_to_scalar_pullback(x, dy) = array_to_scalar_gradient(x) .* dy
array_to_scalar_hessian(x) = Diagonal(-sin.(vec(x)))

function array_to_scalar_ref()
    return Reference(;
        pushforward=array_to_scalar_pushforward,
        pullback=array_to_scalar_pullback,
        gradient=array_to_scalar_gradient,
        hvp=array_to_scalar_hvp,
        hessian=array_to_scalar_hessian,
    )
end

## Array to array

vector_to_vector(x::AbstractVector)::AbstractVector = vcat(sin.(x), cos.(x))

function vector_to_vector!(y::AbstractVector, x::AbstractVector)
    y[1:length(x)] .= sin.(x)
    y[(length(x) + 1):(2length(x))] .= cos.(x)
    return nothing
end

vector_to_vector_pushforward(x, dx) = vcat(cos.(x) .* dx, -sin.(x) .* dx)
vector_to_vector_pullback(x, dy) = cos.(x) .* first_half(dy) .- sin.(x) .* second_half(dy)
vector_to_vector_jacobian(x) = vcat(Diagonal(cos.(x)), Diagonal(-sin.(x)))

function vector_to_vector_ref()
    return Reference(;
        pushforward=vector_to_vector_pushforward,
        pullback=vector_to_vector_pullback,
        jacobian=vector_to_vector_jacobian,
    )
end

vector_to_matrix(x::AbstractVector)::AbstractMatrix = hcat(sin.(x), cos.(x))

function vector_to_matrix!(y::AbstractMatrix, x::AbstractVector)
    y[:, 1] .= sin.(x)
    y[:, 2] .= cos.(x)
    return nothing
end

vector_to_matrix_pushforward(x, dx) = hcat(cos.(x) .* dx, -sin.(x) .* dx)
vector_to_matrix_pullback(x, dy) = cos.(x) .* dy[:, 1] .- sin.(x) .* dy[:, 2]
vector_to_matrix_jacobian(x) = vcat(Diagonal(cos.(x)), Diagonal(-sin.(x)))

function vector_to_matrix_ref()
    return Reference(;
        pushforward=vector_to_matrix_pushforward,
        pullback=vector_to_matrix_pullback,
        jacobian=vector_to_matrix_jacobian,
    )
end

matrix_to_vector(x::AbstractMatrix)::AbstractVector = vcat(vec(sin.(x)), vec(cos.(x)))

function matrix_to_vector!(y::AbstractVector, x::AbstractMatrix)
    n = length(x)
    y[1:n] .= sin.(getindex.(Ref(x), 1:n))
    y[(n + 1):(2n)] .= cos.(getindex.(Ref(x), 1:n))
    return nothing
end

matrix_to_vector_pushforward(x, dx) = vcat(vec(cos.(x) .* dx), vec(-sin.(x) .* dx))
function matrix_to_vector_pullback(x, dy)
    return cos.(x) .* reshape(first_half(dy), size(x)) .-
           sin.(x) .* reshape(second_half(dy), size(x))
end
matrix_to_vector_jacobian(x) = vcat(Diagonal(vec(cos.(x))), Diagonal(vec(-sin.(x))))

function matrix_to_vector_ref()
    return Reference(;
        pushforward=matrix_to_vector_pushforward,
        pullback=matrix_to_vector_pullback,
        jacobian=matrix_to_vector_jacobian,
    )
end

matrix_to_matrix(x::AbstractMatrix)::AbstractMatrix = hcat(vec(sin.(x)), vec(cos.(x)))

function matrix_to_matrix!(y::AbstractMatrix, x::AbstractMatrix)
    n = length(x)
    y[:, 1] .= sin.(getindex.(Ref(x), 1:n))
    y[:, 2] .= cos.(getindex.(Ref(x), 1:n))
    return nothing
end

matrix_to_matrix_pushforward(x, dx) = hcat(vec(cos.(x) .* dx), vec(-sin.(x) .* dx))
function matrix_to_matrix_pullback(x, dy)
    return cos.(x) .* reshape(dy[:, 1], size(x)) .- sin.(x) .* reshape(dy[:, 2], size(x))
end
matrix_to_matrix_jacobian(x) = vcat(Diagonal(vec(cos.(x))), Diagonal(vec(-sin.(x))))

function matrix_to_matrix_ref()
    return Reference(;
        pushforward=matrix_to_matrix_pushforward,
        pullback=matrix_to_matrix_pullback,
        jacobian=matrix_to_matrix_jacobian,
    )
end

## Gather

const SCALING_VEC = Vector(1:12)
const SCALING_MAT = Matrix((1:3) .* transpose(1:4))

function default_scenarios_allocating()
    return [
        Scenario(scalar_to_scalar; x=2.0, ref=scalar_to_scalar_ref()),
        Scenario(
            make_scalar_to_array(SCALING_VEC); x=2.0, ref=scalar_to_array_ref(SCALING_VEC)
        ),
        Scenario(
            make_scalar_to_array(SCALING_MAT); x=2.0, ref=scalar_to_array_ref(SCALING_MAT)
        ),
        Scenario(array_to_scalar; x=float.(1:12), ref=array_to_scalar_ref()),
        Scenario(array_to_scalar; x=float.(reshape(1:12, 3, 4)), ref=array_to_scalar_ref()),
        Scenario(vector_to_vector; x=float.(1:12), ref=vector_to_vector_ref()),
        Scenario(vector_to_matrix; x=float.(1:12), ref=vector_to_matrix_ref()),
        Scenario(
            matrix_to_vector; x=float.(reshape(1:12, 3, 4)), ref=matrix_to_vector_ref()
        ),
        Scenario(
            matrix_to_matrix; x=float.(reshape(1:12, 3, 4)), ref=matrix_to_matrix_ref()
        ),
    ]
end

function default_scenarios_mutating()
    return [
        Scenario(
            make_scalar_to_array!(SCALING_VEC);
            x=2.0,
            y=zeros(12),
            ref=scalar_to_array_ref(SCALING_VEC),
        ),
        Scenario(
            make_scalar_to_array!(SCALING_MAT);
            x=2.0,
            y=zeros(3, 4),
            ref=scalar_to_array_ref(SCALING_MAT),
        ),
        Scenario(
            vector_to_vector!; x=float.(1:12), y=zeros(24), ref=vector_to_vector_ref()
        ),
        Scenario(
            vector_to_matrix!; x=float.(1:12), y=zeros(12, 2), ref=vector_to_matrix_ref()
        ),
        Scenario(
            matrix_to_vector!;
            x=float.(reshape(1:12, 3, 4)),
            y=zeros(24),
            ref=matrix_to_vector_ref(),
        ),
        Scenario(
            matrix_to_matrix!;
            x=float.(reshape(1:12, 3, 4)),
            y=zeros(12, 2),
            ref=matrix_to_matrix_ref(),
        ),
    ]
end

"""
    default_scenarios()

Create a vector of [`Scenario`](@ref)s for testing differentiation. 
"""
function default_scenarios()
    return vcat(
        default_scenarios_allocating(),  #
        default_scenarios_mutating(),  #
    )
end
