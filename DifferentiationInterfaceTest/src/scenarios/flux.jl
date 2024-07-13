using Flux
using LinearAlgebra

NNinput = collect(range(0.01, 0.10, 10))
model1layer = Dense(10 => 1,sigmoid)

function arr_to_num(x)
  Flux.trainables(model1layer)[1][:] = x
  return sum(model1layer(NNinput)) # to reduce 1 element array to float
end

function arr_to_num_gradient(x)
  fx = arr_to_num(x)
  println("fx:", fx)
  return fx * (1.0-fx) * NNinput
end

function arr_to_num_scenarios_onearg(
  x::AbstractArray;
)
  y =    arr_to_num(x)
  grad = arr_to_num_gradient(x)

  # pushforward stays out of place
  scens = [
              GradientScenario(f; x, y, grad, nb_args=1, place=:outofplace),
          ]
  return scens
end


"""
    static_scenarios(rng=Random.default_rng())

Create a vector of [`Scenario`](@ref)s with static array types from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
"""
function static_scenarios(rng::AbstractRNG=default_rng())
    x_ = rand(rng, 10)
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

    SV_6 = SVector{6}
    MV_6 = MVector{6}
    SV_12 = SVector{12}
    MV_12 = MVector{12}

    SM_2_3 = SMatrix{2,3}
    MM_2_3 = MMatrix{2,3}
    SM_6_2 = SMatrix{6,2}
    MM_6_2 = MMatrix{6,2}

    scens = vcat(
        # one argument
        num_to_arr_scenarios_onearg(x_, SV_6; dx=dx_, dy=SV_6(dy_6)),
        num_to_arr_scenarios_onearg(x_, SM_2_3; dx=dx_, dy=SM_2_3(dy_2_3)),
        arr_to_num_scenarios_onearg(SV_6(x_6); dx=SV_6(dx_6), dy=dy_, linalg),
        arr_to_num_scenarios_onearg(SM_2_3(x_2_3); dx=SM_2_3(dx_2_3), dy=dy_, linalg),
        vec_to_vec_scenarios_onearg(SV_6(x_6); dx=SV_6(dx_6), dy=SV_12(dy_12)),
        vec_to_mat_scenarios_onearg(SV_6(x_6); dx=SV_6(dx_6), dy=SM_6_2(dy_6_2)),
        mat_to_vec_scenarios_onearg(SM_2_3(x_2_3); dx=SM_2_3(dx_2_3), dy=SV_12(dy_12)),
        mat_to_mat_scenarios_onearg(SM_2_3(x_2_3); dx=SM_2_3(dx_2_3), dy=SM_6_2(dy_6_2)),
        # two arguments
        num_to_arr_scenarios_twoarg(x_, MV_6; dx=dx_, dy=MV_6(dy_6)),
        num_to_arr_scenarios_twoarg(x_, MM_2_3; dx=dx_, dy=MM_2_3(dy_2_3)),
        vec_to_vec_scenarios_twoarg(MV_6(x_6); dx=MV_6(dx_6), dy=MV_12(dy_12)),
        vec_to_mat_scenarios_twoarg(MV_6(x_6); dx=MV_6(dx_6), dy=MM_6_2(dy_6_2)),
        mat_to_vec_scenarios_twoarg(MM_2_3(x_2_3); dx=MM_2_3(dx_2_3), dy=MV_12(dy_12)),
        mat_to_mat_scenarios_twoarg(MM_2_3(x_2_3); dx=MM_2_3(dx_2_3), dy=MM_6_2(dy_6_2)),
    )
    scens = filter(scens) do s
        place(s) == :outofplace || s.x isa Union{Number,MArray}
    end
    return scens
end
