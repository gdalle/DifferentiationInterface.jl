"""
    make_closure(scen::Scenario)

Return a new [`Scenario`](@ref) with a modified function `f` or `f!` that closes over differentiable data.
"""
function make_closure(scen::Scenario)
    closed_data = Ref(zero(scen.y))
    if nb_args(scen) == 1
        function closure_f(x)
            closed_data[] = scen.f(x)
            return copy(closed_data[])
        end
        return change_function(scen, closure_f)
    elseif nb_args(scen) == 2
        function closure_f!(y, x)
            scen.f(closed_data[], x)
            copyto!(y, closed_data[])
            return nothing
        end
        return change_function(scen, closure_f!)
    end
end
