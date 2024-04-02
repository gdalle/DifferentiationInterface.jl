include("test_imports.jl")

b = AutoTapir()
n = 2

function f!(y, x)
    y .= x .^ 2
    return nothing
end 

value_and_pullback!!(f!, zeros(n), zeros(n), b, float.(1:n), float.(n+1:2n))
