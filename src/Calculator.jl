using JuLIP
using StaticArrays


function calcforces(x::Matrix{Float64})
    cell::Matrix = [16.291494 0 0; 0 16.291494 0; 0 0 16.291494]
    at = Atoms(:Si, x)
    set_cell!(at, cell)
    set_pbc!(at, true)
    
    set_calculator!(at, StillingerWeber(ϵ=2.1683, λ=21.0))

    e = energy(at)
    f = reduce(hcat, forces(at))

    return e, f
end




