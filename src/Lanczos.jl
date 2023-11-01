using LinearAlgebra
# import LinearAlgebra.LAPACK: stegr!
# import LinearAlgebra: checksquare
# import LinearAlgebra:
#     BlasFloat

# import LinearAlgebra.BLAS: @blasfunc, BlasReal, BlasComplex







# Some modified wrappers for Lapack
# import LinearAlgebra:
#     BlasFloat,
#     BlasInt,
#     LAPACKException,
#     DimensionMismatch,
#     SingularException,
#     PosDefException,
#     chkstride1,
#     checksquare
# import LinearAlgebra.BLAS: @blasfunc, BlasReal, BlasComplex
# import LinearAlgebra.LAPACK: chklapackerror
# @static if VERSION >= v"1.7"
#     const liblapack = LinearAlgebra.BLAS.libblastrampoline
# else
#     const liblapack = LinearAlgebra.LAPACK.liblapack
# end

# @static if isdefined(Base, :require_one_based_indexing)
#     import Base: require_one_based_indexing
# else
#     require_one_based_indexing(A...) =
#         !Base.has_offset_axes(A...) || throw(
#             ArgumentError(
#                 "offset arrays are not supported but got an array with index other than 1"
#             )
#         )
# end


include("Calculator.jl")


function A(z, xᵢ, fᵢ)
    del_lanczos = 0.01

    xₜ₊₁ = xᵢ + reshape(z, (3, 215)) * del_lanczos
    e, fₜ₊₁ = calcforces(xₜ₊₁)
    return vec(-(fₜ₊₁ - fᵢ)) / del_lanczos
end

function center(x, mask)
    n = size(x, 1)
    n_f = sum(mask)

    com = sum(x[i, :] for i in 1:n if mask[i]) / n_f

    for i in 1:n
        mask[i] ? (x[i, :] .-= com) : nothing
    end
end

function lanczos(xᵢ::Matrix{Float64}, fᵢ::Matrix{Float64}, q::Matrix{Float64}, isnew_q::Bool, mask=ones(Float64, (3, 215)))
    N = 215
    mask = Matrix{Bool}(mask .> 0)

    Nₗ = 200 # maxvec
    # del_lanczos = 0.01

    # eigvec_conv_thr = 0.0001
    # eigval_conv_thr = 0.01

    V = zeros(Float64, (3 * N, Nₗ))
    H = zeros(Float64, (Nₗ, Nₗ))

    zₜ₋₁ = @view V[:, 1]

    if isnew_q
        zₜ₋₁[:] = 0.5 .- rand(Float64, (3 * N))
        # center(zₜ₋₁, mask)
        qₜ₋₁ = zeros(Float64, (3 * N))
    else
        zₜ₋₁[:] = q
        qₜ₋₁ = q
    end
    zₜ₋₁[:] = normalize(zₜ₋₁)

    q = A(zₜ₋₁, xᵢ, fᵢ)
    α = zₜ₋₁ ⋅ q

    zₜ = @view V[:, 2]
    zₜ[:] = q - α * zₜ₋₁
    β = norm(zₜ)
    zₜ[:] = normalize(zₜ)

    H[1, 1] = α
    H[1, 2], H[2, 1] = β, β

    νₘᵢₙ = νₜ₋₁ = zₜ₋₁
    λₘᵢₙ = λₜ₋₁ = α
    # println("Lanczos step ", 1)
    # println("λₜ₋₁ = ", λₜ₋₁, "\n")

    K = 1

    for i in 2:Nₗ
        K += 1

        zₜ₋₁ = @view V[:, i-1]
        zₜ = @view V[:, i]

        q = A(zₜ, xᵢ, fᵢ)
        α = zₜ ⋅ q
        H[i, i] = α

        if i < Nₗ
            zₜ₊₁ = @view V[:, i+1]
            zₜ₊₁[:] = q - α * zₜ - β * zₜ₋₁
            for k in 1:i-1
                # println("Orthogonalization step ", k)
                zₖ = @view V[:, k]
                zₜ₊₁[:] = zₜ₊₁ - (zₜ₊₁ ⋅ zₖ) * zₖ
            end
            zₜ₊₁[:] = normalize(zₜ₊₁)
            β = zₜ₊₁ ⋅ q
            β /= norm(zₜ₊₁)

            H[i+1, i], H[i, i+1] = β, β

            # println("                                          α, β = $β, $α")
        end

        # println("applications of the linear map ", K)

        F = eigen(@view H[1:i, 1:i])

        iₘᵢₙ = argmin(F.values)
        λₘᵢₙ = F.values[iₘᵢₙ]
        νₘᵢₙ = reshape(F.vectors[:, iₘᵢₙ], (i, 1))

        Vᵢ = @view V[:, 1:i]
        # q = mapslices(vᵢ -> vᵢ ⋅ νₘᵢₙ, Vᵢ, dims=3)[:, :, 1]
        q = Vᵢ * νₘᵢₙ
        q = normalize(q)

        δ = abs((λₘᵢₙ - λₜ₋₁) / λₜ₋₁)

        λₜ₋₁ = λₘᵢₙ
        νₜ₋₁ = νₘᵢₙ



        # println("λₜ₋₁ = ", λₜ₋₁, "\n")
        # println("                                          δ = ", δ)


        if δ < 0.001
            break
        end

    end

    println("applications of the linear map ", K)

    println("λₘᵢₙ = ", λₘᵢₙ)

    qₜ₋₁ ⋅ q < 0 ? q *= -1 : nothing
    q = normalize(q)

    return q

end















abstract type Orthogonalizer end
struct ModifiedGramSchmidt <: Orthogonalizer end
struct ModifiedGramSchmidtIR <: Orthogonalizer end

function orthogonalize!(v::T, b::Vector{T}, ::ModifiedGramSchmidt) where {T}
    α = 0.0
    for (i, q) in enumerate(b)
        α = dot(q, v)
        v = axpy!(-α, q, v)
    end
    nnew = norm(v)
    return (v, α, nnew)
end
reorthogonalize!(v::T, b::Vector{T}, orthogonalizer::ModifiedGramSchmidt) where {T} = orthogonalize!(v, b, orthogonalizer)

function orthogonalize!(v::T, b::Vector{T}, ::ModifiedGramSchmidtIR) where {T}
    v, α, β = orthogonalize!(v, b, ModifiedGramSchmidt())
    β₋₁ = 2 * abs(β) + abs(α)
    while eps(β) < β < 1 / sqrt(2) * β₋₁
        β₋₁ = β
        v, dα, β = reorthogonalize!(v, b, ModifiedGramSchmidt())
        α += dα
    end
    return (v, α, β)
end

struct Lanczos{O<:Orthogonalizer}
    orthogonalizer::O
    krylovdim::Int
    maxiter::Int
    ϵ::Float64
    eager::Bool
    verbosity::Int
end
Lanczos(; krylovdim::Int=30, maxiter::Int=100, ϵ::Float64=1e-12, orthogonalizer::Orthogonalizer=ModifiedGramSchmidtIR(), eager::Bool=false, verbosity::Int=0) = Lanczos(orthogonalizer, krylovdim, maxiter, ϵ, eager, verbosity)

struct LanczosIterator{F,T,O<:Orthogonalizer}
    operator::F
    x₀::T
    orthogonalizer::O
    function LanczosIterator{F,T,O}(operator::F, x₀::T, orthogonalizer::O) where {F,T,O<:Orthogonalizer}
        return new{F,T,O}(operator, x₀, orthogonalizer)
    end
end
LanczosIterator(operator::F, x₀::T, orthogonalizer::O=ModifiedGramSchmidtIR()) where {F,T,O<:Orthogonalizer} = LanczosIterator{F,T,O}(operator, x₀, orthogonalizer)

mutable struct LanczosState{T}
    k::Int # current Krylov dimension
    V::Vector{T} # basis of length k
    αs::Vector{Float64}
    βs::Vector{Float64}
    r::T
end

@inbounds normresidual(F::LanczosState) = F.βs[F.k]
rayleighquotient(F::LanczosState) = SymTridiagonal(F.αs, F.βs)
basis(F::LanczosState) = F.V
residual(F::LanczosState) = F.r

function expand!(iter::LanczosIterator, state::LanczosState; verbosity::Int=0)
    βold = normresidual(state)
    V = state.V
    r = state.r
    V = push!(V, r / βold)
    r, α, β = lanczosrecurrence(iter.operator, V, βold, iter.orthogonalizer)

    αs = push!(state.αs, α)
    βs = push!(state.βs, β)

    state.k += 1
    state.r = r
    # if verbosity > 0
    #     @info "Lanczos iteration step $(state.k): normres = $β"
    # end
    return state
end

function shrink!(state::LanczosState, k)
    state.k == length(state.V) || error("we cannot shrink LanczosState without keeping Lanczos vectors")
    state.k <= k && return state
    V = state.V
    while length(V) > k + 1
        pop!(V)
    end
    r = pop!(V)
    resize!(state.αs, k)
    resize!(state.βs, k)
    state.k = k
    state.r = rmul!(r, normresidual(state))
    return state
end

function initialize(iter::LanczosIterator; verbosity::Int=0)
    x₀ = iter.x₀
    β₀ = norm(x₀)
    iszero(β₀) && throw(ArgumentError("initial vector should not have norm zero"))
    w₀ = iter.operator(x₀)
    α = dot(x₀, w₀) / (β₀ * β₀)
    v = x₀ / β₀

    r = w₀ / β₀
    r = axpy!(-α, v, r)

    # possibly reorthogonalize,
    if iter.orthogonalizer isa ModifiedGramSchmidtIR
        r, dα, β = orthogonalize!(r, [v], iter.orthogonalizer)
        α += dα
    end
    V, αs, βs = Vector([v]), [real(α)], [β]
    # if verbosity > 0
    #     @info "Lanczos iteration step 1: normres = $β"
    # end
    return LanczosState(1, V, αs, βs, r)
end

function restarted_initialize(state::LanczosState, x₀s, operator, orthogonalizer; verbosity::Int=0)
    local new_state
    local V, αs, βs, r = Vector([]), Float64[], Float64[], Matrix{Float64}
    local k = 0
    for i in eachindex(x₀s)
        k += 1
        x₀ = x₀s[i]
        β₀ = norm(x₀)
        iszero(β₀) && throw(ArgumentError("initial vector should not have norm zero"))
        w₀ = operator(x₀)
        α = dot(x₀, w₀) / (β₀ * β₀)
        v = x₀ / β₀
        V = push!(V, v)

        r = w₀ / β₀
        r = axpy!(-α, v, r)

        # possibly reorthogonalize,
        # if iter.orthogonalizer isa ModifiedGramSchmidtIR
        r, dα, β = orthogonalize!(r, V, orthogonalizer)
        α += dα
        # end
        αs, βs = push!(αs, α), push!(βs, β)

    end
    new_state = LanczosState(k, V, αs, βs, r)
    # V, αs, βs = Vector([v]), [real(α)], [β]
    # if verbosity > 0
    #     @info "Lanczos iteration step 1: normres = $β"
    # end
    iter = LanczosIterator(operator, r, orthogonalizer)
    return new_state, iter
end


function lanczosrecurrence(operator, V::Vector, β, orthogonalizer::Orthogonalizer)
    v = V[end]
    w = operator(v)
    w = axpy!(-β, V[end-1], w)

    w, α, β = orthogonalize!(w, V, orthogonalizer)
    return w, α, β
end

# Householder reflector that zeros the elements A[row,r] (except for A[row,k]) upon rmulc!(A,h)
function householder!(H::AbstractMatrix, U::AbstractMatrix, row::Int, r::AbstractRange{Int}, k=first(r))
    i = findfirst(isequal(k), r)
    i isa Nothing && error("k = $k should be in the range r = $r")

    vᵣ = H[row, r]

    σ = sum(vᵣ[[1:i-1; i+1:end]] .^ 2)
    ν = sqrt(vᵣ[i]^2 + σ)

    if σ != zero(σ) && vᵣ[i] != ν
        vᵢ = vᵣ[i] < 0 ? vᵣ[i] - ν : -σ / (vᵣ[i] + ν)
        vᵣ[[1:i-1; i+1:end]] /= vᵢ
        vᵣ[i] = 1
        β = -vᵢ / ν
    else
        β = zero(σ)
    end

    H[i+1, i] = ν
    H[i+1, 1:i-1] .= 0.0
    H[r, axes(H, 2)] -= vᵣ * (vᵣ' * H[r, axes(H, 2)]) * β
    Hᵢ = @view H[1:i, :]
    Hᵢ[axes(Hᵢ, 1), r] -= (Hᵢ[axes(Hᵢ, 1), r] * vᵣ) * (β * vᵣ')
    U[axes(U, 1), r] -= (U[axes(U, 1), r] * vᵣ) * (β * vᵣ')
end

function basistransform!(v::Vector{Matrix{T}}, U::AbstractMatrix) where {T} # U should be unitary or isometric
    v[1:size(U)[2]] = sum(U .* v, dims=1)
    return v
end

function artn_lanczos(operator, x₀, nsv::Int, lkargs::Lanczos; state=nothing)
    krylovdim = lkargs.krylovdim
    maxiter = lkargs.maxiter
    ϵ = lkargs.ϵ
    nsv > krylovdim && error("krylov dimension $(krylovdim) too small to compute $nsv eigenvalues")

    ## FIRST ITERATION: setting up

    # Initialize Lanczos factorization

    if state === nothing
        iter = LanczosIterator(operator, x₀, lkargs.orthogonalizer)
        state = initialize(iter; verbosity=lkargs.verbosity - 2)
    else
        # for i in 1:nsv
        # shrink!(state, state.k-nsv*2)
        # end
        state, iter = restarted_initialize(state, x₀, operator, lkargs.orthogonalizer; verbosity=lkargs.verbosity - 2)

        # state.k = nsv

        # initialize(iter; verbosity=lkargs.verbosity - 2)
        # state = expand!(iter, state; verbosity=lkargs.verbosity - 2)
        # state = expand!(iter, state; verbosity=lkargs.verbosity - 2)
    end
    β = normresidual(state)

    # allocate storage
    HH = zeros(Float64, (krylovdim + 1, krylovdim))
    UU = zeros(Float64, (krylovdim, krylovdim))

    numops, numiter = 1, 1
    nconvsv = 0
    local D, U, f
    while true
        β = normresidual(state)
        k = state.k

        if k == krylovdim || β <= ϵ || (lkargs.eager && k >= nsv)
            U = copyto!(view(UU, 1:k, 1:k), I)
            f = view(HH, k + 1, 1:k)
            T = rayleighquotient(state) # symtridiagonal

            # compute eigenvalues
            if k == 1
                D = [T[1, 1]]
                f[1] = β
                nconvsv = Int(β <= ϵ)
            else
                if k < krylovdim
                    T = deepcopy(T)
                end
                D, U = eigen(T; sortby=real)
                f = view(U, k, :) * β
                nconvsv = 0
                while nconvsv < k && abs(f[nconvsv+1]) <= ϵ
                    nconvsv += 1
                end
            end

            if nconvsv >= nsv
                break
            end
        end

        if k < krylovdim # expand Krylov factorization
            state = expand!(iter, state; verbosity=lkargs.verbosity - 2)
            numops += 1
        else ## shrink and restart
            if numiter == maxiter
                break
            end

            # Determine how many to keep
            keep = div(3 * krylovdim + 2 * nconvsv, 5) # strictly smaller than krylovdim since nconvsv < nsv <= krylovdim, at least equal to nconvsv

            # Restore Lanczos form in the first keep columns
            H = fill!(view(HH, 1:keep+1, 1:keep), zero(eltype(HH)))
            @inbounds for j in 1:keep
                H[j, j], H[keep+1, j] = D[j], f[j]
            end
            @inbounds for j in keep:-1:1
                householder!(H, U, j + 1, 1:j, j)
            end
            @inbounds for j in 1:keep
                state.αs[j], state.βs[j] = H[j, j], H[j+1, j]
            end

            # Update B by applying U using Householder reflections
            B = basis(state)
            basistransform!(B, view(U, :, 1:keep))
            r = residual(state)
            B[keep+1] = r / β

            # Shrink Lanczos factorization
            state = shrink!(state, keep)
            numiter += 1
        end
    end

    if nconvsv > nsv
        nsv = nconvsv
    end
    values = D[1:nsv]

    # Compute eigenvectors
    V = view(U, :, 1:nsv)

    # Compute convergence information
    vectors = let B = basis(state)
        [v' * B for v in eachcol(V)]
    end

    residuals = let r = residual(state)
        [last(v) * r for v in eachcol(V)]
    end
    normresiduals = let f = f
        map(i -> abs(f[i]), 1:nsv)
    end

    println(numops, " applications of the linear map")

    return values, vectors, state
end
