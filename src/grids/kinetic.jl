"""
    KineticGrid{T}

Phase-space grid `(p, ξ, r)` with a momentum grid shared by all radii, and the
phase-space Jacobian `𝒱'` (DREAM `Vp`) at cell centres and at the faces in each
direction. The distribution function is stored as `f[i, j, ir]` with `p` fastest.

In cylindrical geometry `𝒱' = 2π p² V'(r)`, so that the density is
`n(r) = (1/V') Σ_ij 𝒱'_ij Δp_i Δξ_j f_ij`.
"""
struct KineticGrid{T<:AbstractFloat}
    mom::PXiGrid{T}
    rad::RadialGrid{T}
    Vp::Array{T,3}      # (np, nxi, nr)
    Vp_fp::Array{T,3}   # (np + 1, nxi, nr), on p faces
    Vp_fxi::Array{T,3}  # (np, nxi + 1, nr), on ξ faces
    Vp_fr::Array{T,3}   # (np, nxi, nr + 1), on r faces
end

"""
    cylindrical_grid(mom, rad)

Kinetic grid with a homogeneous magnetic field (no trapping, no bounce averaging).
"""
function cylindrical_grid(mom::PXiGrid{T}, rad::RadialGrid{T}) where {T}
    n1, n2, n3 = np(mom), nxi(mom), nr(rad)
    Vp = [2π * mom.p[i]^2 * rad.VpVol[ir] for i in 1:n1, _ in 1:n2, ir in 1:n3]
    Vp_fp = [2π * mom.p_f[i]^2 * rad.VpVol[ir] for i in 1:n1+1, _ in 1:n2, ir in 1:n3]
    Vp_fxi = [2π * mom.p[i]^2 * rad.VpVol[ir] for i in 1:n1, _ in 1:n2+1, ir in 1:n3]
    Vp_fr = [2π * mom.p[i]^2 * rad.VpVol_f[ir] for i in 1:n1, _ in 1:n2, ir in 1:n3+1]
    return KineticGrid{T}(mom, rad, Vp, Vp_fp, Vp_fxi, Vp_fr)
end

np(g::KineticGrid) = np(g.mom)
nxi(g::KineticGrid) = nxi(g.mom)
nr(g::KineticGrid) = nr(g.rad)
Base.size(g::KineticGrid) = (np(g), nxi(g), nr(g))
ncells(g::KineticGrid) = prod(size(g))

"""
    density(g, f)

Density moment `n(r) = (1/V') ∫ 𝒱' f dp dξ` at each radius.
"""
function density(g::KineticGrid{T}, f::AbstractArray{S,3}) where {T,S}
    n = zeros(promote_type(T, S), nr(g))
    for ir in 1:nr(g), j in 1:nxi(g), i in 1:np(g)
        n[ir] += g.Vp[i, j, ir] * g.mom.dp[i] * g.mom.dxi[j] * f[i, j, ir]
    end
    n ./= g.rad.VpVol
    return n
end

"""
    total_particles(g, f)

Total particle content `Σ 𝒱' Δp Δξ Δr f`, conserved by a flux-form discretization.
"""
function total_particles(g::KineticGrid{T}, f::AbstractArray{S,3}) where {T,S}
    N = zero(promote_type(T, S))
    for ir in 1:nr(g), j in 1:nxi(g), i in 1:np(g)
        N += g.Vp[i, j, ir] * g.mom.dp[i] * g.mom.dxi[j] * g.rad.dr[ir] * f[i, j, ir]
    end
    return N
end
