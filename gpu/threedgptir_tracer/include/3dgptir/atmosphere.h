/* Atmosphere volume (voxel cloud + 1D gas). Copied-PTIR extension.
 * Cloud optical truth is voxels only. Gaussians are surfaces, never k_true. */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

enum AtmSpectralMode {
    ATM_SPECTRAL_SW = 0,
    ATM_SPECTRAL_LW = 1,
    ATM_SPECTRAL_SW_CIE_XYZ = 2 /* htrdr default image: CIE 1931 380-780 nm */
};

/* Scene mesh materials. Kernel does not know "sand"; slot 0 is the LUT the host bound. */
enum AtmMeshMaterial {
    ATM_MTL_SPECTRAL_LAMBERT = 0, /* brdf_lambda_nm / brdf_R */
    ATM_MTL_RGB_LAMBERT = 1,      /* mesh.albedo[tri] */
    ATM_MTL_SPECTRAL_SPECULAR = 2 /* htrdr ssf_specular: Dirac + ρ(λ) from the same LUT */
};

enum AtmEvent {
    ATM_EVENT_NONE = 0,
    ATM_EVENT_ABSORB_VOXEL,
    ATM_EVENT_SCATTER_VOXEL,
    ATM_EVENT_ABSORB_GAS,
    ATM_EVENT_SCATTER_GAS,
    ATM_EVENT_HIT_SURFACE,
    ATM_EVENT_ESCAPE
};

/* Regular voxel cloud: ka/ks at cell centers, kext_max per brick. */
struct AtmVoxelGrid {
    int nx, ny, nz;
    int brick; /* cells per brick edge, typically 8 */
    int nbx, nby, nbz;
    float origin[3];
    float dx, dy, dz;
    const float* ka;           /* [nz * ny * nx] */
    const float* ks;           /* [nz * ny * nx] */
    const float* temperature;  /* [nz * ny * nx], Kelvin */
    const float* kext_max;     /* [nbz * nby * nbx], SW: max(ks) per brick */
};

/* Piecewise-constant gas layers along +Z. z_iface has nlayers+1 entries. */
struct AtmGasLayers {
    int nlayers;
    const float* z_iface; /* [nlayers+1], increasing */
    const float* ka;
    const float* ks;
    const float* temperature;
};

struct AtmMesh {
    int nvert;
    int ntri;
    const float* vertices; /* [nvert * 3] */
    const int* indices;    /* [ntri * 3] */
    const float* albedo;   /* [ntri * 3], optional */
    const float* temperature; /* [ntri], optional, Kelvin */
};

struct AtmSun {
    float dir[3]; /* toward the sun, normalized */
    float radiance;
    float cos_half_angle; /* cone; 1 = directional */
};

/* VOLUME GS on the same lattice as AtmVoxelGrid (append-only consumer). */
struct AtmVolumeGs {
    int enabled;
    int n;
    int rx, ry, rz;
    const int* pid;
    const float* center;
    const float* scale;
    const float* quat;
    const float* ka;
    const float* ks;
    const float* kext_max;
    float majorant_boost;
};

struct AtmMeshHit {
    float t;
    int tri;
    float n[3];
    float albedo[3];
    float temperature;
};

struct AtmParams {
    int enabled;     /* 0 = original PTIR raygen (no volume) */
    int gs_enabled;  /* 0 = skip Gaussian surfaces */
    int spectral_mode; /* AtmSpectralMode */
    float wavelength_m;
    float mie_omega; /* droplet ω; 0 => use voxel ks/(ka+ks) */
    float mie_g;     /* HG asymmetry */
    float path_tmax;
    int max_bounces;
    struct AtmVoxelGrid voxels;
    struct AtmGasLayers gas;
    struct AtmMesh mesh;
    struct AtmSun sun;
    /* Appended only: do not insert above (layout of origin/ka/ks/mesh). */
    int cie_enabled; /* 1 = sample CIE X,Y,Z like htrdr draw_pixel_image */
    int cie_nbins;
    float cie_lambda0_nm;
    float cie_dlambda_nm;
    float cie_rcp_integral[3]; /* 1/∫ CMF dλ, nm^-1; pdf = this (htrdr) */
    const float* cie_cdf;      /* [3 * cie_nbins] X then Y then Z */
    int sw_nbands;
    const float* sw_lambda_lo_nm; /* [sw_nbands] */
    const float* sw_lambda_hi_nm; /* [sw_nbands] */
    const float* sw_gas_ka;       /* [sw_nbands * sw_nquad_max * nlayers] */
    const float* sw_gas_ks;
    /* Spectral Lambert LUT from the scene (mountain.sh binds sand.mrumtl). */
    int brdf_n;
    const float* brdf_lambda_nm;
    const float* brdf_R;
    float path_albedo; /* this path's R(λ) from the LUT; <0 = no LUT */
    /* HTGOP quadrature (htrdr iquad) + htsky Mie band averages. Append only. */
    int sw_nquad_max;
    const int* sw_nquad;         /* [sw_nbands] */
    const float* sw_quad_cdf;    /* [sw_nbands * sw_nquad_max] */
    const float* mie_cs_band;    /* [sw_nbands] band-average Cs */
    const float* mie_ca_band;
    const float* mie_g_nm;       /* g(λ) LUT, htrdr per-wavelength HG */
    int mie_g_n;
    float mie_g_lambda0_nm;
    float mie_g_dlambda_nm;
    float mie_cs_ref; /* Cs at voxel dump wavelength (550 nm) */
    float mie_ca_ref;
    float path_ks_scale; /* Cs_band / Cs_ref; 1 if unset */
    float path_ka_scale;
    /* Native HTCP Z faces [nz+1]. Null = uniform g->dz. Do not put this on
     * AtmVoxelGrid (would shift gas/mesh/sun). */
    const float* voxel_z_iface;
    /* 1 = one CIE channel per launch index (depth = 3*spp). Default 0 = one λ → XYZ. */
    int cie_split_launch;
    /* Original film spp (before CIE-split repeat). Compact launches have z=1. */
    int cie_spp;
    /* Per-triangle AtmMeshMaterial. Null / empty = all SPECTRAL_LAMBERT. Append only. */
    const int* mesh_mat_id;
    /* Ground searchlight: Lambert disk in a cone. Le = sun.radiance * (L_lamp / L_sun).
     * L_lamp from real xenon lumens, not a small sun. Append only. */
    int spot_enabled;
    float spot_pos[3];
    float spot_dir[3]; /* disk normal / beam axis, toward the target */
    float spot_sun_scale;
    float spot_cos_half;
    float spot_radius;
    /* HTGOP LW (RRTMG, ka only). Append only.
     * SWIR is SW RTE. MWIR: host sun.radiance>0 → Planck sun + thermal emit.
     * LWIR: host sun.radiance=0 → thermal only.
     * MWIR dump λ = 4000 nm (htrdr -s lw=4000,4001). Not CIE, not SW.
     * blue_water.mrumtl zeros ρ after 2.08 μm — IR must not bind it. */
    int lw_nbands;
    int lw_nquad_max;
    const float* lw_lambda_lo_nm;
    const float* lw_lambda_hi_nm;
    const float* lw_gas_ka; /* [lw_nbands * lw_nquad_max * nlayers] */
    const int* lw_nquad;
    const float* lw_quad_cdf;
    /* SPECTRAL_SW only: LES x_H2O in the HTCP box + HTGOP humidity tabs.
     * CIE/LW leave these null. Append only. */
    const float* voxel_x_h2o; /* [nz*ny*nx]; <0 => no LES sample, use nominal */
    int sw_h2o_ntab;
    const int* sw_h2o_n;       /* [nlayers] knots per layer */
    const float* sw_h2o_x;     /* [nlayers * sw_h2o_ntab] */
    const float* sw_h2o_ka;    /* [sw_nbands * sw_nquad_max * nlayers * sw_h2o_ntab] */
    const float* sw_h2o_ks;
    const float* sw_h2o_ks_max; /* [sw_nbands * sw_nquad_max * nlayers] Woodcock hat ks */
    int path_sw_iband; /* this path's HTGOP SW interval; -1 = unset */
    int path_sw_iquad;
    /* VOLUME GS cloud medium (pin / fit). Append only — shares voxel lattice.
     * enabled=0 → voxel ka/ks (default). Point query: pid stencil + Σ k_i G_i. */
    struct AtmVolumeGs volume_gs;
};

#ifdef __cplusplus
}
#endif
