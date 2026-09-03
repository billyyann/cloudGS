/* Atmosphere volume (voxel cloud + 1D gas). Copied-PTIR extension.
 * Cloud optical truth is voxels only. Gaussians are surfaces, never k_true. */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

enum AtmSpectralMode {
    ATM_SPECTRAL_SW = 0,
    ATM_SPECTRAL_LW = 1
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
};

#ifdef __cplusplus
}
#endif
