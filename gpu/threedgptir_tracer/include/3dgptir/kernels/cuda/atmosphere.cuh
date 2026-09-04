/* Unbiased voxel Woodcock + 1D gas free-path. Matches htrdr hit_filter.
 * k_true is always the voxel (or gas layer) field — never Gaussian k. */

#pragma once

#include <3dgptir/atmosphere.h>

#if !defined(__CUDACC__) && !defined(__OPTIX__)
#include <math.h>
#endif

#ifndef ATM_INF
#define ATM_INF 1.0e20f
#endif

#ifndef ATM_PI
#define ATM_PI 3.14159265358979323846f
#endif

#ifndef ATM_INV_4PI
#define ATM_INV_4PI 0.07957747154594767f
#endif

#if defined(__CUDACC__) || defined(__OPTIX__)
#define ATM_FN __device__ __forceinline__
#else
#ifndef ATM_FN
#define ATM_FN static inline
#endif
#endif

ATM_FN float atm_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

ATM_FN float atm_dot3(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

ATM_FN void atm_normalize3(float v[3]) {
    const float n = sqrtf(atm_dot3(v, v));
    if (n > 1e-20f) {
        v[0] /= n;
        v[1] /= n;
        v[2] /= n;
    }
}

ATM_FN float atm_ran_exp(float u) {
    /* Ts ~ Exp(1), u in (0,1] */
    const float x = u <= 1e-12f ? 1e-12f : u;
    return -logf(x);
}

ATM_FN float atm_planck_monochromatic(float lambda_m, float temperature_K) {
    const float c = 299792458.f;
    const float h = 6.62607015e-34f;
    const float k = 1.380649e-23f;
    if (temperature_K <= 0.f) {
        return 0.f;
    }
    const float lam2 = lambda_m * lambda_m;
    const float lam5 = lam2 * lam2 * lambda_m;
    const float hc_lkT = (h * c) / (lambda_m * k * temperature_K);
    return (2.f * h * c * c / lam5) / (expf(hc_lkT) - 1.f);
}

ATM_FN int atm_voxel_index(const struct AtmVoxelGrid* g, int ix, int iy, int iz) {
    return (iz * g->ny + iy) * g->nx + ix;
}

ATM_FN int atm_imod(int a, int n) {
    if (n <= 0) {
        return 0;
    }
    int r = a % n;
    if (r < 0) {
        r += n;
    }
    return r;
}

ATM_FN int atm_in_grid(const struct AtmVoxelGrid* g, int ix, int iy, int iz) {
    return ix >= 0 && iy >= 0 && iz >= 0 && ix < g->nx && iy < g->ny && iz < g->nz;
}

/* Snap a boundary sample onto the last/first cell when the ray is entering. */
ATM_FN void atm_snap_ijk_enter(const struct AtmVoxelGrid* g, const float dir[3], int ijk[3]) {
    if (ijk[0] >= g->nx && dir[0] < 0.f) {
        ijk[0] = g->nx - 1;
    }
    if (ijk[1] >= g->ny && dir[1] < 0.f) {
        ijk[1] = g->ny - 1;
    }
    if (ijk[2] >= g->nz && dir[2] < 0.f) {
        ijk[2] = g->nz - 1;
    }
    if (ijk[0] < 0 && dir[0] > 0.f) {
        ijk[0] = 0;
    }
    if (ijk[1] < 0 && dir[1] > 0.f) {
        ijk[1] = 0;
    }
    if (ijk[2] < 0 && dir[2] > 0.f) {
        ijk[2] = 0;
    }
}

ATM_FN float atm_ray_eps(float t) {
    return fmaxf(1e-3f, 1.0e-5f * fabsf(t));
}

ATM_FN int atm_ray_aabb(
    const float org[3],
    const float dir[3],
    const float lo[3],
    const float hi[3],
    float tmin,
    float tmax,
    float* t0,
    float* t1) {
    float a = tmin;
    float b = tmax;
    for (int i = 0; i < 3; ++i) {
        if (fabsf(dir[i]) < 1e-20f) {
            if (org[i] < lo[i] || org[i] > hi[i]) {
                return 0;
            }
        } else {
            float ta = (lo[i] - org[i]) / dir[i];
            float tb = (hi[i] - org[i]) / dir[i];
            if (ta > tb) {
                const float tmp = ta;
                ta = tb;
                tb = tmp;
            }
            a = a > ta ? a : ta;
            b = b < tb ? b : tb;
        }
    }
    if (a >= b) {
        return 0;
    }
    *t0 = a;
    *t1 = b;
    return 1;
}

ATM_FN void atm_world_to_ijk(const struct AtmVoxelGrid* g, const float p[3], int ijk[3]) {
    ijk[0] = (int)floorf((p[0] - g->origin[0]) / g->dx);
    ijk[1] = (int)floorf((p[1] - g->origin[1]) / g->dy);
    ijk[2] = (int)floorf((p[2] - g->origin[2]) / g->dz);
}

/* htrdr -r: XY is periodic; Z is a finite slab. */
ATM_FN void atm_wrap_xy_cell(const struct AtmVoxelGrid* g, int* ix, int* iy) {
    if (g->nx > 0) {
        *ix = atm_imod(*ix, g->nx);
    }
    if (g->ny > 0) {
        *iy = atm_imod(*iy, g->ny);
    }
}

ATM_FN float atm_fetch_ka(const struct AtmVoxelGrid* g, int ix, int iy, int iz) {
    if (!g->ka || iz < 0 || iz >= g->nz) {
        return 0.f;
    }
    atm_wrap_xy_cell(g, &ix, &iy);
    if (!g->ka || ix < 0 || iy < 0 || ix >= g->nx || iy >= g->ny) {
        return 0.f;
    }
    return g->ka[atm_voxel_index(g, ix, iy, iz)];
}

ATM_FN float atm_fetch_ks(const struct AtmVoxelGrid* g, int ix, int iy, int iz) {
    if (!g->ks || iz < 0 || iz >= g->nz) {
        return 0.f;
    }
    atm_wrap_xy_cell(g, &ix, &iy);
    if (!g->ks || ix < 0 || iy < 0 || ix >= g->nx || iy >= g->ny) {
        return 0.f;
    }
    return g->ks[atm_voxel_index(g, ix, iy, iz)];
}

ATM_FN float atm_fetch_temp(const struct AtmVoxelGrid* g, int ix, int iy, int iz) {
    if (!g->temperature || iz < 0 || iz >= g->nz) {
        return 0.f;
    }
    atm_wrap_xy_cell(g, &ix, &iy);
    if (!g->temperature || ix < 0 || iy < 0 || ix >= g->nx || iy >= g->ny) {
        return 0.f;
    }
    return g->temperature[atm_voxel_index(g, ix, iy, iz)];
}

ATM_FN float atm_brick_kmax(const struct AtmVoxelGrid* g, int bx, int by, int bz) {
    if (!g->kext_max || g->nbx <= 0 || g->nby <= 0) {
        return 0.f;
    }
    if (bz < 0 || bz >= g->nbz) {
        return 0.f;
    }
    bx = atm_imod(bx, g->nbx);
    by = atm_imod(by, g->nby);
    return g->kext_max[(bz * g->nby + by) * g->nbx + bx];
}

enum AtmKKind { ATM_K_KA = 0, ATM_K_KS = 1, ATM_K_EXT = 2 };

ATM_FN float atm_cell_k(const struct AtmVoxelGrid* g, int ix, int iy, int iz, int kind) {
    const float ka = atm_fetch_ka(g, ix, iy, iz);
    const float ks = atm_fetch_ks(g, ix, iy, iz);
    if (kind == ATM_K_KA) {
        return ka;
    }
    if (kind == ATM_K_KS) {
        return ks;
    }
    return ka + ks;
}

/* Build per-brick majorant = max ks (htrdr SW scattering_hit_filter). Host only. */
#if !defined(__CUDACC__) && !defined(__OPTIX__)
static inline void atm_build_brick_majorants(struct AtmVoxelGrid* g, float* out_kmax) {
    const int b = g->brick > 0 ? g->brick : 8;
    g->brick = b;
    g->nbx = (g->nx + b - 1) / b;
    g->nby = (g->ny + b - 1) / b;
    g->nbz = (g->nz + b - 1) / b;
    for (int bz = 0; bz < g->nbz; ++bz) {
        for (int by = 0; by < g->nby; ++by) {
            for (int bx = 0; bx < g->nbx; ++bx) {
                float m = 0.f;
                for (int z = bz * b; z < (bz + 1) * b && z < g->nz; ++z) {
                    for (int y = by * b; y < (by + 1) * b && y < g->ny; ++y) {
                        for (int x = bx * b; x < (bx + 1) * b && x < g->nx; ++x) {
                            const float ks = atm_fetch_ks(g, x, y, z);
                            if (ks > m) {
                                m = ks;
                            }
                        }
                    }
                }
                out_kmax[(bz * g->nby + by) * g->nbx + bx] = m;
            }
        }
    }
    g->kext_max = out_kmax;
}
#endif

struct AtmHit {
    int event; /* AtmEvent */
    float t;
    float pos[3];
    float ka, ks, kext, temperature;
};

ATM_FN void atm_hit_clear(struct AtmHit* h) {
    h->event = ATM_EVENT_NONE;
    h->t = ATM_INF;
    h->pos[0] = h->pos[1] = h->pos[2] = 0.f;
    h->ka = h->ks = h->kext = h->temperature = 0.f;
}

ATM_FN float atm_path_ks_scale(const struct AtmParams* atm) {
    return (atm && atm->path_ks_scale > 0.f) ? atm->path_ks_scale : 1.f;
}

ATM_FN float atm_path_ka_scale(const struct AtmParams* atm) {
    return (atm && atm->path_ka_scale > 0.f) ? atm->path_ka_scale : 1.f;
}

/* Voxel Woodcock in [0, tmax]. htrdr scattering_hit_filter: Ts ~ Exp(1),
 * brick ks_max, accept ks/ks_max (walk_ks=1). LW may walk kext (walk_ks=0). */
template <typename Rng>
ATM_FN void atm_woodcock_voxels_into(
    const struct AtmVoxelGrid* g,
    const float org[3],
    const float dir[3],
    float tmax,
    Rng& rng,
    float mie_omega,
    int walk_ks,
    struct AtmHit* hit,
    float ks_scale = 1.f,
    float ka_scale = 1.f) {
    atm_hit_clear(hit);
    if (!g || !hit || g->nx <= 0 || tmax <= 1e-8f) {
        return;
    }

    const float gx0 = g->origin[0];
    const float gy0 = g->origin[1];
    const float gz0 = g->origin[2];
    const float gz1 = gz0 + g->dz * (float)g->nz;

    /* htrdr -r / analog τ: clip Z only. XY is periodic. Do not set t1=path_tmax
     * without a Z exit — nadir would never leave the cloud slab. */
    float t0 = 0.f;
    float t1 = tmax;
    if (fabsf(dir[2]) < 1e-20f) {
        if (org[2] < gz0 || org[2] > gz1) {
            return;
        }
    } else {
        float ta = (gz0 - org[2]) / dir[2];
        float tb = (gz1 - org[2]) / dir[2];
        if (ta > tb) {
            const float tmp = ta;
            ta = tb;
            tb = tmp;
        }
        t0 = t0 > ta ? t0 : ta;
        t1 = t1 < tb ? t1 : tb;
    }
    if (t0 >= t1) {
        return;
    }

    float t = t0 > 0.f ? t0 : 0.f;
    float Ts = atm_ran_exp(rng.next_1d());
    const int b = g->brick > 0 ? g->brick : 8;
    const float bdx = g->dx * (float)b;
    const float bdy = g->dy * (float)b;
    const float bdz = g->dz * (float)b;

    /* Safety cap on null collisions */
    for (int step = 0; step < 100000; ++step) {
        if (t >= t1 - 1e-6f) {
            break;
        }
        float p[3] = {org[0] + dir[0] * (t + atm_ray_eps(t)),
                      org[1] + dir[1] * (t + atm_ray_eps(t)),
                      org[2] + dir[2] * (t + atm_ray_eps(t))};
        /* World brick indices (XY may be any tile; Z must stay in the slab). */
        const int bx = (int)floorf((p[0] - gx0) / bdx);
        const int by = (int)floorf((p[1] - gy0) / bdy);
        int bz = (int)floorf((p[2] - gz0) / bdz);
        if (bz >= g->nbz && dir[2] < 0.f) {
            bz = g->nbz - 1;
        }
        if (bz < 0 && dir[2] > 0.f) {
            bz = 0;
        }
        if (bz < 0 || bz >= g->nbz) {
            break;
        }
        const float kmax0 = atm_brick_kmax(g, bx, by, bz);
        const float kmax = kmax0 * (walk_ks ? ks_scale : fmaxf(ka_scale, ks_scale));

        const float bx0 = gx0 + (float)bx * bdx;
        const float by0 = gy0 + (float)by * bdy;
        const float bz0 = gz0 + (float)bz * bdz;
        float t_exit = t1;
        for (int a = 0; a < 3; ++a) {
            const float o = org[a];
            const float d = dir[a];
            const float lo = a == 0 ? bx0 : (a == 1 ? by0 : bz0);
            const float hi = lo + (a == 0 ? bdx : (a == 1 ? bdy : bdz));
            if (fabsf(d) < 1e-20f) {
                continue;
            }
            const float te = ((d > 0.f ? hi : lo) - o) / d;
            if (te < t_exit && te > t) {
                t_exit = te;
            }
        }

        if (kmax <= 1e-12f) {
            t = t_exit + atm_ray_eps(t_exit);
            continue;
        }

        const float vox_dst = t_exit - t;
        const float Tau = vox_dst * kmax;
        if (Ts > Tau) {
            Ts -= Tau;
            t = t_exit + atm_ray_eps(t_exit);
            continue;
        }

        const float collision_dst = Ts / kmax;
        t += collision_dst;
        if (t > t1) {
            break;
        }
        float pos[3] = {org[0] + dir[0] * t, org[1] + dir[1] * t, org[2] + dir[2] * t};
        int cij[3];
        atm_world_to_ijk(g, pos, cij);
        const float ka = atm_fetch_ka(g, cij[0], cij[1], cij[2]) * ka_scale;
        const float ks = atm_fetch_ks(g, cij[0], cij[1], cij[2]) * ks_scale;
        const float kext = ka + ks;
        const float r = rng.next_1d();
        /* htrdr SW: accept ks/ks_max (scatter only). LW: accept kext then ω. */
        const float k_acc = walk_ks ? ks : kext;
        if (r * kmax < k_acc) {
            hit->t = t;
            hit->pos[0] = pos[0];
            hit->pos[1] = pos[1];
            hit->pos[2] = pos[2];
            hit->ka = ka;
            hit->ks = ks;
            hit->kext = kext;
            hit->temperature = atm_fetch_temp(g, cij[0], cij[1], cij[2]);
            if (walk_ks) {
                hit->event = ATM_EVENT_SCATTER_VOXEL;
            } else {
                const float omega_voxel = kext > 1e-20f ? ks / kext : 0.f;
                const float omega = mie_omega >= 0.f ? mie_omega : omega_voxel;
                hit->event = (rng.next_1d() < omega) ? ATM_EVENT_SCATTER_VOXEL : ATM_EVENT_ABSORB_VOXEL;
            }
            return;
        }
        /* Null collision */
        Ts = atm_ran_exp(rng.next_1d());
    }
}

template <typename Rng>
ATM_FN struct AtmHit atm_woodcock_voxels(
    const struct AtmVoxelGrid* g,
    const float org[3],
    const float dir[3],
    float tmax,
    Rng& rng,
    float mie_omega = -1.f,
    int walk_ks = 1) {
    struct AtmHit hit;
    atm_woodcock_voxels_into(g, org, dir, tmax, rng, mie_omega, walk_ks, &hit, 1.f, 1.f);
    return hit;
}

/* Woodcock inside one brick [t_enter, t_exit]. One Ts for this box (SVX leaf).
 * Writes *out; never returned by value (NVRTC IS ABI). */
template <typename Rng>
ATM_FN void atm_woodcock_brick(
    const struct AtmVoxelGrid* g,
    const float org[3],
    const float dir[3],
    float t_enter,
    float t_exit,
    int bx,
    int by,
    int bz,
    Rng& rng,
    float mie_omega,
    int walk_ks,
    struct AtmHit* out) {
    atm_hit_clear(out);
    if (!g || !out || t_exit <= t_enter + 1e-8f) {
        return;
    }
    const float kmax = atm_brick_kmax(g, bx, by, bz);
    if (kmax <= 1e-12f) {
        return;
    }

    float t = t_enter;
    float Ts = atm_ran_exp(rng.next_1d());
    for (int step = 0; step < 100000; ++step) {
        if (t >= t_exit - 1e-6f) {
            break;
        }
        const float vox_dst = t_exit - t;
        const float Tau = vox_dst * kmax;
        if (Ts > Tau) {
            return;
        }
        const float collision_dst = Ts / kmax;
        t += collision_dst;
        if (t > t_exit) {
            break;
        }
        float pos[3] = {org[0] + dir[0] * t, org[1] + dir[1] * t, org[2] + dir[2] * t};
        int cij[3];
        atm_world_to_ijk(g, pos, cij);
        const float ka = atm_fetch_ka(g, cij[0], cij[1], cij[2]);
        const float ks = atm_fetch_ks(g, cij[0], cij[1], cij[2]);
        const float kext = ka + ks;
        const float r = rng.next_1d();
        const float k_acc = walk_ks ? ks : kext;
        if (r * kmax < k_acc) {
            out->t = t;
            out->pos[0] = pos[0];
            out->pos[1] = pos[1];
            out->pos[2] = pos[2];
            out->ka = ka;
            out->ks = ks;
            out->kext = kext;
            out->temperature = atm_fetch_temp(g, cij[0], cij[1], cij[2]);
            if (walk_ks) {
                out->event = ATM_EVENT_SCATTER_VOXEL;
            } else {
                const float omega_voxel = kext > 1e-20f ? ks / kext : 0.f;
                const float omega = mie_omega >= 0.f ? mie_omega : omega_voxel;
                out->event = (rng.next_1d() < omega) ? ATM_EVENT_SCATTER_VOXEL : ATM_EVENT_ABSORB_VOXEL;
            }
            return;
        }
        Ts = atm_ran_exp(rng.next_1d());
    }
}

/* 1D layered gas: SW walks ks (scatter only); LW walks kext then ka/ks split.
 * Writes *hit (NVRTC must not return AtmHit by value). */
template <typename Rng>
ATM_FN void atm_gas_free_path_into(
    const struct AtmGasLayers* gas,
    const float org[3],
    const float dir[3],
    float tmax,
    Rng& rng,
    int walk_ks,
    struct AtmHit* hit) {
    atm_hit_clear(hit);
    if (!gas || !hit || gas->nlayers <= 0 || !gas->z_iface || tmax <= 1e-8f) {
        return;
    }
    float Ts = atm_ran_exp(rng.next_1d());
    float t = 0.f;
    float z = org[2];
    const float dz = dir[2];

    for (int safety = 0; safety < gas->nlayers * 4 + 8; ++safety) {
        if (t >= tmax) {
            break;
        }
        int layer = -1;
        for (int i = 0; i < gas->nlayers; ++i) {
            const float z0 = gas->z_iface[i];
            const float z1 = gas->z_iface[i + 1];
            if (z >= z0 && z < z1) {
                layer = i;
                break;
            }
        }
        if (layer < 0) {
            /* Outside atmosphere: skip to next interface if heading toward it */
            if (dz > 0.f && z < gas->z_iface[0]) {
                const float dt = (gas->z_iface[0] - z) / dz;
                t += dt;
                z = gas->z_iface[0];
                if (t >= tmax) {
                    break;
                }
                continue;
            }
            if (dz < 0.f && z > gas->z_iface[gas->nlayers]) {
                const float dt = (gas->z_iface[gas->nlayers] - z) / dz;
                t += dt;
                z = gas->z_iface[gas->nlayers] - 1e-4f;
                if (t >= tmax) {
                    break;
                }
                continue;
            }
            break;
        }
        const float z0 = gas->z_iface[layer];
        const float z1 = gas->z_iface[layer + 1];
        float t_exit_layer = tmax - t;
        if (fabsf(dz) > 1e-12f) {
            const float z_exit = dz > 0.f ? z1 : z0;
            t_exit_layer = (z_exit - z) / dz;
            if (t_exit_layer < 0.f) {
                t_exit_layer = 0.f;
            }
        }
        const float ka = gas->ka ? gas->ka[layer] : 0.f;
        const float ks = gas->ks ? gas->ks[layer] : 0.f;
        const float kext = ka + ks;
        const float k_walk = walk_ks ? ks : kext;
        if (k_walk <= 1e-12f) {
            t += t_exit_layer + atm_ray_eps(t);
            z = org[2] + dir[2] * t;
            continue;
        }
        const float Tau = t_exit_layer * k_walk;
        if (Ts > Tau) {
            Ts -= Tau;
            t += t_exit_layer + atm_ray_eps(t);
            z = org[2] + dir[2] * t;
            continue;
        }
        const float dt = Ts / k_walk;
        t += dt;
        if (t > tmax) {
            break;
        }
        float pos[3] = {org[0] + dir[0] * t, org[1] + dir[1] * t, org[2] + dir[2] * t};
        if (walk_ks) {
            hit->event = ATM_EVENT_SCATTER_GAS;
        } else {
            const float r = rng.next_1d();
            hit->event = (r < ka / kext) ? ATM_EVENT_ABSORB_GAS : ATM_EVENT_SCATTER_GAS;
        }
        hit->t = t;
        hit->pos[0] = pos[0];
        hit->pos[1] = pos[1];
        hit->pos[2] = pos[2];
        hit->ka = ka;
        hit->ks = ks;
        hit->kext = kext;
        hit->temperature = gas->temperature ? gas->temperature[layer] : 0.f;
        return;
    }
}

template <typename Rng>
ATM_FN struct AtmHit atm_gas_free_path(
    const struct AtmGasLayers* gas,
    const float org[3],
    const float dir[3],
    float tmax,
    Rng& rng,
    int walk_ks = 1) {
    struct AtmHit hit;
    atm_gas_free_path_into(gas, org, dir, tmax, rng, walk_ks, &hit);
    return hit;
}

/* Moller-Trumbore. Returns t or INF. */
ATM_FN float atm_ray_triangle(
    const float org[3],
    const float dir[3],
    const float v0[3],
    const float v1[3],
    const float v2[3]) {
    const float e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    const float e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    const float px = dir[1] * e2[2] - dir[2] * e2[1];
    const float py = dir[2] * e2[0] - dir[0] * e2[2];
    const float pz = dir[0] * e2[1] - dir[1] * e2[0];
    const float det = e1[0] * px + e1[1] * py + e1[2] * pz;
    if (fabsf(det) < 1e-12f) {
        return ATM_INF;
    }
    const float inv = 1.f / det;
    const float tvec[3] = {org[0] - v0[0], org[1] - v0[1], org[2] - v0[2]};
    const float u = (tvec[0] * px + tvec[1] * py + tvec[2] * pz) * inv;
    if (u < 0.f || u > 1.f) {
        return ATM_INF;
    }
    const float qx = tvec[1] * e1[2] - tvec[2] * e1[1];
    const float qy = tvec[2] * e1[0] - tvec[0] * e1[2];
    const float qz = tvec[0] * e1[1] - tvec[1] * e1[0];
    const float v = (dir[0] * qx + dir[1] * qy + dir[2] * qz) * inv;
    if (v < 0.f || u + v > 1.f) {
        return ATM_INF;
    }
    const float t = (e2[0] * qx + e2[1] * qy + e2[2] * qz) * inv;
    return t > 1e-6f ? t : ATM_INF;
}

/* Regular heightfield: nvert = n*n, ntri = 2*(n-1)^2, row-major (iy * n + ix). */
ATM_FN int atm_mesh_grid_n(const struct AtmMesh* mesh) {
    if (!mesh || mesh->nvert < 4 || mesh->ntri < 2 || !mesh->vertices || !mesh->indices) {
        return 0;
    }
    const int n = (int)(sqrt((double)mesh->nvert) + 0.5);
    if (n < 2 || n * n != mesh->nvert) {
        return 0;
    }
    const int n1 = n - 1;
    if (mesh->ntri != 2 * n1 * n1) {
        return 0;
    }
    const float dx = mesh->vertices[3] - mesh->vertices[0];
    const float dy = mesh->vertices[n * 3 + 1] - mesh->vertices[1];
    if (!(dx > 1e-3f && dy > 1e-3f)) {
        return 0;
    }
    return n;
}

ATM_FN void atm_mesh_fill_hit(const struct AtmMesh* mesh, int tri, float t, struct AtmMeshHit* hit) {
    hit->t = t;
    hit->tri = tri;
    const int i0 = mesh->indices[tri * 3 + 0];
    const int i1 = mesh->indices[tri * 3 + 1];
    const int i2 = mesh->indices[tri * 3 + 2];
    const float* v0 = mesh->vertices + i0 * 3;
    const float* v1 = mesh->vertices + i1 * 3;
    const float* v2 = mesh->vertices + i2 * 3;
    const float e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    const float e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    hit->n[0] = e1[1] * e2[2] - e1[2] * e2[1];
    hit->n[1] = e1[2] * e2[0] - e1[0] * e2[2];
    hit->n[2] = e1[0] * e2[1] - e1[1] * e2[0];
    atm_normalize3(hit->n);
    if (mesh->albedo) {
        hit->albedo[0] = mesh->albedo[tri * 3 + 0];
        hit->albedo[1] = mesh->albedo[tri * 3 + 1];
        hit->albedo[2] = mesh->albedo[tri * 3 + 2];
    }
    if (mesh->temperature) {
        hit->temperature = mesh->temperature[tri];
    }
}

ATM_FN void atm_mesh_test_grid_cell(
    const struct AtmMesh* mesh,
    int n,
    int ix,
    int iy,
    const float org[3],
    const float dir[3],
    float* best,
    int* best_i) {
    if (ix < 0 || iy < 0 || ix >= n - 1 || iy >= n - 1) {
        return;
    }
    const int cell = iy * (n - 1) + ix;
    for (int k = 0; k < 2; ++k) {
        const int tri = cell * 2 + k;
        const int i0 = mesh->indices[tri * 3 + 0];
        const int i1 = mesh->indices[tri * 3 + 1];
        const int i2 = mesh->indices[tri * 3 + 2];
        const float t = atm_ray_triangle(
            org,
            dir,
            mesh->vertices + i0 * 3,
            mesh->vertices + i1 * 3,
            mesh->vertices + i2 * 3);
        if (t < *best) {
            *best = t;
            *best_i = tri;
        }
    }
}

/* XY DDA over a heightfield. Brute-force loop is only for tiny non-grid meshes. */
ATM_FN int atm_mesh_grid_closest(
    const struct AtmMesh* mesh,
    int n,
    const float org[3],
    const float dir[3],
    float tmax,
    float* best_t,
    int* best_i) {
    const float gx0 = mesh->vertices[0];
    const float gy0 = mesh->vertices[1];
    const float dx = mesh->vertices[3] - gx0;
    const float dy = mesh->vertices[n * 3 + 1] - gy0;
    const float gx1 = gx0 + dx * (float)(n - 1);
    const float gy1 = gy0 + dy * (float)(n - 1);
    float t0 = 0.f;
    float t1 = tmax;
    for (int a = 0; a < 2; ++a) {
        const float o = org[a];
        const float d = dir[a];
        const float lo = a == 0 ? gx0 : gy0;
        const float hi = a == 0 ? gx1 : gy1;
        if (fabsf(d) < 1e-20f) {
            if (o < lo || o > hi) {
                return 0;
            }
        } else {
            float ta = (lo - o) / d;
            float tb = (hi - o) / d;
            if (ta > tb) {
                const float tmp = ta;
                ta = tb;
                tb = tmp;
            }
            t0 = t0 > ta ? t0 : ta;
            t1 = t1 < tb ? t1 : tb;
        }
    }
    if (t0 >= t1) {
        return 0;
    }
    float t = t0 + 1e-4f;
    const float px = org[0] + dir[0] * t;
    const float py = org[1] + dir[1] * t;
    int ix = (int)floorf((px - gx0) / dx);
    int iy = (int)floorf((py - gy0) / dy);
    if (ix < 0) {
        ix = 0;
    }
    if (iy < 0) {
        iy = 0;
    }
    if (ix > n - 2) {
        ix = n - 2;
    }
    if (iy > n - 2) {
        iy = n - 2;
    }
    int step[2];
    float tDelta[2], tMax[2];
    const int ijk0[2] = {ix, iy};
    const float o2[2] = {org[0], org[1]};
    const float d2[2] = {dir[0], dir[1]};
    const float origin[2] = {gx0, gy0};
    const float sp[2] = {dx, dy};
    for (int a = 0; a < 2; ++a) {
        if (d2[a] > 1e-20f) {
            step[a] = 1;
            tDelta[a] = sp[a] / d2[a];
            tMax[a] = (origin[a] + sp[a] * (float)(ijk0[a] + 1) - o2[a]) / d2[a];
        } else if (d2[a] < -1e-20f) {
            step[a] = -1;
            tDelta[a] = sp[a] / (-d2[a]);
            tMax[a] = (origin[a] + sp[a] * (float)ijk0[a] - o2[a]) / d2[a];
        } else {
            step[a] = 0;
            tDelta[a] = ATM_INF;
            tMax[a] = ATM_INF;
        }
    }
    float best = tmax;
    int besti = -1;
    const int nsteps = 2 * (n + 2);
    for (int s = 0; s < nsteps; ++s) {
        if (t >= best - 1e-6f || t >= t1 - 1e-6f) {
            break;
        }
        atm_mesh_test_grid_cell(mesh, n, ix, iy, org, dir, &best, &besti);
        const float tNext = fminf(t1, fminf(tMax[0], tMax[1]));
        if (tNext >= t1 - 1e-6f) {
            break;
        }
        if (tMax[0] < tMax[1]) {
            ix += step[0];
            tMax[0] += tDelta[0];
        } else {
            iy += step[1];
            tMax[1] += tDelta[1];
        }
        t = tNext;
        if (ix < -1 || iy < -1 || ix > n - 1 || iy > n - 1) {
            break;
        }
    }
    if (besti < 0) {
        return 0;
    }
    *best_t = best;
    *best_i = besti;
    return 1;
}

ATM_FN float atm_mesh_closest(
    const struct AtmMesh* mesh,
    const float org[3],
    const float dir[3],
    float tmax) {
    if (!mesh || mesh->ntri <= 0) {
        return ATM_INF;
    }
    const int n = atm_mesh_grid_n(mesh);
    if (n > 0) {
        float best = tmax;
        int best_i = -1;
        if (atm_mesh_grid_closest(mesh, n, org, dir, tmax, &best, &best_i)) {
            return best;
        }
        return ATM_INF;
    }
    /* Tiny meshes only (two-triangle sand). Never scan millions of tris. */
    if (mesh->ntri > 64) {
        return ATM_INF;
    }
    float best = tmax;
    for (int i = 0; i < mesh->ntri; ++i) {
        const int i0 = mesh->indices[i * 3 + 0];
        const int i1 = mesh->indices[i * 3 + 1];
        const int i2 = mesh->indices[i * 3 + 2];
        const float* v0 = mesh->vertices + i0 * 3;
        const float* v1 = mesh->vertices + i1 * 3;
        const float* v2 = mesh->vertices + i2 * 3;
        const float t = atm_ray_triangle(org, dir, v0, v1, v2);
        if (t < best) {
            best = t;
        }
    }
    return best < tmax ? best : ATM_INF;
}

ATM_FN void atm_mesh_hit_clear(struct AtmMeshHit* h) {
    h->t = ATM_INF;
    h->tri = -1;
    h->n[0] = h->n[1] = 0.f;
    h->n[2] = 1.f;
    h->albedo[0] = h->albedo[1] = h->albedo[2] = 0.5f;
    h->temperature = 0.f;
}

ATM_FN struct AtmMeshHit atm_mesh_closest_hit(
    const struct AtmMesh* mesh,
    const float org[3],
    const float dir[3],
    float tmax) {
    struct AtmMeshHit hit;
    atm_mesh_hit_clear(&hit);
    if (!mesh || mesh->ntri <= 0) {
        return hit;
    }
    float best = tmax;
    int best_i = -1;
    const int n = atm_mesh_grid_n(mesh);
    if (n > 0) {
        if (!atm_mesh_grid_closest(mesh, n, org, dir, tmax, &best, &best_i)) {
            return hit;
        }
    } else if (mesh->ntri > 64) {
        return hit;
    } else {
        for (int i = 0; i < mesh->ntri; ++i) {
            const int i0 = mesh->indices[i * 3 + 0];
            const int i1 = mesh->indices[i * 3 + 1];
            const int i2 = mesh->indices[i * 3 + 2];
            const float* v0 = mesh->vertices + i0 * 3;
            const float* v1 = mesh->vertices + i1 * 3;
            const float* v2 = mesh->vertices + i2 * 3;
            const float t = atm_ray_triangle(org, dir, v0, v1, v2);
            if (t < best) {
                best = t;
                best_i = i;
            }
        }
        if (best_i < 0) {
            return hit;
        }
    }
    atm_mesh_fill_hit(mesh, best_i, best, &hit);
    if (atm_dot3(hit.n, dir) > 0.f) {
        hit.n[0] = -hit.n[0];
        hit.n[1] = -hit.n[1];
        hit.n[2] = -hit.n[2];
    }
    return hit;
}

ATM_FN float atm_hg_phase(float g, float mu) {
    const float g2 = g * g;
    const float d = 1.f + g2 - 2.f * g * mu;
    const float d32 = d * sqrtf(fmaxf(d, 1e-20f));
    return (1.f - g2) * ATM_INV_4PI / fmaxf(d32, 1e-20f);
}

ATM_FN float atm_rayleigh_phase(float mu) {
    return (3.f / (16.f * ATM_PI)) * (1.f + mu * mu);
}

ATM_FN int atm_dir_in_sun_cone(const struct AtmSun* sun, const float dir[3]) {
    if (!sun) {
        return 0;
    }
    const float mu = atm_dot3(sun->dir, dir);
    return mu >= sun->cos_half_angle - 1e-6f;
}

template <typename Rng>
ATM_FN void atm_sample_sun(const struct AtmSun* sun, Rng& rng, float wo[3], float* pdf) {
    wo[0] = sun->dir[0];
    wo[1] = sun->dir[1];
    wo[2] = sun->dir[2];
    const float mu_min = sun->cos_half_angle;
    if (mu_min >= 0.999999f) {
        *pdf = 1.f;
        return;
    }
    const float u1 = rng.next_1d();
    const float u2 = rng.next_1d();
    const float mu = 1.f - u1 * (1.f - mu_min);
    const float st = sqrtf(fmaxf(0.f, 1.f - mu * mu));
    const float phi = 2.f * ATM_PI * u2;
    float T[3], B[3];
    if (fabsf(sun->dir[2]) < 0.999f) {
        T[0] = sun->dir[1];
        T[1] = -sun->dir[0];
        T[2] = 0.f;
    } else {
        T[0] = 1.f;
        T[1] = 0.f;
        T[2] = 0.f;
    }
    atm_normalize3(T);
    B[0] = sun->dir[1] * T[2] - sun->dir[2] * T[1];
    B[1] = sun->dir[2] * T[0] - sun->dir[0] * T[2];
    B[2] = sun->dir[0] * T[1] - sun->dir[1] * T[0];
    const float cp = cosf(phi);
    const float sp = sinf(phi);
    wo[0] = st * cp * T[0] + st * sp * B[0] + mu * sun->dir[0];
    wo[1] = st * cp * T[1] + st * sp * B[1] + mu * sun->dir[1];
    wo[2] = st * cp * T[2] + st * sp * B[2] + mu * sun->dir[2];
    atm_normalize3(wo);
    *pdf = 1.f / fmaxf(2.f * ATM_PI * (1.f - mu_min), 1e-12f);
}

ATM_FN void atm_sample_hg(float g, float u1, float u2, const float wi[3], float wo[3]) {
    float cos_theta;
    if (fabsf(g) < 1e-5f) {
        cos_theta = 2.f * u1 - 1.f;
    } else {
        const float g2 = g * g;
        const float tmp = (1.f - g2) / (1.f - g + 2.f * g * u1);
        cos_theta = (1.f + g2 - tmp * tmp) / (2.f * g);
    }
    cos_theta = atm_clampf(cos_theta, -1.f, 1.f);
    const float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));
    const float phi = 6.28318530718f * u2;
    float T[3], B[3];
    if (fabsf(wi[2]) < 0.999f) {
        T[0] = wi[1];
        T[1] = -wi[0];
        T[2] = 0.f;
    } else {
        T[0] = 1.f;
        T[1] = 0.f;
        T[2] = 0.f;
    }
    atm_normalize3(T);
    B[0] = wi[1] * T[2] - wi[2] * T[1];
    B[1] = wi[2] * T[0] - wi[0] * T[2];
    B[2] = wi[0] * T[1] - wi[1] * T[0];
    const float cp = cosf(phi);
    const float sp = sinf(phi);
    wo[0] = sin_theta * cp * T[0] + sin_theta * sp * B[0] + cos_theta * wi[0];
    wo[1] = sin_theta * cp * T[1] + sin_theta * sp * B[1] + cos_theta * wi[1];
    wo[2] = sin_theta * cp * T[2] + sin_theta * sp * B[2] + cos_theta * wi[2];
    atm_normalize3(wo);
}

/* Rayleigh: pdf ∝ 1+cos²θ over mu∈[-1,1]. Rejection is unbiased. */
template <typename Rng>
ATM_FN void atm_sample_rayleigh(Rng& rng, const float wi[3], float wo[3]) {
    float mu = 0.f;
    for (int i = 0; i < 64; ++i) {
        mu = 2.f * rng.next_1d() - 1.f;
        const float u = rng.next_1d();
        if (u * 2.f <= 1.f + mu * mu) {
            break;
        }
    }
    const float sin_theta = sqrtf(fmaxf(0.f, 1.f - mu * mu));
    const float phi = 6.28318530718f * rng.next_1d();
    float T[3], B[3];
    if (fabsf(wi[2]) < 0.999f) {
        T[0] = wi[1];
        T[1] = -wi[0];
        T[2] = 0.f;
    } else {
        T[0] = 1.f;
        T[1] = 0.f;
        T[2] = 0.f;
    }
    atm_normalize3(T);
    B[0] = wi[1] * T[2] - wi[2] * T[1];
    B[1] = wi[2] * T[0] - wi[0] * T[2];
    B[2] = wi[0] * T[1] - wi[1] * T[0];
    const float cp = cosf(phi);
    const float sp = sinf(phi);
    wo[0] = sin_theta * cp * T[0] + sin_theta * sp * B[0] + mu * wi[0];
    wo[1] = sin_theta * cp * T[1] + sin_theta * sp * B[1] + mu * wi[1];
    wo[2] = sin_theta * cp * T[2] + sin_theta * sp * B[2] + mu * wi[2];
    atm_normalize3(wo);
}

/* Competing risks: surface tmax truncates volume. Volume never reads Gaussian k.
 * Writes *out (NVRTC must not return AtmHit by value). */
template <typename Rng>
ATM_FN void atm_next_volume_event_into(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float t_surface,
    Rng& rng,
    struct AtmHit* out) {
    atm_hit_clear(out);
    if (!out) {
        return;
    }
    const float tmax = t_surface < atm->path_tmax ? t_surface : atm->path_tmax;
    out->t = tmax;
    const int walk_ks = atm->spectral_mode != ATM_SPECTRAL_LW;
    const float omega_arg = atm->mie_omega > 0.f ? atm->mie_omega : -1.f;
    struct AtmHit gas;
    struct AtmHit vox;
    atm_gas_free_path_into(&atm->gas, org, dir, tmax, rng, walk_ks, &gas);
    atm_woodcock_voxels_into(
        &atm->voxels,
        org,
        dir,
        tmax,
        rng,
        omega_arg,
        walk_ks,
        &vox,
        atm_path_ks_scale(atm),
        atm_path_ka_scale(atm));
    if (gas.event != ATM_EVENT_NONE && gas.t <= vox.t && gas.t < tmax) {
        *out = gas;
        return;
    }
    if (vox.event != ATM_EVENT_NONE && vox.t < tmax) {
        *out = vox;
        return;
    }
}

template <typename Rng>
ATM_FN struct AtmHit atm_next_volume_event(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float t_surface,
    Rng& rng) {
    struct AtmHit hit;
    atm_next_volume_event_into(atm, org, dir, t_surface, rng, &hit);
    return hit;
}

ATM_FN float atm_layer_k(const struct AtmGasLayers* gas, int layer, int kind) {
    const float ka = gas->ka ? gas->ka[layer] : 0.f;
    const float ks = gas->ks ? gas->ks[layer] : 0.f;
    if (kind == ATM_K_KA) {
        return ka;
    }
    if (kind == ATM_K_KS) {
        return ks;
    }
    return ka + ks;
}

ATM_FN void atm_gas_k_at_z(const struct AtmGasLayers* gas, float z, float* ka, float* ks) {
    *ka = 0.f;
    *ks = 0.f;
    if (!gas || gas->nlayers <= 0 || !gas->z_iface) {
        return;
    }
    for (int i = 0; i < gas->nlayers; ++i) {
        if (z >= gas->z_iface[i] && z < gas->z_iface[i + 1]) {
            *ka = gas->ka ? gas->ka[i] : 0.f;
            *ks = gas->ks ? gas->ks[i] : 0.f;
            return;
        }
    }
}

/* Analog optical depth through the regular voxel grid (Amanatides–Woo DDA).
 * Homogeneous cells ⇒ exp(-Tmin) matches htrdr transmissivity residual.
 * kind: ATM_K_KA / ATM_K_KS / ATM_K_EXT (gas+particle via caller sum). */
ATM_FN float atm_voxel_optical_depth(
    const struct AtmVoxelGrid* g,
    const float org[3],
    const float dir[3],
    float tmax,
    int kind = ATM_K_EXT,
    float ks_scale = 1.f,
    float ka_scale = 1.f) {
    if (!g || g->nx <= 0 || tmax <= 1e-8f) {
        return 0.f;
    }
    const float gx0 = g->origin[0];
    const float gy0 = g->origin[1];
    const float gz0 = g->origin[2];
    const float dx[3] = {g->dx, g->dy, g->dz};
    const float origin[3] = {gx0, gy0, gz0};
    const int n[3] = {g->nx, g->ny, g->nz};
    const float gz1 = gz0 + dx[2] * (float)n[2];
    /* NEE analog τ: clip Z only (htrdr -r). XY is periodic. Camera Woodcock
     * uses the same Z-slab clip so nadir still exits at z=0 and hits mesh. */
    float t0 = 0.f;
    float t1 = tmax;
    if (fabsf(dir[2]) < 1e-20f) {
        if (org[2] < gz0 || org[2] > gz1) {
            return 0.f;
        }
    } else {
        float ta = (gz0 - org[2]) / dir[2];
        float tb = (gz1 - org[2]) / dir[2];
        if (ta > tb) {
            const float tmp = ta;
            ta = tb;
            tb = tmp;
        }
        t0 = t0 > ta ? t0 : ta;
        t1 = t1 < tb ? t1 : tb;
    }
    if (t0 >= t1) {
        return 0.f;
    }
    float t = t0 > 0.f ? t0 : 0.f;
    t += atm_ray_eps(t);
    if (t >= t1) {
        return 0.f;
    }
    float p[3] = {org[0] + dir[0] * t, org[1] + dir[1] * t, org[2] + dir[2] * t};
    int ijk[3];
    atm_world_to_ijk(g, p, ijk);
    if (ijk[2] >= n[2] && dir[2] < 0.f) {
        ijk[2] = n[2] - 1;
    }
    if (ijk[2] < 0 && dir[2] > 0.f) {
        ijk[2] = 0;
    }
    if (ijk[2] < 0 || ijk[2] >= n[2]) {
        return 0.f;
    }
    int step[3];
    float tDelta[3];
    float tMax[3];
    for (int a = 0; a < 3; ++a) {
        if (dir[a] > 1e-20f) {
            step[a] = 1;
            tDelta[a] = dx[a] / dir[a];
            tMax[a] = (origin[a] + (float)(ijk[a] + 1) * dx[a] - org[a]) / dir[a];
        } else if (dir[a] < -1e-20f) {
            step[a] = -1;
            tDelta[a] = dx[a] / (-dir[a]);
            tMax[a] = (origin[a] + (float)ijk[a] * dx[a] - org[a]) / dir[a];
        } else {
            step[a] = 0;
            tDelta[a] = ATM_INF;
            tMax[a] = ATM_INF;
        }
    }
    float tau = 0.f;
    const int nsteps = 2 * (n[0] + n[1]) + n[2] + 8;
    for (int s = 0; s < nsteps; ++s) {
        if (t >= t1 - 1e-6f) {
            break;
        }
        if (ijk[2] < 0 || ijk[2] >= n[2]) {
            break;
        }
        const int fx = atm_imod(ijk[0], n[0]);
        const int fy = atm_imod(ijk[1], n[1]);
        const float tNext = fminf(t1, fminf(tMax[0], fminf(tMax[1], tMax[2])));
        const float ka = atm_fetch_ka(g, fx, fy, ijk[2]) * ka_scale;
        const float ks = atm_fetch_ks(g, fx, fy, ijk[2]) * ks_scale;
        float kv = ka + ks;
        if (kind == ATM_K_KA) {
            kv = ka;
        } else if (kind == ATM_K_KS) {
            kv = ks;
        }
        tau += kv * fmaxf(tNext - t, 0.f);
        if (tNext >= t1 - 1e-6f) {
            break;
        }
        t = tNext;
        if (tMax[0] <= tMax[1] && tMax[0] <= tMax[2]) {
            ijk[0] += step[0];
            tMax[0] += tDelta[0];
        } else if (tMax[1] <= tMax[2]) {
            ijk[1] += step[1];
            tMax[1] += tDelta[1];
        } else {
            ijk[2] += step[2];
            tMax[2] += tDelta[2];
        }
    }
    return tau;
}

ATM_FN float atm_gas_optical_depth(
    const struct AtmGasLayers* gas,
    const float org[3],
    const float dir[3],
    float tmax,
    int kind = ATM_K_EXT) {
    if (!gas || gas->nlayers <= 0 || !gas->z_iface || tmax <= 1e-8f) {
        return 0.f;
    }
    float t = 0.f;
    float z = org[2];
    const float dz = dir[2];
    float tau = 0.f;
    for (int safety = 0; safety < gas->nlayers * 4 + 8; ++safety) {
        if (t >= tmax) {
            break;
        }
        int layer = -1;
        for (int i = 0; i < gas->nlayers; ++i) {
            const float z0 = gas->z_iface[i];
            const float z1 = gas->z_iface[i + 1];
            if (z >= z0 && z < z1) {
                layer = i;
                break;
            }
        }
        if (layer < 0) {
            if (dz > 0.f && z < gas->z_iface[0]) {
                const float dt = (gas->z_iface[0] - z) / dz;
                t += dt;
                z = gas->z_iface[0];
                if (t >= tmax) {
                    break;
                }
                continue;
            }
            if (dz < 0.f && z > gas->z_iface[gas->nlayers]) {
                const float dt = (gas->z_iface[gas->nlayers] - z) / dz;
                t += dt;
                z = gas->z_iface[gas->nlayers] - 1e-4f;
                if (t >= tmax) {
                    break;
                }
                continue;
            }
            break;
        }
        const float z0 = gas->z_iface[layer];
        const float z1 = gas->z_iface[layer + 1];
        float t_exit_layer = tmax - t;
        if (fabsf(dz) > 1e-12f) {
            const float z_exit = dz > 0.f ? z1 : z0;
            t_exit_layer = (z_exit - z) / dz;
            if (t_exit_layer < 0.f) {
                t_exit_layer = 0.f;
            }
        }
        const float ds = fminf(t_exit_layer, tmax - t);
        tau += atm_layer_k(gas, layer, kind) * fmaxf(ds, 0.f);
        t += ds + atm_ray_eps(t);
        z = org[2] + dir[2] * t;
    }
    return tau;
}

ATM_FN float atm_optical_depth(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float tmax,
    int kind) {
    return atm_voxel_optical_depth(
               &atm->voxels,
               org,
               dir,
               tmax,
               kind,
               atm_path_ks_scale(atm),
               atm_path_ka_scale(atm)) +
           atm_gas_optical_depth(&atm->gas, org, dir, tmax, kind);
}

ATM_FN float atm_transmittance_kind(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float tmax,
    int kind) {
    const float tau = atm_optical_depth(atm, org, dir, tmax, kind);
    if (tau <= 0.f) {
        return 1.f;
    }
    if (tau > 80.f) {
        return 0.f;
    }
    return expf(-tau);
}

ATM_FN float atm_volume_transmittance(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float tmax) {
    return atm_transmittance_kind(atm, org, dir, tmax, ATM_K_EXT);
}

ATM_FN float atm_absorption_transmittance(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float tmax) {
    return atm_transmittance_kind(atm, org, dir, tmax, ATM_K_KA);
}

/* Unbiased shadow / direct visibility through volume. Real collision => occluded. */
template <typename Rng>
ATM_FN int atm_volume_occluded(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float tmax,
    Rng& rng) {
    struct AtmHit h;
    atm_next_volume_event_into(atm, org, dir, tmax, rng, &h);
    return h.event != ATM_EVENT_NONE && h.t < tmax;
}

/* CIE 1931 analytic CMFs: Wyman, Sloan, Shirley JCGT 2013. Same as htrdr. */
ATM_FN float atm_cie_xbar(float lambda) {
    const float a = (lambda - 442.0f) * (lambda < 442.0f ? 0.0624f : 0.0374f);
    const float b = (lambda - 599.8f) * (lambda < 599.8f ? 0.0264f : 0.0323f);
    const float c = (lambda - 501.1f) * (lambda < 501.1f ? 0.0490f : 0.0382f);
    return 0.362f * expf(-0.5f * a * a) + 1.056f * expf(-0.5f * b * b) - 0.065f * expf(-0.5f * c * c);
}

ATM_FN float atm_cie_ybar(float lambda) {
    const float a = (lambda - 568.8f) * (lambda < 568.8f ? 0.0213f : 0.0247f);
    const float b = (lambda - 530.9f) * (lambda < 530.9f ? 0.0613f : 0.0322f);
    return 0.821f * expf(-0.5f * a * a) + 0.286f * expf(-0.5f * b * b);
}

ATM_FN float atm_cie_zbar(float lambda) {
    const float a = (lambda - 437.0f) * (lambda < 437.0f ? 0.0845f : 0.0278f);
    const float b = (lambda - 459.0f) * (lambda < 459.0f ? 0.0385f : 0.0725f);
    return 1.217f * expf(-0.5f * a * a) + 0.681f * expf(-0.5f * b * b);
}

ATM_FN float atm_cie_cmf(int ch, float lambda) {
    if (ch == 0) {
        return atm_cie_xbar(lambda);
    }
    if (ch == 1) {
        return atm_cie_ybar(lambda);
    }
    return atm_cie_zbar(lambda);
}

ATM_FN int atm_sw_find_band(const struct AtmParams* atm, float wlen_nm) {
    if (!atm || atm->sw_nbands <= 0 || !atm->sw_lambda_lo_nm || !atm->sw_lambda_hi_nm) {
        return -1;
    }
    for (int i = 0; i < atm->sw_nbands; ++i) {
        if (wlen_nm >= atm->sw_lambda_lo_nm[i] && wlen_nm < atm->sw_lambda_hi_nm[i]) {
            return i;
        }
    }
    const int last = atm->sw_nbands - 1;
    if (wlen_nm >= atm->sw_lambda_lo_nm[last] && wlen_nm <= atm->sw_lambda_hi_nm[last]) {
        return last;
    }
    return -1;
}

/* Sample λ ~ CMF (htrdr_ran_wlen_cie_xyz). pdf is constant 1/∫CMF (nm^-1). */
ATM_FN float atm_cie_sample_wlen_nm(const struct AtmParams* atm, int ch, float r0, float r1) {
    const int n = atm->cie_nbins;
    const float* cdf = atm->cie_cdf + ch * n;
    if (r0 < 0.f) {
        r0 = 0.f;
    }
    if (r0 >= 1.f) {
        r0 = 0.999999f;
    }
    int lo = 0;
    int hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (cdf[mid] <= r0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const int iband = lo;
    const float lambda_min = atm->cie_lambda0_nm + atm->cie_dlambda_nm * (float)iband;
    const float lambda_max = lambda_min + atm->cie_dlambda_nm;
    const float f_min = atm_cie_cmf(ch, lambda_min);
    const float f_max = atm_cie_cmf(ch, lambda_max);
    const float band_len = atm->cie_dlambda_nm;
    const float a = 0.5f * (f_max - f_min) / band_len;
    const float b = (lambda_max * f_min - lambda_min * f_max) / band_len;
    const float c = -lambda_min * f_min + lambda_min * lambda_min * a;
    const float d = 0.5f * (f_max + f_min) * band_len;
    if (r1 < 0.f) {
        r1 = 0.f;
    }
    if (r1 >= 1.f) {
        r1 = 0.999999f;
    }
    if (fabsf(a) < 1e-12f) {
        return lambda_min + r1 * band_len;
    }
    const float disc = b * b - 4.f * a * (c - d * r1);
    if (disc < 0.f) {
        return 0.5f * (lambda_min + lambda_max);
    }
    const float s = sqrtf(disc);
    const float l1 = (-b - s) / (2.f * a);
    const float l2 = (-b + s) / (2.f * a);
    if (l1 >= lambda_min && l1 < lambda_max) {
        return l1;
    }
    if (l2 >= lambda_min && l2 < lambda_max) {
        return l2;
    }
    return 0.5f * (lambda_min + lambda_max);
}

ATM_FN int atm_sample_quad(const float* cdf, int n, float r) {
    if (!cdf || n <= 1) {
        return 0;
    }
    if (r < 0.f) {
        r = 0.f;
    }
    if (r >= 1.f) {
        r = 0.999999f;
    }
    int lo = 0;
    int hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (cdf[mid] <= r) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/* Bind per-path gas quad + Planck sun + Mie band scale. pdf_m = rcp_integral * 1e9 (m^-1). */
ATM_FN void atm_cie_bind_path(struct AtmParams* atm, int ch, float r0, float r1, float r_quad, float* pdf_m) {
    const float wlen_nm = atm_cie_sample_wlen_nm(atm, ch, r0, r1);
    *pdf_m = atm->cie_rcp_integral[ch] * 1.0e9f;
    atm->wavelength_m = wlen_nm * 1.0e-9f;
    atm->sun.radiance = atm_planck_monochromatic(atm->wavelength_m, 5778.f);
    atm->path_ks_scale = 1.f;
    atm->path_ka_scale = 1.f;
    if (atm->mie_g_nm && atm->mie_g_n > 1 && atm->mie_g_dlambda_nm > 0.f) {
        float x = (wlen_nm - atm->mie_g_lambda0_nm) / atm->mie_g_dlambda_nm;
        if (x < 0.f) {
            x = 0.f;
        }
        const float xmax = (float)(atm->mie_g_n - 1);
        if (x > xmax) {
            x = xmax;
        }
        const int i0 = (int)x;
        const int i1 = i0 + 1 < atm->mie_g_n ? i0 + 1 : i0;
        const float t = x - (float)i0;
        atm->mie_g = atm->mie_g_nm[i0] * (1.f - t) + atm->mie_g_nm[i1] * t;
    }
    const int ib = atm_sw_find_band(atm, wlen_nm);
    if (ib < 0 || !atm->sw_gas_ka || !atm->sw_gas_ks || atm->gas.nlayers <= 0) {
        return;
    }
    int nq = 1;
    if (atm->sw_nquad && ib < atm->sw_nbands) {
        nq = atm->sw_nquad[ib];
    }
    if (nq < 1) {
        nq = 1;
    }
    const int nqmax = atm->sw_nquad_max > 0 ? atm->sw_nquad_max : nq;
    int iq = 0;
    if (atm->sw_quad_cdf && nq > 1) {
        iq = atm_sample_quad(atm->sw_quad_cdf + ib * nqmax, nq, r_quad);
    }
    const int nlayers = atm->gas.nlayers;
    atm->gas.ka = atm->sw_gas_ka + ((ib * nqmax + iq) * nlayers);
    atm->gas.ks = atm->sw_gas_ks + ((ib * nqmax + iq) * nlayers);
    if (atm->mie_cs_band && atm->mie_cs_ref > 0.f) {
        atm->path_ks_scale = atm->mie_cs_band[ib] / atm->mie_cs_ref;
    }
    if (atm->mie_ca_band && atm->mie_ca_ref > 0.f) {
        atm->path_ka_scale = atm->mie_ca_band[ib] / atm->mie_ca_ref;
    }
}

/* mrumtl_fetch_brdf: pick surrounding knot by alpha (same expectation as lerp). */
ATM_FN float atm_mrumtl_fetch_R(const struct AtmParams* atm, float wlen_nm, float u) {
    if (!atm || atm->brdf_n <= 0 || !atm->brdf_lambda_nm || !atm->brdf_R) {
        return -1.f;
    }
    const int n = atm->brdf_n;
    if (wlen_nm <= atm->brdf_lambda_nm[0]) {
        return atm->brdf_R[0];
    }
    if (wlen_nm >= atm->brdf_lambda_nm[n - 1]) {
        return atm->brdf_R[n - 1];
    }
    int i = 0;
    while (i + 1 < n && atm->brdf_lambda_nm[i + 1] < wlen_nm) {
        ++i;
    }
    const float l0 = atm->brdf_lambda_nm[i];
    const float l1 = atm->brdf_lambda_nm[i + 1];
    const float den = l1 - l0;
    const float alpha = den > 1e-12f ? (l1 - wlen_nm) / den : 1.f;
    /* htrdr: sample < alpha → lambda0, else lambda1 */
    return u < alpha ? atm->brdf_R[i] : atm->brdf_R[i + 1];
}
