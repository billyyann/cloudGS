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

ATM_FN void atm_world_to_ijk(const struct AtmVoxelGrid* g, const float p[3], int ijk[3]) {
    ijk[0] = (int)floorf((p[0] - g->origin[0]) / g->dx);
    ijk[1] = (int)floorf((p[1] - g->origin[1]) / g->dy);
    ijk[2] = (int)floorf((p[2] - g->origin[2]) / g->dz);
}

ATM_FN float atm_fetch_ka(const struct AtmVoxelGrid* g, int ix, int iy, int iz) {
    if (!g->ka || !atm_in_grid(g, ix, iy, iz)) {
        return 0.f;
    }
    return g->ka[atm_voxel_index(g, ix, iy, iz)];
}

ATM_FN float atm_fetch_ks(const struct AtmVoxelGrid* g, int ix, int iy, int iz) {
    if (!g->ks || !atm_in_grid(g, ix, iy, iz)) {
        return 0.f;
    }
    return g->ks[atm_voxel_index(g, ix, iy, iz)];
}

ATM_FN float atm_fetch_temp(const struct AtmVoxelGrid* g, int ix, int iy, int iz) {
    if (!g->temperature || !atm_in_grid(g, ix, iy, iz)) {
        return 0.f;
    }
    return g->temperature[atm_voxel_index(g, ix, iy, iz)];
}

ATM_FN float atm_brick_kmax(const struct AtmVoxelGrid* g, int bx, int by, int bz) {
    if (!g->kext_max) {
        return 0.f;
    }
    if (bx < 0 || by < 0 || bz < 0 || bx >= g->nbx || by >= g->nby || bz >= g->nbz) {
        return 0.f;
    }
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

/* Voxel Woodcock in [0, tmax]. htrdr scattering_hit_filter: Ts ~ Exp(1),
 * brick ks_max, accept ks/ks_max (walk_ks=1). LW may walk kext (walk_ks=0). */
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
    atm_hit_clear(&hit);
    if (!g || g->nx <= 0 || tmax <= 1e-8f) {
        return hit;
    }

    const float gx0 = g->origin[0];
    const float gy0 = g->origin[1];
    const float gz0 = g->origin[2];
    const float gx1 = gx0 + g->dx * (float)g->nx;
    const float gy1 = gy0 + g->dy * (float)g->ny;
    const float gz1 = gz0 + g->dz * (float)g->nz;

    float t0 = 0.f;
    float t1 = tmax;
    /* Ray-AABB of the whole grid */
    for (int a = 0; a < 3; ++a) {
        const float o = org[a];
        const float d = dir[a];
        const float lo = a == 0 ? gx0 : (a == 1 ? gy0 : gz0);
        const float hi = a == 0 ? gx1 : (a == 1 ? gy1 : gz1);
        if (fabsf(d) < 1e-20f) {
            if (o < lo || o > hi) {
                return hit;
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
        return hit;
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
        int ijk[3];
        atm_world_to_ijk(g, p, ijk);
        atm_snap_ijk_enter(g, dir, ijk);
        if (!atm_in_grid(g, ijk[0], ijk[1], ijk[2])) {
            break;
        }
        const int bx = ijk[0] / b;
        const int by = ijk[1] / b;
        const int bz = ijk[2] / b;
        const float kmax = atm_brick_kmax(g, bx, by, bz);

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
        const float ka = atm_fetch_ka(g, cij[0], cij[1], cij[2]);
        const float ks = atm_fetch_ks(g, cij[0], cij[1], cij[2]);
        const float kext = ka + ks;
        const float r = rng.next_1d();
        /* htrdr SW: accept ks/ks_max (scatter only). LW: accept kext then ω. */
        const float k_acc = walk_ks ? ks : kext;
        if (r * kmax < k_acc) {
            hit.t = t;
            hit.pos[0] = pos[0];
            hit.pos[1] = pos[1];
            hit.pos[2] = pos[2];
            hit.ka = ka;
            hit.ks = ks;
            hit.kext = kext;
            hit.temperature = atm_fetch_temp(g, cij[0], cij[1], cij[2]);
            if (walk_ks) {
                hit.event = ATM_EVENT_SCATTER_VOXEL;
            } else {
                const float omega_voxel = kext > 1e-20f ? ks / kext : 0.f;
                const float omega = mie_omega >= 0.f ? mie_omega : omega_voxel;
                hit.event = (rng.next_1d() < omega) ? ATM_EVENT_SCATTER_VOXEL : ATM_EVENT_ABSORB_VOXEL;
            }
            return hit;
        }
        /* Null collision */
        Ts = atm_ran_exp(rng.next_1d());
    }
    return hit;
}

/* 1D layered gas: SW walks ks (scatter only); LW walks kext then ka/ks split. */
template <typename Rng>
ATM_FN struct AtmHit atm_gas_free_path(
    const struct AtmGasLayers* gas,
    const float org[3],
    const float dir[3],
    float tmax,
    Rng& rng,
    int walk_ks = 1) {
    struct AtmHit hit;
    atm_hit_clear(&hit);
    if (!gas || gas->nlayers <= 0 || !gas->z_iface || tmax <= 1e-8f) {
        return hit;
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
            t += t_exit_layer + 1e-5f;
            z = org[2] + dir[2] * t;
            continue;
        }
        const float Tau = t_exit_layer * k_walk;
        if (Ts > Tau) {
            Ts -= Tau;
            t += t_exit_layer + 1e-5f;
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
            hit.event = ATM_EVENT_SCATTER_GAS;
        } else {
            const float r = rng.next_1d();
            hit.event = (r < ka / kext) ? ATM_EVENT_ABSORB_GAS : ATM_EVENT_SCATTER_GAS;
        }
        hit.t = t;
        hit.pos[0] = pos[0];
        hit.pos[1] = pos[1];
        hit.pos[2] = pos[2];
        hit.ka = ka;
        hit.ks = ks;
        hit.kext = kext;
        hit.temperature = gas->temperature ? gas->temperature[layer] : 0.f;
        return hit;
    }
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

ATM_FN float atm_mesh_closest(
    const struct AtmMesh* mesh,
    const float org[3],
    const float dir[3],
    float tmax) {
    float best = tmax;
    if (!mesh || mesh->ntri <= 0) {
        return ATM_INF;
    }
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
    hit.t = best;
    hit.tri = best_i;
    const int i0 = mesh->indices[best_i * 3 + 0];
    const int i1 = mesh->indices[best_i * 3 + 1];
    const int i2 = mesh->indices[best_i * 3 + 2];
    const float* v0 = mesh->vertices + i0 * 3;
    const float* v1 = mesh->vertices + i1 * 3;
    const float* v2 = mesh->vertices + i2 * 3;
    const float e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    const float e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    hit.n[0] = e1[1] * e2[2] - e1[2] * e2[1];
    hit.n[1] = e1[2] * e2[0] - e1[0] * e2[2];
    hit.n[2] = e1[0] * e2[1] - e1[1] * e2[0];
    atm_normalize3(hit.n);
    if (atm_dot3(hit.n, dir) > 0.f) {
        hit.n[0] = -hit.n[0];
        hit.n[1] = -hit.n[1];
        hit.n[2] = -hit.n[2];
    }
    if (mesh->albedo) {
        hit.albedo[0] = mesh->albedo[best_i * 3 + 0];
        hit.albedo[1] = mesh->albedo[best_i * 3 + 1];
        hit.albedo[2] = mesh->albedo[best_i * 3 + 2];
    }
    if (mesh->temperature) {
        hit.temperature = mesh->temperature[best_i];
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

/* Competing risks: surface tmax truncates volume. Volume never reads Gaussian k. */
template <typename Rng>
ATM_FN struct AtmHit atm_next_volume_event(
    const struct AtmParams* atm,
    const float org[3],
    const float dir[3],
    float t_surface,
    Rng& rng) {
    const float tmax = t_surface < atm->path_tmax ? t_surface : atm->path_tmax;
    const int walk_ks = atm->spectral_mode != ATM_SPECTRAL_LW;
    const float omega_arg = atm->mie_omega > 0.f ? atm->mie_omega : -1.f;
    struct AtmHit gas = atm_gas_free_path(&atm->gas, org, dir, tmax, rng, walk_ks);
    struct AtmHit vox = atm_woodcock_voxels(&atm->voxels, org, dir, tmax, rng, omega_arg, walk_ks);
    if (gas.event != ATM_EVENT_NONE && gas.t <= vox.t && gas.t < tmax) {
        return gas;
    }
    if (vox.event != ATM_EVENT_NONE && vox.t < tmax) {
        return vox;
    }
    struct AtmHit none;
    atm_hit_clear(&none);
    none.t = tmax;
    none.event = ATM_EVENT_NONE;
    return none;
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
    int kind = ATM_K_EXT) {
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
    /* NEE analog τ: clip Z only (htrdr -r). XY is periodic; a finite XY AABB
     * lets sun rays from the bottom/left exit before the opposite-tile cloud.
     * Camera Woodcock stays finite-XY. */
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
        tau += atm_cell_k(g, fx, fy, ijk[2], kind) * fmaxf(tNext - t, 0.f);
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
        t += ds + 1e-5f;
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
    return atm_voxel_optical_depth(&atm->voxels, org, dir, tmax, kind) +
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
    struct AtmHit h = atm_next_volume_event(atm, org, dir, tmax, rng);
    return h.event != ATM_EVENT_NONE && h.t < tmax;
}
