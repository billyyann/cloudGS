/* Convert htrdr mountain.obj + materials → 3DGS, render like mountain.sh
 * (cyclic ground -R, cyclic clouds -r, sand lambertian, SW sun).
 */

#include <3dgptir/kernels/cuda/atmosphere.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Vec3 {
    float x, y, z;
};

static Vec3 vadd(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vec3 vsub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static Vec3 vmul(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static float vdot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3 vcross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static float vlen(Vec3 a) { return std::sqrt(vdot(a, a)); }
static Vec3 vnorm(Vec3 a) {
    const float n = vlen(a);
    return n > 1e-12f ? vmul(a, 1.f / n) : Vec3{0, 0, 1};
}

static float wrap01(float x, float period) {
    x = std::fmod(x, period);
    if (x < 0.f) {
        x += period;
    }
    return x;
}

struct GsParticle {
    Vec3 pos, n, scale;
    float quat[4];
    float albedo[3];
    float roughness, metallic, temperature, opacity;
};

struct HostRng {
    unsigned long long state, inc;
    void init(unsigned long long seed, unsigned long long seq = 1) {
        state = 0;
        inc = (seq << 1u) | 1ull;
        next_u32();
        state += seed;
        next_u32();
    }
    unsigned next_u32() {
        const unsigned long long old = state;
        state = old * 6364136223846793005ull + inc;
        unsigned x = static_cast<unsigned>(((old >> 18) ^ old) >> 27);
        const unsigned r = static_cast<unsigned>(old >> 59);
        return (x >> r) | (x << ((-static_cast<int>(r)) & 31));
    }
    float next_1d() { return (next_u32() & 0x00FFFFFFu) * (1.0f / 16777216.0f); }
};

static void quat_from_normal(const Vec3& n, float q[4]) {
    const float c = n.z;
    float ax = -n.y, ay = n.x, az = 0.f;
    if (ax * ax + ay * ay < 1e-12f) {
        q[0] = c >= 0.f ? 1.f : 0.f;
        q[1] = c >= 0.f ? 0.f : 1.f;
        q[2] = 0.f;
        q[3] = 0.f;
        return;
    }
    q[0] = 1.f + c;
    q[1] = ax;
    q[2] = ay;
    q[3] = az;
    const float qn = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    q[0] /= qn;
    q[1] /= qn;
    q[2] /= qn;
    q[3] /= qn;
}

/* mountain.mtls: sand lambertian @ 300 K, ~0.16 at 600 nm. Slope only darkens. */
static void sand_albedo(float slope_deg, float a[3]) {
    const float s = atm_clampf(slope_deg / 55.f, 0.f, 1.f);
    a[0] = 0.22f * (1.f - 0.22f * s);
    a[1] = 0.16f * (1.f - 0.18f * s);
    a[2] = 0.10f * (1.f - 0.12f * s);
}

static int load_mountain_grid(const char* path, int& n, std::vector<Vec3>& v) {
    FILE* f = std::fopen(path, "r");
    if (!f) {
        std::perror(path);
        return 1;
    }
    char line[256];
    v.clear();
    v.reserve(4198401);
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            Vec3 p;
            if (std::sscanf(line + 2, "%f %f %f", &p.x, &p.y, &p.z) == 3) {
                v.push_back(p);
            }
        } else if (std::strncmp(line, "usemtl", 6) == 0) {
            break;
        }
    }
    std::fclose(f);
    const int nv = static_cast<int>(v.size());
    const int side = static_cast<int>(std::sqrt(static_cast<double>(nv)) + 0.5);
    if (side * side != nv) {
        std::fprintf(stderr, "vertex count %d is not a square grid\n", nv);
        return 1;
    }
    n = side;
    std::printf("loaded %d x %d heightfield from %s\n", n, n, path);
    return 0;
}

static std::vector<GsParticle> downsample_to_gs(const std::vector<Vec3>& src, int nsrc, int step) {
    const int nd = (nsrc - 1) / step + 1;
    std::vector<GsParticle> gs;
    gs.reserve(static_cast<size_t>(nd) * nd);
    const float dx = 10000.f / (nd - 1);
    for (int j = 0; j < nd; ++j) {
        for (int i = 0; i < nd; ++i) {
            const int i0 = std::min(i * step, nsrc - 1);
            const int j0 = std::min(j * step, nsrc - 1);
            const Vec3 p = src[static_cast<size_t>(j0) * nsrc + i0];
            const int ip = std::min(i0 + step, nsrc - 1);
            const int im = std::max(i0 - step, 0);
            const int jp = std::min(j0 + step, nsrc - 1);
            const int jm = std::max(j0 - step, 0);
            const Vec3 tng = vsub(src[static_cast<size_t>(j0) * nsrc + ip], src[static_cast<size_t>(j0) * nsrc + im]);
            const Vec3 btg = vsub(src[static_cast<size_t>(jp) * nsrc + i0], src[static_cast<size_t>(jm) * nsrc + i0]);
            Vec3 n = vnorm(vcross(tng, btg));
            if (n.z < 0.f) {
                n = vmul(n, -1.f);
            }
            GsParticle g{};
            g.pos = p;
            g.n = n;
            const float slope = std::acos(atm_clampf(n.z, 0.f, 1.f)) * 180.f / static_cast<float>(M_PI);
            sand_albedo(slope, g.albedo);
            g.roughness = 1.f;
            g.metallic = 0.f;
            g.temperature = 300.f;
            g.scale = {dx * 0.55f, dx * 0.55f, dx * 0.18f};
            quat_from_normal(n, g.quat);
            g.opacity = 0.99f;
            gs.push_back(g);
        }
    }
    std::printf("3DGS particles: %zu (grid %d x %d, spacing %.1f m)\n", gs.size(), nd, nd, dx);
    return gs;
}

static int write_ply(const char* path, const std::vector<GsParticle>& gs) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        std::perror(path);
        return 1;
    }
    std::fprintf(f, "ply\nformat ascii 1.0\ncomment cyclic GS mountain from mountain.obj usemtl air:sand\n");
    std::fprintf(f, "element vertex %zu\n", gs.size());
    std::fprintf(f, "property float x\nproperty float y\nproperty float z\n");
    std::fprintf(f, "property float nx\nproperty float ny\nproperty float nz\n");
    std::fprintf(f, "property float scale_0\nproperty float scale_1\nproperty float scale_2\n");
    std::fprintf(f, "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n");
    std::fprintf(f, "property float albedo_r\nproperty float albedo_g\nproperty float albedo_b\n");
    std::fprintf(f, "property float roughness\nproperty float metallic\nproperty float temperature\n");
    std::fprintf(f, "property float opacity\nend_header\n");
    for (const auto& g : gs) {
        std::fprintf(f,
                     "%.4f %.4f %.4f %.5f %.5f %.5f %.4f %.4f %.4f %.5f %.5f %.5f %.5f %.4f %.4f %.4f %.4f %.4f %.2f %.4f\n",
                     g.pos.x, g.pos.y, g.pos.z, g.n.x, g.n.y, g.n.z, g.scale.x, g.scale.y, g.scale.z,
                     g.quat[0], g.quat[1], g.quat[2], g.quat[3], g.albedo[0], g.albedo[1], g.albedo[2],
                     g.roughness, g.metallic, g.temperature, g.opacity);
    }
    std::fclose(f);
    std::printf("wrote %s\n", path);
    return 0;
}

struct HeightGrid {
    int n;       /* vertices along one edge (2049) */
    int ncell;   /* n-1, cyclic period in cells */
    float period;
    float dx;
    float zmax;
    std::vector<float> z;
    std::vector<std::vector<float>> mip; /* cell-max pyramids, power-of-two ncell */
    int wrap_i(int i) const {
        int m = ncell;
        i %= m;
        if (i < 0) {
            i += m;
        }
        return i;
    }
    float hz(int i, int j) const { return z[static_cast<size_t>(wrap_i(j)) * n + wrap_i(i)]; }
};

static HeightGrid make_cyclic_grid(const std::vector<Vec3>& src, int nsrc) {
    HeightGrid g;
    g.n = nsrc;
    g.ncell = nsrc - 1;
    g.period = 10000.f;
    g.dx = g.period / (float)g.ncell;
    g.z.resize(static_cast<size_t>(nsrc) * nsrc);
    g.zmax = 0.f;
    for (int k = 0; k < nsrc * nsrc; ++k) {
        g.z[k] = src[k].z;
        if (g.z[k] > g.zmax) {
            g.zmax = g.z[k];
        }
    }
    int w = g.ncell;
    std::vector<float> L0(static_cast<size_t>(w) * w);
    for (int j = 0; j < w; ++j) {
        for (int i = 0; i < w; ++i) {
            float m = g.hz(i, j);
            m = std::max(m, g.hz(i + 1, j));
            m = std::max(m, g.hz(i, j + 1));
            m = std::max(m, g.hz(i + 1, j + 1));
            L0[static_cast<size_t>(j) * w + i] = m;
        }
    }
    g.mip.push_back(std::move(L0));
    int pw = w;
    while (pw > 4) {
        const int nw = pw / 2;
        const auto& prev = g.mip.back();
        std::vector<float> nxt(static_cast<size_t>(nw) * nw);
        for (int j = 0; j < nw; ++j) {
            for (int i = 0; i < nw; ++i) {
                float m = prev[static_cast<size_t>(2 * j) * pw + 2 * i];
                m = std::max(m, prev[static_cast<size_t>(2 * j) * pw + 2 * i + 1]);
                m = std::max(m, prev[static_cast<size_t>(2 * j + 1) * pw + 2 * i]);
                m = std::max(m, prev[static_cast<size_t>(2 * j + 1) * pw + 2 * i + 1]);
                nxt[static_cast<size_t>(j) * nw + i] = m;
            }
        }
        g.mip.push_back(std::move(nxt));
        pw = nw;
    }
    return g;
}

static float bilinear_h(const HeightGrid& g, float x, float y) {
    x = wrap01(x, g.period);
    y = wrap01(y, g.period);
    const float gx = x / g.dx;
    const float gy = y / g.dx;
    int i = (int)std::floor(gx);
    int j = (int)std::floor(gy);
    const float u = gx - i;
    const float v = gy - j;
    const float h00 = g.hz(i, j);
    const float h10 = g.hz(i + 1, j);
    const float h01 = g.hz(i, j + 1);
    const float h11 = g.hz(i + 1, j + 1);
    return (h00 * (1.f - u) + h10 * u) * (1.f - v) + (h01 * (1.f - u) + h11 * u) * v;
}

static Vec3 bilinear_n(const HeightGrid& g, float x, float y) {
    const float e = g.dx;
    const float hx = bilinear_h(g, x + e, y) - bilinear_h(g, x - e, y);
    const float hy = bilinear_h(g, x, y + e) - bilinear_h(g, x, y - e);
    Vec3 n = vnorm(Vec3{-hx / (2.f * e), -hy / (2.f * e), 1.f});
    return n;
}

/* Continuous cyclic heightfield (htrdr -R). Sphere-trace vs bilinear height. */
static bool intersect_ground(const HeightGrid& g, const Vec3& o, const Vec3& d, float tmin, float tmax, float& tHit, Vec3& nHit, float alb[3]) {
    if (o.z > g.zmax + 2.f && d.z >= -1e-5f) {
        return false;
    }
    float t = std::max(tmin, 0.05f);
    bool hit = false;
    float t_cross = tmax;
    const int nlevels = static_cast<int>(g.mip.size());
    for (int s = 0; s < 4500 && t < tmax; ++s) {
        const Vec3 p = vadd(o, vmul(d, t));
        if (p.z > g.zmax + 8.f && d.z >= 0.f) {
            return false;
        }
        const float h = bilinear_h(g, p.x, p.y);
        const float above = p.z - h;
        if (above <= 0.25f) {
            hit = true;
            t_cross = t;
            break;
        }
        int lod = 0;
        while (lod + 1 < nlevels && g.dx * (float)(1 << (lod + 1)) < above * 0.85f) {
            ++lod;
        }
        const int w = g.ncell >> lod;
        const float cell = g.dx * (float)(1 << lod);
        const float wx = wrap01(p.x, g.period);
        const float wy = wrap01(p.y, g.period);
        int ix = (int)std::floor(wx / cell);
        int iy = (int)std::floor(wy / cell);
        ix %= w;
        iy %= w;
        if (ix < 0) {
            ix += w;
        }
        if (iy < 0) {
            iy += w;
        }
        const float hmax = g.mip[lod][static_cast<size_t>(iy) * w + ix];
        const float above_max = p.z - hmax;
        float dt = std::max(cell * 0.65f, std::max(above * 0.55f, above_max * 0.7f));
        dt = std::max(dt, t * 0.006f);
        if (d.z < -1e-4f && above_max > 0.f) {
            dt = std::min(dt, above_max / (-d.z) * 0.92f);
        }
        if (above < 12.f) {
            dt = std::min(dt, 2.2f);
        }
        dt = std::min(std::max(dt, 0.35f), 140.f);
        t += dt;
    }
    if (!hit) {
        return false;
    }
    float lo = std::max(0.f, t_cross - 80.f);
    float hi = t_cross;
    for (int i = 0; i < 18; ++i) {
        const float mid = 0.5f * (lo + hi);
        const Vec3 p = vadd(o, vmul(d, mid));
        if (p.z <= bilinear_h(g, p.x, p.y) + 0.05f) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    tHit = hi;
    const Vec3 p = vadd(o, vmul(d, tHit));
    nHit = bilinear_n(g, p.x, p.y);
    if (vdot(nHit, d) > 0.f) {
        nHit = vmul(nHit, -1.f);
    }
    const float slope = std::acos(atm_clampf(nHit.z, 0.f, 1.f)) * 180.f / static_cast<float>(M_PI);
    sand_albedo(slope, alb);
    return true;
}

static float hash31(int x, int y, int z) {
    unsigned h = (unsigned)(x * 374761393 + y * 668265263 + z * 1597334677u);
    h = (h ^ (h >> 13)) * 1274126177u;
    return (h & 0x00FFFFFFu) * (1.f / 16777216.f);
}
static float vnoise(float x, float y, float z) {
    int ix = (int)std::floor(x), iy = (int)std::floor(y), iz = (int)std::floor(z);
    float fx = x - ix, fy = y - iy, fz = z - iz;
    fx = fx * fx * (3.f - 2.f * fx);
    fy = fy * fy * (3.f - 2.f * fy);
    fz = fz * fz * (3.f - 2.f * fz);
    auto at = [&](int dx, int dy, int dz) { return hash31(ix + dx, iy + dy, iz + dz); };
    float n000 = at(0, 0, 0), n100 = at(1, 0, 0), n010 = at(0, 1, 0), n110 = at(1, 1, 0);
    float n001 = at(0, 0, 1), n101 = at(1, 0, 1), n011 = at(0, 1, 1), n111 = at(1, 1, 1);
    float x00 = n000 + (n100 - n000) * fx;
    float x10 = n010 + (n110 - n010) * fx;
    float x01 = n001 + (n101 - n001) * fx;
    float x11 = n011 + (n111 - n011) * fx;
    float y0 = x00 + (x10 - x00) * fy;
    float y1 = x01 + (x11 - x01) * fy;
    return y0 + (y1 - y0) * fz;
}
static float fbm(float x, float y, float z) {
    float a = 0.f, w = 0.5f, s = 0.f;
    for (int i = 0; i < 5; ++i) {
        a += w * vnoise(x, y, z);
        s += w;
        w *= 0.5f;
        x *= 2.03f;
        y *= 2.03f;
        z *= 2.03f;
    }
    return a / s;
}

static float cell_mask(float u, float v, float h) {
    const float cells = 5.8f;
    const float x = u * cells + 0.35f * (fbm(u * 3.f, v * 3.f, h * 2.f) - 0.5f);
    const float y = v * cells + 0.35f * (fbm(u * 3.f + 2.1f, v * 3.f, h * 2.f) - 0.5f);
    const int ix = (int)std::floor(x);
    const int iy = (int)std::floor(y);
    float keep = 0.f;
    for (int dj = -1; dj <= 1; ++dj) {
        for (int di = -1; di <= 1; ++di) {
            const int cx = ix + di;
            const int cy = iy + dj;
            if (hash31(cx, cy, 19) > 0.30f) {
                continue;
            }
            const float px = (float)cx + 0.18f + 0.64f * hash31(cx, cy, 1);
            const float py = (float)cy + 0.18f + 0.64f * hash31(cx, cy, 2);
            const float pz = 0.28f + 0.40f * hash31(cx, cy, 7);
            const float rad = 0.32f + 0.38f * hash31(cx, cy, 3);
            const float radz = 0.22f + 0.28f * hash31(cx, cy, 11);
            const float dx = (x - px) / std::max(rad, 1e-3f);
            const float dy = (y - py) / std::max(rad, 1e-3f);
            const float dz = (h - pz) / std::max(radz, 1e-3f);
            const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
            const float w = std::max(0.f, 1.f - d);
            keep = std::max(keep, w * w * (1.f - 0.25f * d));
        }
    }
    return keep;
}

/* Broken Cu, tiled 6.4 km (DZVAR ~25% cover, base ~1 km), AABB covers the camera frustum. */
static void fill_cumulus(AtmVoxelGrid& grid, std::vector<float>& ka, std::vector<float>& ks, std::vector<float>& temp, std::vector<float>& kmax) {
    grid.nx = 280;
    grid.ny = 168;
    grid.nz = 48;
    grid.brick = 8;
    grid.origin[0] = -45000.f;
    grid.origin[1] = -15000.f;
    grid.origin[2] = 850.f;
    const float sx = 70000.f, sy = 42000.f, sz = 1400.f;
    grid.dx = sx / grid.nx;
    grid.dy = sy / grid.ny;
    grid.dz = sz / grid.nz;
    const int n = grid.nx * grid.ny * grid.nz;
    ka.assign(n, 0.f);
    ks.assign(n, 0.f);
    temp.assign(n, 278.f);
    const float tile = 6400.f;
    for (int iz = 0; iz < grid.nz; ++iz) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int ix = 0; ix < grid.nx; ++ix) {
                const float x = grid.origin[0] + (ix + 0.5f) * grid.dx;
                const float y = grid.origin[1] + (iy + 0.5f) * grid.dy;
                const float z = grid.origin[2] + (iz + 0.5f) * grid.dz;
                const float u = wrap01(x, tile) / tile;
                const float v = wrap01(y, tile) / tile;
                const float h = (z - 850.f) / 1400.f;
                const float mask = cell_mask(u, v, h);
                const float detail = fbm(u * 9.0f, v * 9.0f, h * 6.0f);
                float dens = mask * std::max(0.f, 0.65f + 0.55f * (detail - 0.5f));
                dens *= 0.10f;
                /* Fade at volume AABB so the grid box is not visible. */
                const float mx = std::min(x - grid.origin[0], grid.origin[0] + sx - x) / grid.dx;
                const float my = std::min(y - grid.origin[1], grid.origin[1] + sy - y) / grid.dy;
                const float fade = atm_clampf(std::min(mx, my) / 5.f, 0.f, 1.f);
                dens *= fade;
                const int id = (iz * grid.ny + iy) * grid.nx + ix;
                ks[id] = dens;
                ka[id] = dens * 0.001f;
            }
        }
    }
    grid.ka = ka.data();
    grid.ks = ks.data();
    grid.temperature = temp.data();
    kmax.resize(((grid.nx + 7) / 8) * ((grid.ny + 7) / 8) * ((grid.nz + 7) / 8));
    atm_build_brick_majorants(&grid, kmax.data());
    int ncloudy = 0;
    float ksum = 0.f, kpeak = 0.f;
    for (int i = 0; i < n; ++i) {
        if (ks[i] > 1e-5f) {
            ++ncloudy;
            ksum += ks[i];
            kpeak = std::max(kpeak, ks[i]);
        }
    }
    std::printf("cumulus: %.1f%% cloudy voxels, mean ks=%.4f peak=%.4f\n",
                100.f * ncloudy / (float)n, ncloudy ? ksum / ncloudy : 0.f, kpeak);
}

static Vec3 sky_color(const Vec3& dir, const Vec3& sun) {
    const float muz = std::max(dir.z, 0.f);
    Vec3 zen{0.12f, 0.38f, 0.88f};
    Vec3 hor{0.62f, 0.74f, 0.90f};
    Vec3 col = vadd(vmul(hor, 1.f - muz), vmul(zen, muz));
    const float mu = std::max(0.f, vdot(dir, sun));
    col = vadd(col, vmul(Vec3{1.0f, 0.88f, 0.55f}, std::pow(mu, 12.f) * 0.85f));
    if (mu > 0.9995f) {
        col = vadd(col, Vec3{18.f, 16.f, 12.f});
    }
    return col;
}

struct Scene {
    HeightGrid ground;
    AtmParams atm;
};

static Vec3 haze_mix(Vec3 L, const Vec3& sky, float dist) {
    const float beta = 1.f / 11000.f;
    const float t = std::exp(-beta * dist);
    return vadd(vmul(L, t), vmul(sky, 1.f - t));
}

static Vec3 trace_path(const Scene& sc, Vec3 o, Vec3 d, HostRng& rng) {
    d = vnorm(d);
    Vec3 thr{1, 1, 1};
    Vec3 L{0, 0, 0};
    const Vec3 sun{sc.atm.sun.dir[0], sc.atm.sun.dir[1], sc.atm.sun.dir[2]};
    const float Lsun = sc.atm.sun.radiance;
    const Vec3 sky0 = sky_color(d, sun);

    for (int bounce = 0; bounce < 4; ++bounce) {
        float org[3] = {o.x, o.y, o.z};
        float dir[3] = {d.x, d.y, d.z};
        float tG = ATM_INF;
        Vec3 nG{0, 0, 1};
        float alb[3] = {0.22f, 0.16f, 0.10f};
        const bool ghit = intersect_ground(sc.ground, o, d, 0.05f, sc.atm.path_tmax, tG, nG, alb);
        const float tSurf = ghit ? tG : sc.atm.path_tmax;
        const AtmHit vol = atm_next_volume_event(&sc.atm, org, dir, tSurf, rng);

        if (vol.event == ATM_EVENT_ABSORB_VOXEL || vol.event == ATM_EVENT_ABSORB_GAS) {
            break;
        }
        if (vol.event == ATM_EVENT_SCATTER_VOXEL || vol.event == ATM_EVENT_SCATTER_GAS) {
            const int voxel = vol.event == ATM_EVENT_SCATTER_VOXEL;
            float wo[3], pdf = 1.f;
            atm_sample_sun(&sc.atm.sun, rng, wo, &pdf);
            float p[3] = {vol.pos[0] + wo[0] * 0.4f, vol.pos[1] + wo[1] * 0.4f, vol.pos[2] + wo[2] * 0.4f};
            float tgs = ATM_INF;
            Vec3 ng;
            float dummy[3];
            const bool blocked = intersect_ground(sc.ground, Vec3{p[0], p[1], p[2]}, Vec3{wo[0], wo[1], wo[2]}, 4.f, sc.atm.path_tmax, tgs, ng, dummy);
            int occ = blocked ? 1 : atm_volume_occluded(&sc.atm, p, wo, sc.atm.path_tmax, rng);
            const float mu = dir[0] * wo[0] + dir[1] * wo[1] + dir[2] * wo[2];
            const float phase = voxel ? atm_hg_phase(sc.atm.mie_g, mu) : atm_rayleigh_phase(mu);
            if (!occ) {
                L = vadd(L, vmul(thr, phase * Lsun / std::max(pdf, 1e-6f)));
            }
            /* Weak sky fill; sun NEE + HG does the white Cu cores. */
            L = vadd(L, vmul(thr, voxel ? 0.12f : 0.04f));
            float scatter[3];
            if (voxel) {
                atm_sample_hg(sc.atm.mie_g, rng.next_1d(), rng.next_1d(), dir, scatter);
            } else {
                atm_sample_rayleigh(rng, dir, scatter);
            }
            o = Vec3{vol.pos[0], vol.pos[1], vol.pos[2]};
            d = Vec3{scatter[0], scatter[1], scatter[2]};
            continue;
        }

        if (ghit) {
            if (vdot(nG, d) > 0.f) {
                nG = vmul(nG, -1.f);
            }
            float wo[3], pdf = 1.f;
            atm_sample_sun(&sc.atm.sun, rng, wo, &pdf);
            const Vec3 wi{wo[0], wo[1], wo[2]};
            const float ndl = std::max(0.f, vdot(nG, wi));
            Vec3 p = vadd(vadd(o, vmul(d, tG)), vmul(nG, 2.5f));
            float po[3] = {p.x, p.y, p.z};
            float tgs = ATM_INF;
            Vec3 ng;
            float dummy[3];
            const bool gblock = intersect_ground(sc.ground, p, wi, 3.f, sc.atm.path_tmax, tgs, ng, dummy);
            int vocc = gblock ? 1 : atm_volume_occluded(&sc.atm, po, wo, sc.atm.path_tmax, rng);
            const float fr = 1.f / static_cast<float>(M_PI);
            Vec3 Lo{0, 0, 0};
            if (!vocc && ndl > 0.f) {
                const float w = fr * ndl * Lsun / std::max(pdf, 1e-6f);
                Lo = Vec3{alb[0] * w, alb[1] * w, alb[2] * w};
            }
            /* Hemispheric sky fill (original htrdr has this via path; we approximate). */
            const float skyfill = 0.42f * (0.50f + 0.50f * std::max(nG.z, 0.f));
            Lo = vadd(Lo, Vec3{alb[0] * skyfill, alb[1] * skyfill, alb[2] * skyfill});
            Lo = Vec3{thr.x * Lo.x, thr.y * Lo.y, thr.z * Lo.z};
            L = vadd(L, haze_mix(Lo, sky0, tG));
            break;
        }

        L = vadd(L, Vec3{thr.x * sky0.x, thr.y * sky0.y, thr.z * sky0.z});
        break;
    }
    return L;
}

static int write_ppm(const char* path, int w, int h, const std::vector<Vec3>& img, float exposure) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::perror(path);
        return 1;
    }
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int i = 0; i < w * h; ++i) {
        float r = img[i].x * exposure;
        float g = img[i].y * exposure;
        float b = img[i].z * exposure;
        r = r / (1.f + r);
        g = g / (1.f + g);
        b = b / (1.f + b);
        auto to8 = [](float x) {
            x = std::pow(std::max(0.f, x), 1.f / 2.2f);
            return (unsigned char)std::min(255, (int)(x * 255.f + 0.5f));
        };
        unsigned char px[3] = {to8(r), to8(g), to8(b)};
        std::fwrite(px, 1, 3, f);
    }
    std::fclose(f);
    std::printf("wrote %s\n", path);
    return 0;
}

static int write_scene_json(const char* path, const char* ply, int ngs) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        std::perror(path);
        return 1;
    }
    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"name\": \"mountain_gs\",\n");
    std::fprintf(f, "  \"match\": \"mountain.sh: cyclic ground -R, cyclic clouds -r, sand@300K, same camera/sun\",\n");
    std::fprintf(f, "  \"gaussians\": {\"path\": \"%s\", \"count\": %d},\n", ply, ngs);
    std::fprintf(f, "  \"camera\": {\"pos\": [10245.37, 615.02, 422.34], \"tgt\": [-2977.39, 6829.40, 842.40], \"fov\": 30},\n");
    std::fprintf(f, "  \"sun\": {\"azimuth_deg\": 225, \"elevation_deg\": 30}\n");
    std::fprintf(f, "}\n");
    std::fclose(f);
    return 0;
}

int main(int argc, char** argv) {
    const char* obj = argc > 1 ? argv[1]
                               : "/home/szyj2025/scatterGS/cloudGS/htrdr-atmosphere-spk/htrdr-Atmosphere-Starter-Pack-0.8.0/models/mountain.obj";
    const char* outdir = argc > 2 ? argv[2] : "/home/szyj2025/scatterGS/cloudGS/scenes/mountain_gs";
    const int step = argc > 3 ? std::atoi(argv[3]) : 8;
    const int W = argc > 4 ? std::atoi(argv[4]) : 960;
    const int H = argc > 5 ? std::atoi(argv[5]) : 540;
    const int spp = argc > 6 ? std::atoi(argv[6]) : 32;

    int nsrc = 0;
    std::vector<Vec3> verts;
    if (load_mountain_grid(obj, nsrc, verts)) {
        return 1;
    }
    auto gs = downsample_to_gs(verts, nsrc, step);
    const std::string ply = std::string(outdir) + "/mountain_gs.ply";
    const std::string json = std::string(outdir) + "/scene.json";
    const std::string ppm = std::string(outdir) + "/render.ppm";
    if (write_ply(ply.c_str(), gs)) {
        return 1;
    }
    write_scene_json(json.c_str(), "mountain_gs.ply", (int)gs.size());
    gs.clear();
    gs.shrink_to_fit();

    /* Intersect the full cyclic 2049² field, not the downsampled GS grid. */
    HeightGrid ground = make_cyclic_grid(verts, nsrc);
    verts.clear();
    verts.shrink_to_fit();

    std::vector<float> ka, ks, temp, kmax;
    AtmParams atm{};
    atm.enabled = 1;
    atm.spectral_mode = ATM_SPECTRAL_SW;
    atm.wavelength_m = 5.5e-7f;
    atm.mie_omega = 0.999f;
    atm.mie_g = 0.75f;
    atm.path_tmax = 35000.f;
    atm.max_bounces = 4;
    fill_cumulus(atm.voxels, ka, ks, temp, kmax);

    /* Aerial perspective is haze_mix on surfaces; 1D gas scatter is too noisy at preview spp. */
    atm.gas.nlayers = 0;

    const float az = 225.f * static_cast<float>(M_PI) / 180.f;
    const float el = 30.f * static_cast<float>(M_PI) / 180.f;
    atm.sun.dir[0] = std::cos(el) * std::cos(az);
    atm.sun.dir[1] = std::cos(el) * std::sin(az);
    atm.sun.dir[2] = std::sin(el);
    /* Dirac sun: pdf=1, radiance = SW irradiance scale (not disk L / Ω). */
    atm.sun.radiance = 24.f;
    atm.sun.cos_half_angle = 1.f;

    Scene sc;
    sc.ground = std::move(ground);
    sc.atm = atm;

    const Vec3 cam_pos{10245.37f, 615.02f, 422.34f};
    const Vec3 tgt{-2977.39f, 6829.40f, 842.40f};
    const Vec3 cam_fwd = vnorm(vsub(tgt, cam_pos));
    const Vec3 cam_right = vnorm(vcross(cam_fwd, Vec3{0, 0, 1}));
    const Vec3 cam_up = vnorm(vcross(cam_right, cam_fwd));
    const float tan_half = std::tan(15.f * static_cast<float>(M_PI) / 180.f);

    std::vector<Vec3> img(static_cast<size_t>(W) * H);
    std::printf("rendering %dx%d spp=%d (cyclic ground 2049^2) ...\n", W, H, spp);
#pragma omp parallel for schedule(dynamic, 2)
    for (int y = 0; y < H; ++y) {
        HostRng rng;
        rng.init(0xC0FFEEULL + (unsigned)y * 17u, (unsigned)y + 1u);
        for (int x = 0; x < W; ++x) {
            Vec3 acc{0, 0, 0};
            for (int s = 0; s < spp; ++s) {
                const float u = (x + rng.next_1d()) / (float)W * 2.f - 1.f;
                const float v = (1.f - (y + rng.next_1d()) / (float)H) * 2.f - 1.f;
                const float aspect = (float)W / (float)H;
                Vec3 dir = vadd(cam_fwd, vadd(vmul(cam_right, u * aspect * tan_half), vmul(cam_up, v * tan_half)));
                acc = vadd(acc, trace_path(sc, cam_pos, dir, rng));
            }
            img[static_cast<size_t>(y) * W + x] = vmul(acc, 1.f / (float)spp);
        }
        if ((y % 30) == 0) {
            std::printf("  row %d / %d\n", y, H);
        }
    }
    return write_ppm(ppm.c_str(), W, H, img, 1.25f);
}
