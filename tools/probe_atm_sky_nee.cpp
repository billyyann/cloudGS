/* Host replica of GPU volume event + analog sun NEE (no mesh).
 * Compile via tools/probe_atm_sky_nee.py
 */
#include <3dgptir/kernels/cuda/atmosphere.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct HostRng {
    unsigned long long state;
    unsigned long long inc;
    void init(unsigned long long seed, unsigned long long seq = 1) {
        state = 0;
        inc = (seq << 1u) | 1ull;
        next_u32();
        state += seed;
        next_u32();
    }
    unsigned int next_u32() {
        const unsigned long long oldstate = state;
        state = oldstate * 6364136223846793005ull + inc;
        unsigned int xorshifted = static_cast<unsigned int>(((oldstate >> 18ull) ^ oldstate) >> 27ull);
        const unsigned int rot = static_cast<unsigned int>(oldstate >> 59ull);
        return (xorshifted >> rot) | (xorshifted << ((-static_cast<int>(rot)) & 31));
    }
    float next_1d() {
        const unsigned int u = next_u32() & 0x00FFFFFFu;
        return static_cast<float>(u) * (1.0f / 16777216.0f);
    }
};

static std::vector<char> read_all(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(1);
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf(static_cast<size_t>(n));
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        std::fprintf(stderr, "short read %s\n", path);
        std::exit(1);
    }
    std::fclose(f);
    return buf;
}

static void persp_dir(
    int px,
    int py,
    int w,
    int h,
    const float pos[3],
    const float tgt[3],
    float fov_y_deg,
    float out[3]) {
    float look[3] = {tgt[0] - pos[0], tgt[1] - pos[1], tgt[2] - pos[2]};
    atm_normalize3(look);
    float up[3] = {0.f, 0.f, 1.f};
    const float ud = atm_dot3(up, look);
    up[0] -= look[0] * ud;
    up[1] -= look[1] * ud;
    up[2] -= look[2] * ud;
    atm_normalize3(up);
    float right[3] = {
        look[1] * up[2] - look[2] * up[1],
        look[2] * up[0] - look[0] * up[2],
        look[0] * up[1] - look[1] * up[0]};
    atm_normalize3(right);
    const float half_v = tanf(0.5f * fov_y_deg * ATM_PI / 180.f);
    const float half_u = half_v * (static_cast<float>(w) / static_cast<float>(h));
    const float js = (static_cast<float>(px) + 0.5f) / static_cast<float>(w);
    const float is_ = (static_cast<float>(py) + 0.5f) / static_cast<float>(h);
    const float u = (js - 0.5f) * 2.f * half_u;
    const float v = (0.5f - is_) * 2.f * half_v;
    out[0] = look[0] + right[0] * u + up[0] * v;
    out[1] = look[1] + right[1] * u + up[1] * v;
    out[2] = look[2] + right[2] * u + up[2] * v;
    atm_normalize3(out);
}

static float percentile(std::vector<float>& v, float p) {
    if (v.empty()) {
        return 0.f;
    }
    std::sort(v.begin(), v.end());
    const float t = p * static_cast<float>(v.size() - 1);
    const int i = static_cast<int>(t);
    const int j = std::min(i + 1, static_cast<int>(v.size()) - 1);
    const float a = t - static_cast<float>(i);
    return v[static_cast<size_t>(i)] * (1.f - a) + v[static_cast<size_t>(j)] * a;
}

static void probe_ray(
    const char* name,
    AtmParams* atm,
    const float org[3],
    const float dir[3],
    const float sun[3],
    float tmax,
    int nsamp) {
    const float tau_g_view = atm_gas_optical_depth(&atm->gas, org, dir, tmax, ATM_K_KS);
    const float tau_v_view = atm_voxel_optical_depth(&atm->voxels, org, dir, tmax, ATM_K_KS);
    std::printf(
        "\n== %s ==\n  dir=(%.4f,%.4f,%.4f) elev=%.2f deg\n"
        "  view tau_ks gas=%.4g voxel=%.4g  1-exp(-gas)=%.4f\n",
        name,
        dir[0],
        dir[1],
        dir[2],
        asinf(atm_clampf(dir[2], -1.f, 1.f)) * 180.f / ATM_PI,
        tau_g_view,
        tau_v_view,
        1.f - expf(-tau_g_view));

    HostRng rng;
    rng.init(1);
    int n_gas = 0, n_vox = 0, n_none = 0, n_ks0 = 0;
    double sum_nee = 0.0, sum_tr = 0.0, sum_ksp = 0.0, sum_tr_abs = 0.0;
    std::vector<float> z_gas;
    z_gas.reserve(static_cast<size_t>(nsamp));
    for (int i = 0; i < nsamp; ++i) {
        AtmHit h;
        atm_next_volume_event_into(atm, org, dir, tmax, rng, &h);
        if (h.event == ATM_EVENT_SCATTER_GAS || h.event == ATM_EVENT_ABSORB_GAS) {
            ++n_gas;
            z_gas.push_back(h.pos[2]);
            float ka_g = 0.f, ks_g = 0.f;
            atm_gas_k_at_z(&atm->gas, h.pos[2], &ka_g, &ks_g);
            int ijk[3];
            atm_world_to_ijk(&atm->voxels, h.pos, ijk);
            const float ks_p = atm_fetch_ks(&atm->voxels, ijk[0], ijk[1], ijk[2]);
            const float ks_tot = ks_g + ks_p;
            sum_ksp += ks_p;
            if (ks_tot <= 1e-20f) {
                ++n_ks0;
                continue;
            }
            const float tau_v = atm_voxel_optical_depth(&atm->voxels, h.pos, sun, tmax, ATM_K_EXT);
            const float tau_g = atm_gas_optical_depth(&atm->gas, h.pos, sun, tmax, ATM_K_EXT);
            const float tau = tau_v + tau_g;
            const float tr = (tau <= 0.f) ? 1.f : (tau > 80.f ? 0.f : expf(-tau));
            const float tr_abs = atm_absorption_transmittance(atm, org, dir, h.t);
            const float mu = atm_dot3(dir, sun);
            const float phase = atm_rayleigh_phase(mu);
            sum_tr += tr;
            sum_tr_abs += tr_abs;
            sum_nee += static_cast<double>(tr_abs * phase * 24.f * tr);
        } else if (h.event == ATM_EVENT_SCATTER_VOXEL || h.event == ATM_EVENT_ABSORB_VOXEL) {
            ++n_vox;
        } else {
            ++n_none;
        }
    }
    const double inv = 1.0 / static_cast<double>(nsamp);
    std::printf(
        "  events gas=%.4f vox=%.4f none=%.4f  ks_tot=0 among gas=%.4f\n",
        n_gas * inv,
        n_vox * inv,
        n_none * inv,
        n_gas ? static_cast<double>(n_ks0) / n_gas : 0.0);
    if (!z_gas.empty()) {
        std::vector<float> zs = z_gas;
        std::printf(
            "  gas hit z p10/50/90 = %.1f / %.1f / %.1f  mean ks_p=%.4g\n",
            percentile(zs, 0.1f),
            percentile(z_gas, 0.5f),
            percentile(zs, 0.9f),
            n_gas ? sum_ksp / n_gas : 0.0);
        const int n_ok = n_gas - n_ks0;
        std::printf(
            "  analog NEE at gas hits (Rayleigh, no mesh): mean Tr=%.4g mean Tr_abs=%.4g mean contrib=%.4g\n"
            "  path-mean contrib (incl misses)=%.4g\n",
            n_ok ? sum_tr / n_ok : 0.0,
            n_ok ? sum_tr_abs / n_ok : 0.0,
            n_ok ? sum_nee / n_ok : 0.0,
            sum_nee * inv);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s voxels.bin gas_layers.bin\n", argv[0]);
        return 1;
    }
    auto voxb = read_all(argv[1]);
    const char* vp = voxb.data();
    int nx = 0, ny = 0, nz = 0, pad = 0;
    std::memcpy(&nx, vp, 4);
    std::memcpy(&ny, vp + 4, 4);
    std::memcpy(&nz, vp + 8, 4);
    std::memcpy(&pad, vp + 12, 4);
    (void)pad;
    float origin[3], spacing[3];
    std::memcpy(origin, vp + 16, 12);
    std::memcpy(spacing, vp + 28, 12);
    const int ncell = nx * ny * nz;
    const float* ks = reinterpret_cast<const float*>(vp + 40);
    const float* ka = ks + ncell;
    const float* temp = ka + ncell;
    if (voxb.size() < 40 + static_cast<size_t>(ncell) * 12) {
        std::fprintf(stderr, "voxel file too small nx=%d ny=%d nz=%d\n", nx, ny, nz);
        return 1;
    }

    auto gasb = read_all(argv[2]);
    const char* gp = gasb.data();
    int nlayers = 0;
    std::memcpy(&nlayers, gp, 4);
    const float* zif = reinterpret_cast<const float*>(gp + 4);
    const float* gka = zif + nlayers + 1;
    const float* gks = gka + nlayers;
    const float* gt = gks + nlayers;
    if (gasb.size() < 4 + static_cast<size_t>(nlayers + 1 + nlayers * 3) * 4) {
        std::fprintf(stderr, "gas file too small nlayers=%d\n", nlayers);
        return 1;
    }

    AtmVoxelGrid grid{};
    grid.nx = nx;
    grid.ny = ny;
    grid.nz = nz;
    grid.origin[0] = origin[0];
    grid.origin[1] = origin[1];
    grid.origin[2] = origin[2];
    grid.dx = spacing[0];
    grid.dy = spacing[1];
    grid.dz = spacing[2];
    grid.ka = ka;
    grid.ks = ks;
    grid.temperature = temp;
    grid.brick = 8;
    std::vector<float> kmax;
    kmax.resize(static_cast<size_t>(((nx + 7) / 8) * ((ny + 7) / 8) * ((nz + 7) / 8)));
    atm_build_brick_majorants(&grid, kmax.data());

    AtmGasLayers gas{};
    gas.nlayers = nlayers;
    gas.z_iface = zif;
    gas.ka = gka;
    gas.ks = gks;
    gas.temperature = gt;

    AtmParams atm{};
    atm.enabled = 1;
    atm.spectral_mode = ATM_SPECTRAL_SW;
    atm.mie_omega = 1.f;
    atm.mie_g = 0.865f;
    atm.path_tmax = 1.2e5f;
    atm.voxels = grid;
    atm.gas = gas;

    const float az = 225.f * ATM_PI / 180.f;
    const float el = 30.f * ATM_PI / 180.f;
    float sun[3] = {cosf(el) * cosf(az), cosf(el) * sinf(az), sinf(el)};
    atm_normalize3(sun);

    const float pos[3] = {10245.37f, 615.02f, 422.34f};
    const float tgt[3] = {-2977.39f, 6829.40f, 842.40f};
    const float tmax = 1.2e5f;
    const int nsamp = 4000;

    std::printf(
        "voxels %dx%dx%d origin=(%.1f,%.1f,%.1f) dx=(%.1f,%.1f,%.1f) gas nlayers=%d z=[%.1f,%.1f]\n"
        "sun=(%.4f,%.4f,%.4f) path_tmax=%.0f  (NEE analog, mesh occlusion NOT in this probe)\n",
        nx,
        ny,
        nz,
        origin[0],
        origin[1],
        origin[2],
        spacing[0],
        spacing[1],
        spacing[2],
        nlayers,
        zif[0],
        zif[nlayers],
        sun[0],
        sun[1],
        sun[2],
        tmax);

    float dir[3];
    persp_dir(0, 0, 1280, 720, pos, tgt, 30.f, dir);
    probe_ray("mountain.sh pixel (0,0) top-left", &atm, pos, dir, sun, tmax, nsamp);
    persp_dir(640, 0, 1280, 720, pos, tgt, 30.f, dir);
    probe_ray("mountain.sh pixel (640,0) top-center", &atm, pos, dir, sun, tmax, nsamp);
    persp_dir(200, 40, 1280, 720, pos, tgt, 30.f, dir);
    probe_ray("mountain.sh pixel (200,40) upper-left", &atm, pos, dir, sun, tmax, nsamp);

    /* Gas column only: force empty voxels to isolate Rayleigh NEE. */
    std::vector<float> zero(static_cast<size_t>(ncell), 0.f);
    atm.voxels.ks = zero.data();
    atm.voxels.ka = zero.data();
    std::fill(kmax.begin(), kmax.end(), 0.f);
    atm.voxels.kext_max = kmax.data();
    persp_dir(0, 0, 1280, 720, pos, tgt, 30.f, dir);
    probe_ray("GAS-ONLY pixel (0,0)", &atm, pos, dir, sun, tmax, nsamp);
    return 0;
}
