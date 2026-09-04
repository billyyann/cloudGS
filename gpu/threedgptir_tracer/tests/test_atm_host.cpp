/* Host-side checks of GPU Woodcock / 1D gas (same formulas as atmosphere.cuh).
 * Compile:
 *   g++ -O2 -std=c++17 -I ../include test_atm_host.cpp -o test_atm_host -lm
 */

#include <3dgptir/kernels/cuda/atmosphere.cuh>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

static int g_fail = 0;

static void expect_near(const char* name, double got, double want, double tol) {
    const double err = std::fabs(got - want);
    if (err > tol) {
        std::fprintf(stderr, "FAIL %s: got %.6g want %.6g (err %.6g > %.6g)\n", name, got, want, err, tol);
        ++g_fail;
    } else {
        std::printf("ok   %s: %.6g (want %.6g)\n", name, got, want);
    }
}

int main() {
    /* Homogeneous voxel slab: mean free path = 1/k */
    const int n = 32;
    const float k = 0.2f;
    std::vector<float> ka(n * n * n, 0.f);
    std::vector<float> ks(n * n * n, k);
    std::vector<float> temp(n * n * n, 280.f);
    std::vector<float> kmax;

    AtmVoxelGrid grid{};
    grid.nx = grid.ny = grid.nz = n;
    grid.origin[0] = grid.origin[1] = grid.origin[2] = 0.f;
    grid.dx = grid.dy = grid.dz = 1.f;
    grid.ka = ka.data();
    grid.ks = ks.data();
    grid.temperature = temp.data();
    grid.brick = 8;
    kmax.resize(((n + 7) / 8) * ((n + 7) / 8) * ((n + 7) / 8));
    atm_build_brick_majorants(&grid, kmax.data());

    HostRng rng;
    rng.init(42);
    const int nsamp = 8000;
    double sum_t = 0.0;
    int nreal = 0;
    for (int i = 0; i < nsamp; ++i) {
        float org[3] = {0.5f, 16.f, 16.f};
        float dir[3] = {1.f, 0.f, 0.f};
        AtmHit h = atm_woodcock_voxels(&grid, org, dir, 100.f, rng, -1.f);
        if (h.event == ATM_EVENT_SCATTER_VOXEL || h.event == ATM_EVENT_ABSORB_VOXEL) {
            sum_t += h.t;
            ++nreal;
        }
    }
    expect_near("voxel_mfp", sum_t / nreal, 1.0 / k, 0.08);
    /* Z-slab + XY wrap: +X never leaves the cloud band, so τ = k * tmax. */
    expect_near("voxel_hit_frac", static_cast<double>(nreal) / nsamp, 1.0 - std::exp(-k * 100.f), 0.02);

    /* Nadir must exit the Z slab (do not walk to tmax inside an infinite XY box). */
    rng.init(3);
    int n_beyond_slab = 0;
    int n_nadir_hit = 0;
    for (int i = 0; i < nsamp; ++i) {
        float org[3] = {16.f, 16.f, 64.f};
        float dir[3] = {0.f, 0.f, -1.f};
        AtmHit h = atm_woodcock_voxels(&grid, org, dir, 200.f, rng, -1.f);
        if (h.event == ATM_EVENT_SCATTER_VOXEL || h.event == ATM_EVENT_ABSORB_VOXEL) {
            ++n_nadir_hit;
            if (h.t > 64.5f) {
                ++n_beyond_slab;
            }
        }
    }
    expect_near("nadir_z_hit_frac", static_cast<double>(n_nadir_hit) / nsamp, 1.0 - std::exp(-k * 32.f), 0.03);
    expect_near("nadir_z_exit", static_cast<double>(n_beyond_slab) / nsamp, 0.0, 0.01);

    /* Camera outside the canonical XY tile must still sample wrapped k. */
    rng.init(11);
    int n_wrap = 0;
    for (int i = 0; i < nsamp; ++i) {
        float org[3] = {-8.f, 16.f, 16.f};
        float dir[3] = {1.f, 0.f, 0.f};
        AtmHit h = atm_woodcock_voxels(&grid, org, dir, 40.f, rng, -1.f);
        if (h.event == ATM_EVENT_SCATTER_VOXEL || h.event == ATM_EVENT_ABSORB_VOXEL) {
            ++n_wrap;
        }
    }
    expect_near("wrap_xy_hit_frac", static_cast<double>(n_wrap) / nsamp, 1.0 - std::exp(-k * 40.f), 0.03);

    /* 1D gas layer along z */
    const float kg = 0.5f;
    float zif[3] = {0.f, 20.f, 40.f};
    float gka[2] = {0.f, 0.f};
    float gks[2] = {kg, 0.f};
    float gt[2] = {250.f, 250.f};
    AtmGasLayers gas{};
    gas.nlayers = 2;
    gas.z_iface = zif;
    gas.ka = gka;
    gas.ks = gks;
    gas.temperature = gt;

    rng.init(7);
    sum_t = 0.0;
    nreal = 0;
    for (int i = 0; i < nsamp; ++i) {
        float org[3] = {0.f, 0.f, 1.f};
        float dir[3] = {0.f, 0.f, 1.f};
        AtmHit h = atm_gas_free_path(&gas, org, dir, 100.f, rng);
        if (h.event == ATM_EVENT_SCATTER_GAS || h.event == ATM_EVENT_ABSORB_GAS) {
            sum_t += h.t;
            ++nreal;
        }
    }
    expect_near("gas_mfp", sum_t / nreal, 1.0 / kg, 0.08);

    /* Competing risks: voxel ks=0.1, gas ks=0.3 along +Z. P(gas) = 0.75 */
    std::vector<float> ka2(n * n * n, 0.f);
    std::vector<float> ks2(n * n * n, 0.1f);
    std::vector<float> kmax2;
    AtmVoxelGrid grid2 = grid;
    grid2.ka = ka2.data();
    grid2.ks = ks2.data();
    grid2.origin[0] = grid2.origin[1] = 0.f;
    grid2.origin[2] = 0.f;
    kmax2.resize(kmax.size());
    atm_build_brick_majorants(&grid2, kmax2.data());

    float zif2[2] = {0.f, 32.f};
    float gka2[1] = {0.f};
    float gks2[1] = {0.3f};
    float gt2[1] = {260.f};
    AtmGasLayers gas2{};
    gas2.nlayers = 1;
    gas2.z_iface = zif2;
    gas2.ka = gka2;
    gas2.ks = gks2;
    gas2.temperature = gt2;

    AtmParams atm{};
    atm.enabled = 1;
    atm.path_tmax = 100.f;
    atm.mie_omega = 0.f;
    atm.voxels = grid2;
    atm.gas = gas2;

    rng.init(99);
    int n_gas = 0, n_vox = 0, n_none = 0;
    const int ncomp = 12000;
    for (int i = 0; i < ncomp; ++i) {
        float org[3] = {16.f, 16.f, 1.f};
        float dir[3] = {0.f, 0.f, 1.f};
        AtmHit h = atm_next_volume_event(&atm, org, dir, 30.f, rng);
        if (h.event == ATM_EVENT_SCATTER_GAS || h.event == ATM_EVENT_ABSORB_GAS) {
            ++n_gas;
        } else if (h.event == ATM_EVENT_SCATTER_VOXEL || h.event == ATM_EVENT_ABSORB_VOXEL) {
            ++n_vox;
        } else {
            ++n_none;
        }
    }
    const double p_gas = static_cast<double>(n_gas) / (n_gas + n_vox);
    expect_near("compete_p_gas", p_gas, 0.3 / 0.4, 0.03);
    expect_near("compete_escape", static_cast<double>(n_none) / ncomp, 0.0, 0.02);

    /* Planck sanity: brighter at higher T */
    const float B_hot = atm_planck_monochromatic(10e-6f, 300.f);
    const float B_cold = atm_planck_monochromatic(10e-6f, 250.f);
    if (!(B_hot > B_cold && B_hot > 0.f)) {
        std::fprintf(stderr, "FAIL planck order hot=%g cold=%g\n", B_hot, B_cold);
        ++g_fail;
    } else {
        std::printf("ok   planck_order hot=%.6g > cold=%.6g\n", B_hot, B_cold);
    }

    /* Brick majorant is max of kext */
    if (std::fabs(grid.kext_max[0] - k) > 1e-5f) {
        std::fprintf(stderr, "FAIL brick kmax %g vs %g\n", grid.kext_max[0], k);
        ++g_fail;
    } else {
        std::printf("ok   brick_kmax %g\n", grid.kext_max[0]);
    }

    if (g_fail) {
        std::fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    std::printf("all host atmosphere checks passed\n");
    return 0;
}
