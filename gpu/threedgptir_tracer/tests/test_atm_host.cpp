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

    /* Irregular Z analog τ: thin lower layer must not be stretched to mean dz. */
    {
        const int nz = 2, ny = 1, nx = 1;
        std::vector<float> ka_i(nz, 0.f);
        std::vector<float> ks_i(nz, 0.f);
        ks_i[0] = 1.f;
        std::vector<float> tmp_i(nz, 280.f);
        std::vector<float> kmax_i(1, 1.f);
        float zifv[3] = {0.f, 5.f, 20.f};
        AtmVoxelGrid gi{};
        gi.nx = nx;
        gi.ny = ny;
        gi.nz = nz;
        gi.brick = 8;
        gi.nbx = gi.nby = gi.nbz = 1;
        gi.origin[0] = gi.origin[1] = gi.origin[2] = 0.f;
        gi.dx = gi.dy = 1.f;
        gi.dz = 10.f; /* mean dz, must not be used when z_iface is set */
        gi.ka = ka_i.data();
        gi.ks = ks_i.data();
        gi.temperature = tmp_i.data();
        gi.kext_max = kmax_i.data();
        float orgz[3] = {0.5f, 0.5f, -1.f};
        float dirz[3] = {0.f, 0.f, 1.f};
        const float tau_nat = atm_voxel_optical_depth(&gi, orgz, dirz, 40.f, ATM_K_KS, 1.f, 1.f, zifv);
        const float tau_mean = atm_voxel_optical_depth(&gi, orgz, dirz, 40.f, ATM_K_KS, 1.f, 1.f, nullptr);
        expect_near("irreg_z_tau_native", tau_nat, 5.0, 2e-3);
        expect_near("irreg_z_tau_mean_dz", tau_mean, 10.0, 2e-3);
        int ijk[3];
        float pmid[3] = {0.5f, 0.5f, 4.f};
        atm_world_to_ijk(&gi, pmid, ijk, zifv);
        if (ijk[2] != 0) {
            std::fprintf(stderr, "FAIL z_to_iz mid-layer got %d\n", ijk[2]);
            ++g_fail;
        } else {
            std::printf("ok   z_to_iz native layer 0\n");
        }
        pmid[2] = 12.f;
        atm_world_to_ijk(&gi, pmid, ijk, zifv);
        if (ijk[2] != 1) {
            std::fprintf(stderr, "FAIL z_to_iz upper layer got %d\n", ijk[2]);
            ++g_fail;
        } else {
            std::printf("ok   z_to_iz native layer 1\n");
        }
    }

    /* Periodic XY analog must integrate past one domain (old cap 2*(nx+ny)+nz). */
    {
        const int n = 16;
        const float k = 0.05f;
        std::vector<float> ka_w(static_cast<size_t>(n * n * 4), 0.f);
        std::vector<float> ks_w(static_cast<size_t>(n * n * 4), k);
        std::vector<float> tmp_w(static_cast<size_t>(n * n * 4), 280.f);
        std::vector<float> kmax_w;
        AtmVoxelGrid gw{};
        gw.nx = gw.ny = n;
        gw.nz = 4;
        gw.origin[0] = gw.origin[1] = gw.origin[2] = 0.f;
        gw.dx = gw.dy = gw.dz = 1.f;
        gw.ka = ka_w.data();
        gw.ks = ks_w.data();
        gw.temperature = tmp_w.data();
        gw.brick = 8;
        kmax_w.resize(1);
        atm_build_brick_majorants(&gw, kmax_w.data());
        float orgw[3] = {0.5f, 8.f, 2.f};
        float dirw[3] = {1.f, 0.f, 0.f};
        const float tmax_w = 100.f;
        const float tau_w = atm_voxel_optical_depth(&gw, orgw, dirw, tmax_w, ATM_K_KS);
        expect_near("analog_xy_wrap_tau", tau_w, k * tmax_w, 0.05);
    }

    /* htgop three-case fetch: below first knot, linear through zero, log-log. */
    {
        const float xt[3] = {0.01f, 0.02f, 0.04f};
        const float kt[3] = {1.f, 4.f, 16.f};
        expect_near("htgop_k_below", atm_htgop_tab_k(xt, kt, 3, 0.005f), 0.5, 1e-5);
        expect_near("htgop_k_zero_k", atm_htgop_tab_k(xt, kt, 3, 0.0f), 0.0, 1e-12);
        const float kmid = atm_htgop_tab_k(xt, kt, 3, 0.02f);
        expect_near("htgop_k_knot", kmid, 4.0, 1e-5);
        const float kll = atm_htgop_tab_k(xt, kt, 3, 0.028284271f); /* sqrt(0.02*0.04) → sqrt(4*16)=8 */
        expect_near("htgop_k_loglog", kll, 8.0, 0.05);
        const float kz[2] = {0.f, 2.f};
        const float xz[2] = {0.01f, 0.02f};
        expect_near("htgop_k_linear_zero", atm_htgop_tab_k(xz, kz, 2, 0.015f), 1.0, 1e-5);
    }

    /* VOLUME GS pid-stencil: one particle at cell (1,1,1), eval at center → ka/ks. */
    {
        const int n = 4;
        std::vector<float> vka(n * n * n, 0.f), vks(n * n * n, 0.f), vtemp(n * n * n, 280.f);
        AtmVoxelGrid vg{};
        vg.nx = vg.ny = vg.nz = n;
        vg.origin[0] = vg.origin[1] = vg.origin[2] = 0.f;
        vg.dx = vg.dy = vg.dz = 1.f;
        vg.ka = vka.data();
        vg.ks = vks.data();
        vg.temperature = vtemp.data();
        vg.brick = 8;
        std::vector<float> vkmax;
        vkmax.resize(1);
        atm_build_brick_majorants(&vg, vkmax.data());

        std::vector<int> pid(n * n * n, -1);
        const int pix = 1, piy = 1, piz = 1;
        pid[(piz * n + piy) * n + pix] = 0;
        float center[3] = {1.5f, 1.5f, 1.5f};
        float scale[3] = {0.7071f, 0.7071f, 0.7071f};
        float quat[4] = {1.f, 0.f, 0.f, 0.f};
        float gka[1] = {0.1f};
        float gks[1] = {1.0f};
        AtmVolumeGs vgs{};
        vgs.enabled = 1;
        vgs.n = 1;
        vgs.rx = vgs.ry = vgs.rz = 2;
        vgs.pid = pid.data();
        vgs.center = center;
        vgs.scale = scale;
        vgs.quat = quat;
        vgs.ka = gka;
        vgs.ks = gks;
        vgs.kext_max = nullptr;
        vgs.majorant_boost = 4.f;
        float oka = 0.f, oks = 0.f;
        atm_volume_gs_eval_ka_ks(&vgs, &vg, center, &oka, &oks);
        expect_near("volume_gs_at_center_ka", oka, 0.1, 1e-5);
        expect_near("volume_gs_at_center_ks", oks, 1.0, 1e-5);
        float far[3] = {10.f, 10.f, 10.f};
        atm_volume_gs_eval_ka_ks(&vgs, &vg, far, &oka, &oks);
        expect_near("volume_gs_far_ka", oka, 0.0, 1e-6);
        expect_near("volume_gs_far_ks", oks, 0.0, 1e-6);
    }

    if (g_fail) {
        std::fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    std::printf("all host atmosphere checks passed\n");
    return 0;
}
