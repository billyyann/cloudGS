/* Dump DZVAR .htcp liquid water to a regular voxel ks/ka/T npy-like raw file. */
#include <high_tune/htcp.h>
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
        : "/home/szyj2025/scatterGS/cloudGS/htrdr-atmosphere-spk/htrdr-Atmosphere-Starter-Pack-0.8.0/clouds/DZVAR.1.ARMCU.008.diaKCL.htcp";
    const char* out = argc > 2 ? argv[2] : "/tmp/dzvar_voxels.bin";
    struct htcp* c = nullptr;
    if (htcp_create(nullptr, nullptr, 0, &c) != RES_OK) {
        std::fprintf(stderr, "htcp_create failed\n");
        return 1;
    }
    if (htcp_load(c, path) != RES_OK) {
        std::fprintf(stderr, "htcp_load failed: %s\n", path);
        return 1;
    }
    struct htcp_desc d;
    htcp_get_desc(c, &d);
    const int nx = (int)d.spatial_definition[0];
    const int ny = (int)d.spatial_definition[1];
    const int nz = (int)d.spatial_definition[2];
    std::printf("htcp %d x %d x %d  irregular_z=%d  time=%zu\n", nx, ny, nz, d.irregular_z, d.time_definition);
    std::printf("bbox [%.3f %.3f %.3f] - [%.3f %.3f %.3f]\n", d.lower[0], d.lower[1], d.lower[2], d.upper[0], d.upper[1], d.upper[2]);
    std::printf("vxsz_xy %.3f %.3f\n", d.vxsz_x, d.vxsz_y);

    double z0 = d.lower[2], z1 = d.upper[2];
    if (d.irregular_z && d.coord_z) {
        z0 = d.coord_z[0];
        z1 = d.coord_z[nz - 1] + (d.vxsz_z ? d.vxsz_z[nz - 1] : 0.0);
        std::printf("z[0]=%.3f z[last]=%.3f\n", d.coord_z[0], d.coord_z[nz - 1]);
    }

    const int nreg = 80;
    const float z0f = 0.f;
    const float z1f = 4000.f;
    const float dzr = (z1f - z0f) / (float)nreg;
    std::vector<float> ks((size_t)nx * ny * nreg, 0.f);
    std::vector<float> ka((size_t)nx * ny * nreg, 0.f);
    std::vector<float> temp((size_t)nx * ny * nreg, 280.f);
    std::vector<float> wsum((size_t)nx * ny * nreg, 0.f);
    int npos = 0;
    double rct_max = 0, rct_sum = 0;
    for (int iz = 0; iz < nz; ++iz) {
        const float zlo = d.irregular_z && d.coord_z ? (float)d.coord_z[iz] : (float)(d.lower[2] + iz * (d.upper[2] - d.lower[2]) / nz);
        const float dz = (d.vxsz_z ? (float)d.vxsz_z[iz] : dzr);
        const float zhi = zlo + dz;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                const double rct = htcp_desc_RCT_at(&d, (size_t)ix, (size_t)iy, (size_t)iz, 0);
                if (rct > rct_max) {
                    rct_max = rct;
                }
                rct_sum += rct;
                if (rct < 1e-6) {
                    continue;
                }
                /* RCT is kg/kg; LWC≈RCT*ρ_air. k_ext ≈ 150 m^2/kg * LWC. */
                const float lwc = (float)rct * 1.0f;
                const float k = lwc * 120.f;
                int iz0 = (int)std::floor((zlo - z0f) / dzr);
                int iz1 = (int)std::floor((zhi - z0f) / dzr);
                if (iz1 >= nreg) {
                    iz1 = nreg - 1;
                }
                if (iz0 < 0) {
                    iz0 = 0;
                }
                for (int rz = iz0; rz <= iz1 && rz < nreg; ++rz) {
                    const float clo = z0f + rz * dzr;
                    const float chi = clo + dzr;
                    const float ov = std::min(zhi, chi) - std::max(zlo, clo);
                    if (ov <= 0.f) {
                        continue;
                    }
                    const int id = (rz * ny + iy) * nx + ix;
                    ks[id] += k * (ov / dzr);
                    ka[id] += k * 0.001f * (ov / dzr);
                    wsum[id] += ov;
                    double tk = 280.0;
                    if (d.T) {
                        tk = htcp_desc_T_at(&d, (size_t)ix, (size_t)iy, (size_t)iz, 0);
                    }
                    temp[id] = (float)tk;
                    ++npos;
                }
            }
        }
    }
    std::printf("RCT max=%.6g mean=%.6g  deposited cells=%d\n", rct_max, rct_sum / (nx * ny * nz), npos);

    FILE* f = std::fopen(out, "wb");
    if (!f) {
        std::perror(out);
        return 1;
    }
    int hdr[8] = {nx, ny, nreg, d.irregular_z};
    float origin[3] = {(float)d.lower[0], (float)d.lower[1], z0f};
    float spacing[3] = {(float)d.vxsz_x, (float)d.vxsz_y, dzr};
    std::fwrite(hdr, 4, 4, f);
    std::fwrite(origin, 4, 3, f);
    std::fwrite(spacing, 4, 3, f);
    std::fwrite(ks.data(), 4, ks.size(), f);
    std::fwrite(ka.data(), 4, ka.size(), f);
    std::fwrite(temp.data(), 4, temp.size(), f);
    std::fclose(f);
    std::printf("wrote %s origin=(%.1f,%.1f,%.1f) dx=(%.2f,%.2f,%.2f) nreg=%d\n", out, origin[0], origin[1], origin[2], spacing[0], spacing[1], spacing[2], nreg);
    htcp_ref_put(c);
    return 0;
}
