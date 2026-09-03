# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.utils.cpp_extension

from threedgrut.datasets.protocols import Batch
from threedgrut.model.filters import Filter
from threedgrut.model.ptir_helper import post_processing
from threedgrut.utils.logger import logger as rich_logger
from threedgrut.utils.timer import CudaTimer

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
#

_threedgptir_plugin = None


@dataclass(frozen=True)
class LightBuffers:
    lights: torch.Tensor
    alias_table: torch.Tensor
    mesh_vertices: torch.Tensor
    mesh_triangles: torch.Tensor
    mesh_lights: torch.Tensor
    mesh_triangle_alias_table: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device | str) -> "LightBuffers":
        return cls(
            lights=torch.empty((0, 9), dtype=torch.float32, device=device),
            alias_table=torch.empty((5, 0), dtype=torch.float32, device=device),
            mesh_vertices=torch.empty((0, 3), dtype=torch.float32, device=device),
            mesh_triangles=torch.empty((0, 3), dtype=torch.int32, device=device),
            mesh_lights=torch.empty((0, 8), dtype=torch.float32, device=device),
            mesh_triangle_alias_table=torch.empty(
                (3, 0), dtype=torch.float32, device=device
            ),
        )

    @classmethod
    def from_batch(cls, batch: Batch, device: torch.device | str) -> "LightBuffers":
        empty = cls.empty(device)

        def value_or_empty(name: str, fallback: torch.Tensor) -> torch.Tensor:
            value = getattr(batch, name, None)
            return fallback if value is None else value

        return cls(
            lights=value_or_empty("lights", empty.lights),
            alias_table=value_or_empty("light_alias_table", empty.alias_table),
            mesh_vertices=value_or_empty("mesh_light_vertices", empty.mesh_vertices),
            mesh_triangles=value_or_empty("mesh_light_triangles", empty.mesh_triangles),
            mesh_lights=value_or_empty("mesh_lights", empty.mesh_lights),
            mesh_triangle_alias_table=value_or_empty(
                "mesh_light_triangle_alias_table",
                empty.mesh_triangle_alias_table,
            ),
        ).to(device)

    def to(self, device: torch.device | str) -> "LightBuffers":
        return LightBuffers(
            lights=self.lights.to(device=device, dtype=torch.float32).contiguous(),
            alias_table=self.alias_table.to(
                device=device, dtype=torch.float32
            ).contiguous(),
            mesh_vertices=self.mesh_vertices.to(
                device=device, dtype=torch.float32
            ).contiguous(),
            mesh_triangles=self.mesh_triangles.to(
                device=device, dtype=torch.int32
            ).contiguous(),
            mesh_lights=self.mesh_lights.to(
                device=device, dtype=torch.float32
            ).contiguous(),
            mesh_triangle_alias_table=self.mesh_triangle_alias_table.to(
                device=device, dtype=torch.float32
            ).contiguous(),
        )

    def tensors(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            self.lights,
            self.alias_table,
            self.mesh_vertices,
            self.mesh_triangles,
            self.mesh_lights,
            self.mesh_triangle_alias_table,
        )


def load_threedgptir_plugin(conf):
    global _threedgptir_plugin
    if _threedgptir_plugin is None:
        try:
            from . import libthreedgptir_cc as threedgptir  # type: ignore
        except ImportError:
            from .setup_threedgptir import setup_threedgptir

            threedgptir = setup_threedgptir(conf)
        _threedgptir_plugin = threedgptir


# ----------------------------------------------------------------------------
#
class Tracer:
    class _Autograd(torch.autograd.Function):
        # Set for the duration of an inference render when mog pack cache is active.
        _active_pack_cache = None

        @staticmethod
        def forward(
            ctx,
            tracer_wrapper,
            frame_id,
            ray_to_world,
            ray_ori,
            ray_dir,
            mog_pos,
            mog_rot,
            mog_scl,
            mog_dns,
            mog_sph,
            mog_snrm,
            mog_malb,
            mog_mrgh,
            mog_mmet,
            environment,
            environment_alias_table,
            light_buffers,
            render_opts,
            sph_degree,
            min_transmittance,
            max_bounces,
        ):
            pack_cache = Tracer._Autograd._active_pack_cache
            if (
                pack_cache is not None
                and pack_cache.get("num_gaussians") == mog_pos.shape[0]
                and "particle_density" in pack_cache
                and "particle_material" in pack_cache
            ):
                particle_density = pack_cache["particle_density"]
                particle_material = pack_cache["particle_material"]
            else:
                # 格式: [位置(3), 密度(1), 旋转(4), 缩放(3), 占位(1)] = 12维
                particle_density = torch.concat(
                    [mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)],
                    dim=1,
                )
                # 格式: [反照率(3), 粗糙度(1), 金属度(1)] = 5维
                particle_material = torch.concat(
                    [mog_malb, mog_mrgh, mog_mmet], dim=1
                )
            light_buffers = light_buffers.to(ray_dir.device)
            (
                lights,                           # 光源参数
                light_alias_table,               # 重要性采样别名表
                mesh_light_vertices,             # 网格光源顶点
                mesh_light_triangles,            # 网格光源三角形
                mesh_lights,                     # 网格光源列表
                mesh_light_triangle_alias_table, # 三角形别名表
            ) = light_buffers.tensors()
            (
                ray_radiance,                    # 最终渲染颜色
                ray_density,                     # 体密度（透明度）
                ray_hit_distance,                # 命中距离（深度）
                ray_hit_distance_second_moment,  # 深度二阶矩
                ray_depth_distortion,            # 深度畸变
                ray_normals,                     # 几何法线
                ray_shadingnormal,               # 着色法线
                ray_material,                    # 材质参数
                hits_count,                      # 命中计数
                mog_visibility,                  # 高斯可见性
                ray_pbr,                         # PBR输出
                ray_light,                       # 光照分量
                pbr_components,                  # PBR分量分解
            ) = tracer_wrapper.trace(
                frame_id,
                ray_to_world,
                ray_ori,
                ray_dir,
                particle_density,
                particle_material,
                mog_sph,
                mog_snrm,
                environment,
                environment_alias_table,
                lights,
                light_alias_table,
                mesh_light_vertices,
                mesh_light_triangles,
                mesh_lights,
                mesh_light_triangle_alias_table,
                render_opts,
                sph_degree,
                min_transmittance,
                max_bounces,
            )
            # 输入参数、输出结果（用于梯度计算）、场景参数
            ctx.save_for_backward(
                ray_to_world,
                ray_ori,
                ray_dir,
                ray_radiance,
                ray_density,
                ray_hit_distance,
                ray_hit_distance_second_moment,
                ray_depth_distortion,
                ray_normals,
                ray_shadingnormal,
                ray_material,
                ray_pbr,
                ray_light,
                particle_density,
                particle_material,
                mog_sph,
                mog_snrm,
                environment,
                environment_alias_table,
                lights,
                light_alias_table,
                mesh_light_vertices,
                mesh_light_triangles,
                mesh_lights,
                mesh_light_triangle_alias_table,
            )
            ctx.frame_id = frame_id
            ctx.render_opts = render_opts
            ctx.sph_degree = sph_degree
            ctx.min_transmittance = min_transmittance
            ctx.max_bounces = max_bounces
            ctx.tracer_wrapper = tracer_wrapper
            return (
                ray_radiance,                    # RGB颜色 [B,H,W,3]
                ray_density,                     # 不透明度
                ray_hit_distance[:,:,:,0:1],     # 只取第一层深度
                ray_hit_distance_second_moment,  # 深度方差
                ray_depth_distortion,            # 畸变
                ray_normals,                     # 法线
                ray_shadingnormal,               # 着色法线
                ray_material,                    # 材质
                hits_count,                      # 命中计数
                mog_visibility,                  # 可见性
                ray_pbr,                         # PBR结果
                ray_light,                       # 光照
                pbr_components,                  # PBR分量
            )

        @staticmethod
        def backward(
            ctx,
            ray_radiance_grd,
            ray_density_grd,
            ray_hit_distance_grd,
            ray_hit_distance_second_moment_grd,
            ray_depth_distortion_grd,
            ray_normals_grd,
            ray_shadingnormal_grd,
            ray_material_grd,
            ray_hits_count_grd_UNUSED,
            mog_visibility_grd_UNUSED,
            ray_pbr_grd,
            ray_light_grd,
            pbr_components_grd_UNUSED,
        ):
            (
                ray_to_world,
                ray_ori,
                ray_dir,
                ray_radiance,
                ray_density,
                ray_hit_distance,
                ray_hit_distance_second_moment,
                ray_depth_distortion,
                ray_normals,
                ray_shadingnormal,
                ray_material,
                ray_pbr,
                ray_light,
                particle_density,
                particle_material,
                mog_sph,
                mog_snrm,
                environment,
                environment_alias_table,
                lights,
                light_alias_table,
                mesh_light_vertices,
                mesh_light_triangles,
                mesh_lights,
                mesh_light_triangle_alias_table,
            ) = ctx.saved_variables
            frame_id = ctx.frame_id
            if ray_light_grd is None:
                ray_light_grd = torch.zeros_like(ray_light)
            (
                particle_density_grd,
                particle_material_grd,
                mog_sph_grd,
                mog_sn_grd,
                environment_grd,
            ) = ctx.tracer_wrapper.trace_bwd(
                frame_id,
                ray_to_world,
                ray_ori,
                ray_dir,
                ray_radiance,
                ray_density,
                ray_hit_distance,
                ray_hit_distance_second_moment,
                ray_depth_distortion,
                ray_normals,
                ray_shadingnormal,
                ray_material,
                ray_pbr,
                ray_light,
                particle_density,
                particle_material,
                mog_sph,
                mog_snrm,
                environment,
                environment_alias_table,
                lights,
                light_alias_table,
                mesh_light_vertices,
                mesh_light_triangles,
                mesh_lights,
                mesh_light_triangle_alias_table,
                ray_radiance_grd,
                ray_density_grd,
                ray_hit_distance_grd,
                ray_hit_distance_second_moment_grd,
                ray_depth_distortion_grd,
                ray_normals_grd,
                ray_shadingnormal_grd,
                ray_material_grd,
                ray_pbr_grd,
                ray_light_grd,
                ctx.render_opts,
                ctx.sph_degree,
                ctx.min_transmittance,
                ctx.max_bounces,
            )
            mog_pos_grd, mog_dns_grd, mog_rot_grd, mog_scl_grd, _ = torch.split(
                particle_density_grd, [3, 1, 4, 3, 1], dim=1
            )
            mog_malb_grd, mog_mrgh_grd, mog_mmet_grd = torch.split(
                particle_material_grd, [3, 1, 1], dim=1
            )
            return (
                None,
                None,
                None,
                None,
                None,
                mog_pos_grd,
                mog_rot_grd,
                mog_scl_grd,
                mog_dns_grd,
                mog_sph_grd,
                mog_sn_grd,
                mog_malb_grd,
                mog_mrgh_grd,
                mog_mmet_grd,
                environment_grd,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    class RenderOpts(IntEnum):
        NONE = 0
        INDIRECT = 1
        DEFAULT = NONE

    _MULTISPP_PATTERNS = {
        1: ((0.5000, 0.5000),),
        2: ((0.2500, 0.2500), (0.7500, 0.7500)),
        4: ((0.3750, 0.1250), (0.8750, 0.3750), (0.6250, 0.8750), (0.1250, 0.6250)),
        8: (
            (0.5625, 0.6875),
            (0.4375, 0.3125),
            (0.8125, 0.4375),
            (0.3125, 0.8125),
            (0.1875, 0.1875),
            (0.0625, 0.5625),
            (0.6875, 0.0625),
            (0.9375, 0.9375),
        ),
        16: (
            (0.5625, 0.4375),
            (0.4375, 0.6875),
            (0.3125, 0.3750),
            (0.7500, 0.5625),
            (0.1875, 0.6250),
            (0.6250, 0.1875),
            (0.1875, 0.3125),
            (0.6875, 0.8125),
            (0.3750, 0.1250),
            (0.5000, 0.9375),
            (0.2500, 0.8750),
            (0.1250, 0.2500),
            (0.0000, 0.5000),
            (0.9375, 0.7500),
            (0.8750, 0.0625),
            (0.0625, 0.0000),
        ),
        32: tuple(
            ((x + 0.5) / 8.0, (y + 0.5) / 4.0) for y in range(4) for x in range(8)
        ),
    }

    def __init__(self, conf):

        self.device = "cuda"
        self.conf = conf
        self.num_update_bvh = 0
        self._warned_spp_fallback = False
        self._logged_spp_configs = set()
        self.pred_pbr_filter = Filter(self.conf.render.get("filter_type", "none"))
        self.visualize_lights = bool(self.conf.render.get("visualize_lights", False))

        logger.info(
            f'🔆 Creating threedgptir Optix tracing pipeline.. Using CUDA path: "{torch.utils.cpp_extension.CUDA_HOME}"'
        )
        torch.zeros(
            1, device=self.device
        )  # Create a dummy tensor to force cuda context init
        load_threedgptir_plugin(conf)

        self.tracer_wrapper = _threedgptir_plugin.OptixTracer(
            os.path.dirname(__file__),
            torch.utils.cpp_extension.CUDA_HOME,
            self.conf.render.pipeline_type,
            self.conf.render.backward_pipeline_type,
            self.conf.render.primitive_type,
            self.conf.render.particle_kernel_degree,
            self.conf.render.particle_kernel_min_response,
            self.conf.render.particle_kernel_density_clamping,
            self.conf.render.particle_radiance_sph_degree,
            self.conf.render.enable_normals,
            self.conf.render.enable_hitcounts,
            self.conf.render.enable_mis,
            self.conf.render.get(
                "enable_metallic",
                self.conf.model.get("optimize_material_metallic", False),
            ),
            self.visualize_lights,
            bool(self.conf.render.get("enable_raygen_setup_profiling", False)),
            int(self.conf.render.get("raygen_ablation_stage", 1)),
        )

        self.frame_timer = (
            CudaTimer() if self.conf.render.enable_kernel_timings else None
        )
        self.timings = {}
        self._mog_pack_cache = None

    def invalidate_mog_pack_cache(self) -> None:
        self._mog_pack_cache = None
        Tracer._Autograd._active_pack_cache = None

    def _cache_mog_pack_enabled(self, train: bool) -> bool:
        return (not train) and bool(self.conf.render.get("cache_mog_pack", True))

    def _gather_mog_tensors(self, gaussians) -> dict:
        return {
            "mog_pos": gaussians.positions.contiguous(),
            "mog_rot": gaussians.get_rotation().contiguous(),
            "mog_scl": gaussians.get_scale().contiguous(),
            "mog_dns": gaussians.get_density().contiguous(),
            "mog_sph": gaussians.get_features().contiguous(),
            "mog_snrm": gaussians.get_shading_normal().contiguous(),
            "mog_malb": gaussians.get_material_albedo().contiguous(),
            "mog_mrgh": gaussians.get_material_roughness().contiguous(),
            "mog_mmet": gaussians.get_material_metallic().contiguous(),
            "num_gaussians": int(gaussians.num_gaussians),
        }

    def _build_mog_pack(self, gaussians) -> dict:
        pack = self._gather_mog_tensors(gaussians)
        mog_pos = pack["mog_pos"]
        mog_dns = pack["mog_dns"]
        mog_rot = pack["mog_rot"]
        mog_scl = pack["mog_scl"]
        pack["particle_density"] = torch.concat(
            [mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)],
            dim=1,
        )
        pack["particle_material"] = torch.concat(
            [pack["mog_malb"], pack["mog_mrgh"], pack["mog_mmet"]], dim=1
        )
        return pack

    def _get_mog_pack(self, gaussians, use_cache: bool) -> dict:
        num_gaussians = int(gaussians.num_gaussians)
        if use_cache and self._mog_pack_cache is not None:
            if self._mog_pack_cache.get("num_gaussians") == num_gaussians:
                return self._mog_pack_cache
        if use_cache:
            pack = self._build_mog_pack(gaussians)
            cached = {
                key: (value.detach() if torch.is_tensor(value) else value)
                for key, value in pack.items()
            }
            self._mog_pack_cache = cached
            return cached
        return self._gather_mog_tensors(gaussians)

    def _required_train_output_keys(self) -> set[str]:
        keys = {
            "pred_pbr",
            "pred_material",
        }

        lambda_light = float(self.conf.loss.get("lambda_light", 0.0))
        if (
            bool(self.conf.loss.get("use_light_consistency", False))
            or lambda_light > 0.0
        ):
            keys.add("pred_light")

        if bool(self.conf.loss.get("use_edge_aware_smoothness", False)):
            keys.update(self.conf.loss.get("edge_aware_smoothness_outputs", []))

        return keys

    def _train_output_key_sets(
        self, train: bool
    ) -> tuple[set[str] | None, set[str] | None]:
        if not train or not bool(
            self.conf.render.get("hide_intermediate_outputs", False)
        ):
            return None, None

        final_output_keys = self._required_train_output_keys()
        accumulation_keys = set(final_output_keys)
        if "pred_pbr" in accumulation_keys:
            accumulation_keys.add("pred_opacity")
        if "pred_direct" in accumulation_keys or "pred_indirect" in accumulation_keys:
            accumulation_keys.update(("pbr_components", "pred_opacity"))
        return final_output_keys, accumulation_keys

    def _postprocess_pbr_outputs(self, outputs: dict[str, torch.Tensor]) -> None:
        pred_opacity = outputs.get("pred_opacity")
        opacity_weighted_keys = ("pred_pbr", "pred_direct", "pred_indirect")
        if pred_opacity is not None:
            for key in opacity_weighted_keys:
                value = outputs.get(key)
                if value is not None:
                    outputs[key] = value * pred_opacity

        for key in (*opacity_weighted_keys, "pred_light"):
            value = outputs.get(key)
            if value is not None:
                outputs[key] = self.pred_pbr_filter(value)

    @staticmethod
    def _normalize_depth_output(outputs: dict[str, torch.Tensor]) -> None:
        pred_dist = outputs.get("pred_dist")
        pred_opacity = outputs.get("pred_opacity")
        if pred_dist is not None and pred_opacity is not None:
            outputs["pred_dist"] = torch.nan_to_num(
                pred_dist / pred_opacity, 0.0, 0.0
            )

    def _get_spp(self, train: bool) -> int:
        if train:
            spp = self.conf.render.inversion_spp
        else:
            spp = self.conf.render.render_spp
        return max(1, int(spp))

    def _get_spp_chunk(self, spp: int) -> int:
        return min(spp, max(1, int(self.conf.render.get("spp_chunk", spp))))

    def _warn_spp_fallback(self, reason: str):
        if not self._warned_spp_fallback:
            rich_logger.warning(f"PTIR SPP fallback to spp=1: {reason}.")
            self._warned_spp_fallback = True

    @staticmethod
    def _flat(value, dtype: torch.dtype, device: torch.device):
        if value is None:
            return None
        return torch.as_tensor(value, dtype=dtype, device=device).flatten()

    def _extract_pinhole_intrinsics(
        self,
        gpu_batch: Batch,
        dtype: torch.dtype,
        device: torch.device,
        warn: bool = True,
    ):
        # Case 1: direct intrinsics, expected as [fx, fy, cx, cy] or 3x3 K.
        intrinsics = self._flat(getattr(gpu_batch, "intrinsics", None), dtype, device)
        if intrinsics is not None:
            if intrinsics.numel() == 4:
                return tuple(intrinsics)
            if intrinsics.numel() == 9:
                K = intrinsics.reshape(3, 3)
                return K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            if warn:
                self._warn_spp_fallback("unsupported intrinsics format")
            return None

        # Case 2: nerfstudio OpenCV pinhole camera parameters.
        params = getattr(
            gpu_batch, "intrinsics_OpenCVPinholeCameraModelParameters", None
        )
        if params is None:
            if warn:
                self._warn_spp_fallback("missing pinhole intrinsics")
            return None

        def get(name):
            if isinstance(params, dict):
                return params.get(name)
            return getattr(params, name, None)

        # SPP ray expansion assumes distortion-free pinhole rays.
        for name in ("radial_coeffs", "tangential_coeffs", "thin_prism_coeffs"):
            coeffs = self._flat(get(name), dtype, device)
            if coeffs is not None and coeffs.numel() > 0:
                if torch.any(torch.abs(coeffs) > 1e-8).item():
                    if warn:
                        self._warn_spp_fallback("distorted camera model")
                    return None

        focal = self._flat(get("focal_length"), dtype, device)
        principal = self._flat(get("principal_point"), dtype, device)

        if focal is None or principal is None or focal.numel() < 1 or principal.numel() < 2:
            if warn:
                self._warn_spp_fallback("incomplete pinhole intrinsics")
            return None

        fx = focal[0]
        fy = focal[1] if focal.numel() > 1 else focal[0]
        cx, cy = principal[:2]
        return fx, fy, cx, cy

    def _can_expand_spp(self, gpu_batch: Batch) -> bool:
        return (
            getattr(gpu_batch, "pixel_coords", None) is not None
            and self._extract_pinhole_intrinsics(
                gpu_batch,
                gpu_batch.rays_dir.dtype,
                gpu_batch.rays_dir.device,
                warn=False,
            )
            is not None
        )

    def _make_spp_jitter(
        self,
        spp: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
        frame_id: int,
    ):
        if spp in self._MULTISPP_PATTERNS:
            return torch.tensor(
                self._MULTISPP_PATTERNS[spp], dtype=dtype, device=device
            ).view(spp, 1, 1, 2)

        grid_w = int(math.ceil(math.sqrt(spp)))
        grid_h = int(math.ceil(spp / grid_w))
        sample_idx = torch.arange(spp, dtype=dtype, device=device)
        jitter = torch.stack(
            (
                (torch.remainder(sample_idx, grid_w) + 0.5) / grid_w,
                (torch.div(sample_idx, grid_w, rounding_mode="floor") + 0.5) / grid_h,
            ),
            dim=-1,
        )

        generator = torch.Generator(device=device)
        generator.manual_seed((int(frame_id) + 1) * 1315423911 + spp * 2654435761)
        shift = torch.rand((1, 2), dtype=dtype, device=device, generator=generator)
        return torch.remainder(jitter + shift, 1.0).view(spp, 1, 1, 2)

    def _expand_rays_for_spp(
        self,
        gpu_batch: Batch,           # 输入的批次数据，包含光线信息
        spp: int,                   # samples per pixel，每个像素的采样数
        frame_id: int,              # 帧ID，用于生成随机种子
        jitter: torch.Tensor | None = None,  # 可选的抖动偏移量
    ):
        rays_ori = gpu_batch.rays_ori.contiguous()
        rays_dir = gpu_batch.rays_dir.contiguous()
        if spp == 1 and jitter is None:
            return rays_ori, rays_dir, 1

        pixel_coords = getattr(gpu_batch, "pixel_coords", None)
        intrinsics = self._extract_pinhole_intrinsics(
            gpu_batch, rays_dir.dtype, rays_dir.device
        )
        if intrinsics is None or pixel_coords is None:
            if pixel_coords is None:
                self._warn_spp_fallback("batch has no pixel_coords")
            return rays_ori, rays_dir, 1

        base_batch, h, w, _ = rays_dir.shape
        #fx和fy是水平和垂直方向的焦距（单位：像素）；cx和cy是主点坐标（
        fx, fy, cx, cy = intrinsics
        pixel_origin = (
            pixel_coords.to(dtype=rays_dir.dtype, device=rays_dir.device) - 0.5
        )
        if jitter is None:
            jitter = self._make_spp_jitter(
                spp, h, w, rays_dir.device, rays_dir.dtype, frame_id
            )

        pixel_origin = pixel_origin.unsqueeze(0)
        jitter = (
            jitter.view(spp, 1, 1, 1, 2)
            if jitter.shape[1:3] == (1, 1)
            else jitter.view(spp, 1, h, w, 2)
        )
        dirs = torch.stack(
            (   
                #计算射线方向的 X、Y 分量
                (pixel_origin[..., 0] + jitter[..., 0] - cx) / fx,
                (pixel_origin[..., 1] + jitter[..., 1] - cy) / fy,
                #固定 Z 分量为 1
                torch.ones(
                    (spp, base_batch, h, w),
                    dtype=rays_dir.dtype,
                    device=rays_dir.device,
                ),
            ),
            dim=-1,
        )
        dirs = (
            torch.nn.functional.normalize(dirs, dim=-1)
            .reshape(spp * base_batch, h, w, 3)
            .contiguous()
        )
        origins = (
            rays_ori.unsqueeze(0)
            .expand(spp, *rays_ori.shape)
            .reshape(spp * base_batch, h, w, 3)
            .contiguous()
        )
        return origins, dirs, spp

    @staticmethod
    def _average_spp_output(
        value: torch.Tensor, spp: int, base_batch: int
    ) -> torch.Tensor:
        if spp == 1:
            return value
        return value.reshape(spp, base_batch, *value.shape[1:]).mean(dim=0)

    def build_acc(self, gaussians, rebuild=True):
        with torch.cuda.nvtx.range(f"build-bvh-full-build-{rebuild}"):
            allow_bvh_update = (
                self.conf.render.max_consecutive_bvh_update > 1
            ) and not self.conf.render.particle_kernel_density_clamping
            rebuild_bvh = (
                rebuild
                or self.conf.render.particle_kernel_density_clamping
                or self.num_update_bvh >= self.conf.render.max_consecutive_bvh_update
            )
            self.tracer_wrapper.build_bvh(
                gaussians.positions.view(-1, 3).contiguous(),
                gaussians.rotation_activation(gaussians.rotation)
                .view(-1, 4)
                .contiguous(),
                gaussians.scale_activation(gaussians.scale).view(-1, 3).contiguous(),
                gaussians.density_activation(gaussians.density)
                .view(-1, 1)
                .contiguous(),
                rebuild_bvh,
                allow_bvh_update,
            )
            self.num_update_bvh = 0 if rebuild_bvh else self.num_update_bvh + 1
            self.invalidate_mog_pack_cache()

    def render(
        self,
        gaussians,
        gpu_batch: Batch,
        train=False,
        frame_id=0,
        sh_indirect: bool = False,
    ):
        #gaussians 一般是 MixtureOfGaussians
        num_gaussians = gaussians.num_gaussians
        with torch.cuda.nvtx.range(f"model.forward({num_gaussians} gaussians)"):
            if self.frame_timer is not None:
                self.frame_timer.start()
            #gpu_batch 是已经搬到 GPU 上的 Batch 数据类，表示当前这一帧/这一批相机观测
            base_batch = gpu_batch.rays_dir.shape[0]
            spp = self._get_spp(train)
            if spp > 1 and not self._can_expand_spp(gpu_batch):
                self._warn_spp_fallback(
                    "batch has no supported pinhole intrinsics/pixel_coords"
                )
                spp = 1

            spp_chunk = self._get_spp_chunk(spp)
            h, w = gpu_batch.rays_dir.shape[1:3]
            spp_jitter = None
            if spp > 1:
                spp_jitter = self._make_spp_jitter(
                    spp,
                    h,
                    w,
                    gpu_batch.rays_dir.device,
                    gpu_batch.rays_dir.dtype,
                    frame_id,
                )
            spp_config = (
                bool(train),
                int(spp),
                int(spp_chunk),
                int(base_batch),
                int(h),
                int(w),
            )
            if spp_config not in self._logged_spp_configs:
                num_chunks = (spp + spp_chunk - 1) // spp_chunk
                rich_logger.warning(
                    f"PTIR effective SPP: train={train} spp={spp} spp_chunk={spp_chunk} "
                    f"chunks={num_chunks} base_batch={base_batch} chunk_ray_batch<={spp_chunk * base_batch} "
                    f"resolution={w}x{h}"
                )
                self._logged_spp_configs.add(spp_config)

            """
            训练时	['color']	['color']	只关心颜色 Loss，法线/深度只在验证时用
            推理/验证时	['color', 'depth', 'normal']	['color', 'depth', 'normal']	需要多模态输出用于分析或可视化
            调试时	['color', 'depth', 'normal', 'variance']	['color', 'depth', 'normal', 'variance']	调试时需要所有中间结果
            """
            final_output_keys, accumulation_keys = self._train_output_key_sets(train)

            accumulated_outputs = {}
            mog_visibility = None
            total_spp = 0

            device = gpu_batch.rays_dir.device
            environment = gaussians.get_environment()
            if environment is None:
                environment = torch.empty(
                    (0, 0, 4), dtype=torch.float32, device=device
                )
            alias_table = getattr(gaussians, "environment_alias_table", None)
            if alias_table is None:
                environment_alias_table = torch.empty(
                    0, dtype=torch.float32, device=device
                )
            else:
                environment_alias_table = (
                    torch.concat(
                        [
                            alias_table.prob.reshape(
                                1, alias_table.height, alias_table.width
                            ),
                            alias_table.alias.reshape(
                                1, alias_table.height, alias_table.width
                            ).to(dtype=torch.float32),
                            alias_table.pdf.reshape(
                                1, alias_table.height, alias_table.width
                            ),
                        ],
                        dim=0,
                    )
                    .to(device=device)
                    .contiguous()
                )
            light_buffers = LightBuffers.from_batch(gpu_batch, device=device)
            ray_to_world = gpu_batch.T_to_world.contiguous()
            use_mog_pack_cache = self._cache_mog_pack_enabled(train)
            mog_pack = self._get_mog_pack(gaussians, use_mog_pack_cache)
            mog_pos = mog_pack["mog_pos"]
            mog_rot = mog_pack["mog_rot"]
            mog_scl = mog_pack["mog_scl"]
            mog_dns = mog_pack["mog_dns"]
            mog_sph = mog_pack["mog_sph"]
            mog_snrm = mog_pack["mog_snrm"]
            mog_malb = mog_pack["mog_malb"]
            mog_mrgh = mog_pack["mog_mrgh"]
            mog_mmet = mog_pack["mog_mmet"]
            render_opts = int(
                Tracer.RenderOpts.INDIRECT
                if sh_indirect
                else Tracer.RenderOpts.DEFAULT
            )
            sph_degree = gaussians.n_active_features
            min_transmittance = self.conf.render.min_transmittance
            max_bounces = self.conf.render.get("max_bounces", 3)

            #将多个“渲染块（chunk）”的输出合并成一个累积结果，并正确处理不同 SPP 下的加权平均。
            def accumulate_output(name: str, value: torch.Tensor) -> None:
                if accumulation_keys is not None and name not in accumulation_keys:
                    return
                averaged = self._average_spp_output(value, chunk_spp, base_batch)
                weighted = averaged * chunk_spp
                previous = accumulated_outputs.get(name)
                accumulated_outputs[name] = (
                    weighted if previous is None else previous + weighted
                )

            try:
                if use_mog_pack_cache:
                    Tracer._Autograd._active_pack_cache = mog_pack

                for spp_start in range(0, spp, spp_chunk):
                    chunk_spp = min(spp_chunk, spp - spp_start)
                    #默认一条射线打像素中心。SPP>1 时要在同一像素里打多条略不同方向的射线，再平均。抗锯齿
                    chunk_jitter = (
                        None
                        if spp_jitter is None
                        else spp_jitter[spp_start : spp_start + chunk_spp]
                    )
                    rays_ori, rays_dir, chunk_spp = self._expand_rays_for_spp(
                        gpu_batch,
                        chunk_spp,
                        frame_id + spp_start,
                        jitter=chunk_jitter,
                    )

                    (
                            chunk_pred_rgb,                      # 预测的RGB颜色 [B, H, W, 3]
                            chunk_pred_opacity,                  # 不透明度/透射率
                            chunk_pred_dist,                     # 深度/距离
                            chunk_pred_dist_second_moment,       # 深度的二阶矩（方差相关）
                            chunk_pred_distortion,               # 畸变/扭曲程度
                            chunk_pred_normals,                  # 法线
                            chunk_pred_shadingnormal,            # 着色法线（用于光照计算）
                            chunk_pred_material,                 # 材质属性
                            chunk_hits_count,                    # 命中次数统计
                            chunk_mog_visibility,                # MoG（混合高斯）可见性
                            chunk_pred_pbr,                      # PBR渲染结果
                            chunk_pred_light,                    # 光照信息
                            chunk_pbr_components,               # PBR各分量分解
                    ) = Tracer._Autograd.apply(
                        self.tracer_wrapper,
                        frame_id + spp_start,
                        ray_to_world,
                        rays_ori,
                        rays_dir,
                        mog_pos,
                        mog_rot,
                        mog_scl,
                        mog_dns,
                        mog_sph,
                        mog_snrm,
                        mog_malb,
                        mog_mrgh,
                        mog_mmet,
                        environment,
                        environment_alias_table,
                        light_buffers,
                        render_opts,
                        sph_degree,
                        min_transmittance,
                        max_bounces,
                    )

                    chunk_pred_rgb, chunk_pred_opacity = gaussians.background(
                        ray_to_world,
                        rays_dir,
                        chunk_pred_rgb,
                        chunk_pred_opacity,
                        train,
                    )

                    for name, value in (
                        ("pred_rgb", chunk_pred_rgb),
                        ("pred_opacity", chunk_pred_opacity),
                        ("pred_dist", chunk_pred_dist),
                        ("pred_depth_second_moment", chunk_pred_dist_second_moment),
                        ("pred_depth_distortion", chunk_pred_distortion),
                        ("pred_normals", chunk_pred_normals),
                        ("pred_shadingnormal", chunk_pred_shadingnormal),
                        ("pred_material", chunk_pred_material),
                        ("hits_count", chunk_hits_count),
                        ("pred_pbr", chunk_pred_pbr),
                        ("pred_light", chunk_pred_light),
                        ("pbr_components", chunk_pbr_components.detach()),
                    ):
                        accumulate_output(name, value)

                    mog_visibility = (
                        chunk_mog_visibility
                        if mog_visibility is None
                        else torch.maximum(mog_visibility, chunk_mog_visibility)
                    )
                    total_spp += chunk_spp
            finally:
                Tracer._Autograd._active_pack_cache = None

            if self.frame_timer is not None:
                self.frame_timer.end()

            accumulated_outputs = {
                name: accumulated / total_spp
                for name, accumulated in accumulated_outputs.items()
            }
            pbr_components = accumulated_outputs.pop("pbr_components", None)
            if pbr_components is not None:
                accumulated_outputs["pred_direct"] = pbr_components[..., 0, :]
                accumulated_outputs["pred_indirect"] = pbr_components[..., 1, :]

            self._normalize_depth_output(accumulated_outputs)
            # https://github.com/fudan-zvg/IRGS/blob/main/gaussian_renderer/__init__.py#L233
            self._postprocess_pbr_outputs(accumulated_outputs)

        if self.frame_timer is not None:
            self.timings["forward_render"] = self.frame_timer.timing()

        output_keys = (
            "pred_rgb",
            "pred_opacity",
            "pred_dist",
            "pred_depth_second_moment",
            "pred_depth_distortion",
            "pred_normals",
            "pred_shadingnormal",
            "pred_material",
            "pred_pbr",
            "pred_light",
            "pred_direct",
            "pred_indirect",
            "hits_count",
        )
        output_values = {key: accumulated_outputs.get(key) for key in output_keys}
        if output_values["pred_normals"] is not None:
            output_values["pred_normals"] = torch.nn.functional.normalize(
                output_values["pred_normals"], dim=3
            )
        outputs = {
            key: value
            for key, value in output_values.items()
            if value is not None
            and (final_output_keys is None or key in final_output_keys)
        }
        outputs["frame_time_ms"] = (
            self.frame_timer.timing() if self.frame_timer is not None else 0.0
        )
        outputs["mog_visibility"] = mog_visibility
        return post_processing(outputs, gpu_batch, self.visualize_lights)

    def trace_atmosphere(self, ray_ori, ray_dir, buffers, frame_number: int = 0):
        """Volume path trace: voxel Woodcock + 1D gas + optional GS/mesh surfaces."""
        from .atmosphere import trace_atmosphere as _trace_atmosphere

        return _trace_atmosphere(self.tracer_wrapper, ray_ori, ray_dir, buffers, frame_number)
