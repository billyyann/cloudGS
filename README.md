# cloudGS

初步实现云物理仿真。

GPU 上用拷贝版 PTIR 做大气短波路径追踪：体素云（Woodcock）、高斯地面、三角 mesh 补洞，并与 htrdr `DZVAR` 俯视结果对照。

## 布局

- `gpu/threedgptir_tracer/` — OptiX 大气路径追踪
- `tools/` — 场景与俯视渲染脚本
- `scenes/mountain_gs/` — 山地/沙漠俯视对照
- `docs/` — 辐射传输笔记

## 渲染

```bash
python tools/render_mountain_gs_gpu.py --res 512 --spp 16 --max-bounces 1024
```
