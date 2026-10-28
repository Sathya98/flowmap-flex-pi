# Patched ray-tracing camera shader

`rt/` is SAPIEN 3.0.0b1's `vulkan_shader/rt` (the camera shader RoboTwin renders with), copied
unchanged except for one guard in `rt/camera.rchit` (marked "RoboTwin patch").

On a surface that reflects nothing (base color (0, 0, 0), no specular, opaque: a black pad, or
the pure-black pixels of some background textures), the original shader samples no bounce and
traces the next ray with a zero direction. On GPUs without RT cores (A100, H100) that traversal
sometimes never ends: the render hangs or the GPU reports "device lost". The guard absorbs the
path there instead. The later bounces of such a path already had zero weight, so they added
nothing to the image before the patch either.
