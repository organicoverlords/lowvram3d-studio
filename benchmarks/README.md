# Creature benchmark

`raccoon_creature_benchmark.jpg` is the first end-to-end image benchmark.

Classification: **Creature**, not vehicle.

Expected semantic policy:

- keep the organic body, head, limbs, paws, and tail continuous for deformation;
- separate rifle, backpack, armour, pouches, straps, and rigid equipment where geometry permits;
- do not claim semantic separation unless the generated mesh contains separable geometry;
- use 2K gameplay textures first;
- target approximately 40–50k triangles;
- produce two LODs, PBR maps, a template creature rig, previews, and validation;
- mark skinning/deformation as requiring visual review.

In 3D Gen Studio import this image and choose **LowVRAM One-Click — Creature**.
