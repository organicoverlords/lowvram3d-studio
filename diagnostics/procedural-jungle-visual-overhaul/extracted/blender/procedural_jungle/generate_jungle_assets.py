from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Vector

SEED_DEFAULT = 20260804
WORLD_SIZE = 1000.0


def args_after_double_dash() -> list[str]:
    argv = list(__import__('sys').argv)
    return argv[argv.index('--') + 1:] if '--' in argv else []


def reset_scene() -> None:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.images):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def fbm(x: float, y: float, seed: int, octaves: int = 6) -> float:
    value = 0.0
    amplitude = 1.0
    frequency = 0.0032
    normalizer = 0.0
    for octave in range(octaves):
        phase = seed * 0.1137 + octave * 2.173
        value += amplitude * (
            math.sin((x + phase * 47.0) * frequency) * 0.52
            + math.cos((y - phase * 29.0) * frequency * 1.31) * 0.34
            + math.sin((x * 0.73 + y) * frequency * 0.67 + phase) * 0.28
        )
        normalizer += amplitude
        amplitude *= 0.51
        frequency *= 2.03
    return value / max(normalizer, 1e-6)


RIVER_POINTS = [
    Vector((-470.0, 238.0, 53.0)),
    Vector((-390.0, 205.0, 49.0)),
    Vector((-300.0, 160.0, 45.0)),
    Vector((-205.0, 128.0, 41.0)),
    Vector((-108.0, 102.0, 38.0)),
    Vector((-15.0, 68.0, 35.0)),
    Vector((78.0, 26.0, 32.0)),
    Vector((150.0, -26.0, 29.0)),
    Vector((210.0, -84.0, 27.0)),
    Vector((246.0, -124.0, -14.0)),
    Vector((300.0, -178.0, -18.0)),
    Vector((385.0, -230.0, -21.0)),
    Vector((470.0, -268.0, -23.0)),
]
WATERFALL_SEGMENT = 8
POOL_CENTER = Vector((294.0, -176.0, -18.0))


def closest_polyline_xy(x: float, y: float) -> tuple[float, float, int, float]:
    point = Vector((x, y))
    best = (1e30, 0.0, 0, 0.0)
    cumulative = 0.0
    total = sum((Vector((b.x, b.y)) - Vector((a.x, a.y))).length for a, b in zip(RIVER_POINTS, RIVER_POINTS[1:]))
    for index, (a3, b3) in enumerate(zip(RIVER_POINTS, RIVER_POINTS[1:])):
        a = Vector((a3.x, a3.y))
        b = Vector((b3.x, b3.y))
        ab = b - a
        length = max(ab.length, 1e-6)
        t = max(0.0, min(1.0, (point - a).dot(ab) / (length * length)))
        nearest = a + ab * t
        distance = (point - nearest).length
        progress = (cumulative + length * t) / max(total, 1e-6)
        if distance < best[0]:
            best = (distance, progress, index, t)
        cumulative += length
    return best


def river_surface_z(segment: int, t: float) -> float:
    if segment == WATERFALL_SEGMENT:
        return RIVER_POINTS[segment].z if t < 0.46 else RIVER_POINTS[segment + 1].z
    return RIVER_POINTS[segment].z * (1.0 - t) + RIVER_POINTS[segment + 1].z * t


def terrain_height(x: float, y: float, seed: int) -> float:
    rolling = 17.0 * fbm(x, y, seed)
    ridges = 8.0 * math.sin(x * 0.0051 + 0.8) + 6.0 * math.cos(y * 0.0042 - 0.6)
    valley_sides = 0.000075 * (x * x + y * y)
    z = 17.0 + rolling + ridges + valley_sides

    distance, _, segment, t = closest_polyline_xy(x, y)
    channel = max(0.0, 1.0 - distance / 48.0)
    bank = max(0.0, 1.0 - distance / 86.0)
    water_z = river_surface_z(segment, t)
    if segment == WATERFALL_SEGMENT:
        carved = water_z - 5.0
        z = z * (1.0 - channel * 0.97) + carved * channel * 0.97
    else:
        carved = water_z - 4.8 - 0.035 * distance
        z = z * (1.0 - channel * 0.94) + carved * channel * 0.94
    z -= bank * (1.0 - channel) * 3.5ed = waa.y))).le - charnel))3 in zip(RIVER_POINTS, RIVER_POINTS[1:]))
    for index, (a3, b3) in enumerate(zip(RIVER_456092) 0 RIVERPOIN * x + yo* y)
ύοÒ), 4) + carnt == WATERFAL1.0 (zip(RIVPOINrce_roovave 1(a3, b3) d== WAWCAAINrce_roovavofPath = Join-Path $ProofRoot 'gameplay_runtime_proof.json'
if (-not (Test-Path -LiteralPath $RuntimeProofPath)) { throw 'Runtime proof JSON was not written' }
$RuntimeProof = Get-Content -LiteralPath $RuntimeProofPath -Raw | ConvertFrom-Json
iegcfPath 7f = Get-Content -LiteralPt -Lite
   --<43gtydtut.nes),
   ntime_proof.jite
   --<43gtydtut), 4) + carnt == WAm __future__ import an') -Command {
    & $ WAm ___future_d {)
        $LASNouereoy_roova___fuv pfuture_d {)
        $LASNouereoy_r= sum(r$ WAm ___future_d {)
        $LaFbx
   fore-tuer {adPROVEN'
    averag     $Lx
   ffre-tuer {adP  chan ffre(t  averag     $Lx
   ffrnonl index, t)
     $LASNou.dat]
   f_rnonno)
     $'
    proW  y: float, seed: int.0proM  y: t(rig.get('bones',  pr
- MagicMusic(4 + carnonno)er {adGblniver_surface_uutnenonno           errors.append(f"{record['file']} is excessively black: {record['near_black_fraction']}")

    average_green = statistics.fmean(recort]}")

    aer than two re_d {)
Ρ•\LM:
          'iter two re_d 
  Ρ•\