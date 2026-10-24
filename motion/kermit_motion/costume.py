"""Frog hand-rod puppet costume: felt head around the hand, petal collar, and a dangling body.

Head: built in the puppet frame (x forward, y left, z up) with its origin at the back of the mouth.
The skull (upper jaw) and the lower jaw are separate cosmetic bodies on hinges; each hand profile
drives their angles from its own kinematics (fingers -> skull, thumb / lower jaw -> jaw), so the
felt always shows what the real hand inside is doing.

Body: a free torso softly welded to a mocap anchor at the neck, with damped ball-jointed noodle
arms and legs. It is simulated (gravity + inertia), so it swings and follows through whenever the
head moves. The anchor follows the head's position and part of its heading, not its pitch or roll:
the body hangs from the neck like a real puppet's.
"""
from __future__ import annotations

import mujoco
import numpy as np

GREEN = (0.30, 0.68, 0.14, 1)
GREEN_SHADE = (0.25, 0.58, 0.11, 1)
COLLAR = (0.58, 0.78, 0.14, 1)
MOUTH = (0.55, 0.06, 0.12, 1)
MOUTH_DEEP = (0.25, 0.02, 0.05, 1)
WHITE = (0.97, 0.97, 0.94, 1)
BLACK = (0.03, 0.03, 0.03, 1)

ELL = mujoco.mjtGeom.mjGEOM_ELLIPSOID
CAP = mujoco.mjtGeom.mjGEOM_CAPSULE
SPH = mujoco.mjtGeom.mjGEOM_SPHERE

UPPER_MAX = np.deg2rad(7.0)    # skull lift at full mouth open (the upper jaw moves only a little)
LOWER_MAX = np.deg2rad(40.0)   # lower jaw drop at full mouth open


def mat2quat(R) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).ravel())
    return q.tolist()


def _frame(x, z_hint):
    x = np.asarray(x, float)
    x /= np.linalg.norm(x)
    y = np.cross(z_hint, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def g(body, name, gtype, size, pos, rgba, quat=(1, 0, 0, 0), mat=None, **kw):
    geom = body.add_geom(name=name, type=gtype, size=list(size) + [0] * (3 - len(size)), pos=list(pos),
                         quat=list(quat), rgba=list(rgba), contype=0, conaffinity=0, group=1, density=0, **kw)
    if mat:
        geom.material = mat
    return geom


def _materials(spec):
    if "felt" not in {mat.name for mat in spec.materials}:
        spec.add_material(name="felt", rgba=GREEN, specular=0.02, shininess=0.02, reflectance=0)
        spec.add_material(name="felt_shade", rgba=GREEN_SHADE, specular=0.02, shininess=0.02)
        spec.add_material(name="collar", rgba=COLLAR, specular=0.02, shininess=0.02)
        spec.add_material(name="mouth", rgba=MOUTH, specular=0.1, shininess=0.1)
        spec.add_material(name="eye", rgba=WHITE, specular=0.25, shininess=0.4)


def add_head(spec: mujoco.MjSpec, parent: mujoco.MjsBody, pos, quat, prefix: str = "puppet"):
    """Felt head on ``parent`` at the mouth hinge. Returns (head, skull_joint_name, jaw_joint_name)."""
    _materials(spec)
    head = parent.add_body(name=f"{prefix}_head", pos=list(pos), quat=list(quat))

    def hinged(name, axis):
        b = head.add_body(name=name, mass=0.01, inertia=[1e-6] * 3, explicitinertial=True)
        b.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis)
        return b

    # skull = upper jaw: wide flat oval (18 cm wide), eyes on top at the front
    skull = hinged(f"{prefix}_skull", [0, -1, 0])  # + = lift
    g(skull, "skull", ELL, [0.092, 0.092, 0.046], [0.022, 0, 0.032], GREEN, mat="felt")
    g(skull, "palate", ELL, [0.083, 0.083, 0.004], [0.02, 0, -0.0115], MOUTH, mat="mouth")
    g(skull, "throat", ELL, [0.03, 0.06, 0.03], [-0.04, 0, 0.0], MOUTH_DEEP)
    for s, side in [(1, "l"), (-1, "r")]:
        # ping-pong-ball eyes sitting on top at the front, half sunk into the felt
        c = np.array([0.062, 0.036 * s, 0.07])
        r = 0.029
        g(skull, f"eye_{side}", SPH, [r], c, WHITE, mat="eye")
        # pupils converge on a point ~0.45 m in front of the face, tucked up under the lid line
        focus = np.array([0.5, 0, 0.13])
        d = focus - c
        d /= np.linalg.norm(d)
        R = _frame(d, np.array([0, 0, 1.0]))
        g(skull, f"pupil_{side}", ELL, [0.003, 0.0115, 0.0072], c + d * (r - 0.0018), BLACK, quat=mat2quat(R))
        # lid line: a thin black disc slicing the ball, so only its rim shows, tilted down to the nose
        n = np.array([0.25, -0.2 * s, 1.0])
        n /= np.linalg.norm(n)
        h = 0.007  # height of the line above the ball's centre
        rr = np.sqrt(r * r - h * h) + 0.0007
        lid_R = _frame(n, np.array([1.0, 0, 0]))
        g(skull, f"lid_line_{side}", ELL, [0.0012, rr, rr], c + n * h, BLACK, quat=mat2quat(lid_R))

    # lower jaw: hinged at the back of the mouth, no tongue or teeth
    jaw = hinged(f"{prefix}_jaw", [0, 1, 0])  # + = drop
    g(jaw, "jaw", ELL, [0.088, 0.088, 0.03], [0.026, 0, -0.022], GREEN, mat="felt")
    g(jaw, "mouth_floor", ELL, [0.08, 0.08, 0.004], [0.022, 0, 0.0035], MOUTH, mat="mouth")

    # back of the head / neck: covers the wrist and flange behind the skull
    g(head, "nape", ELL, [0.075, 0.062, 0.046], [-0.062, 0, -0.006], GREEN, mat="felt")
    return head, f"{prefix}_skull", f"{prefix}_jaw"


# ---------------------------------------------------------------------------------------------------
# dangling body

TORSO_LEN = 0.11


def _limb(spec, parent, name, pos, segs, radius, axis_dir, damping, stiffness):
    """Chain of capsule segments hanging along -z from ``pos``, ball-jointed. Returns the last body."""
    body = parent
    p = list(pos)
    for i, length in enumerate(segs):
        b = body.add_body(name=f"{name}{i}", pos=p, mass=0.004 + 0.004 * radius / 0.01, inertia=[2e-6] * 3,
                          explicitinertial=True)
        b.add_joint(name=f"{name}{i}", type=mujoco.mjtJoint.mjJNT_BALL, damping=damping, stiffness=stiffness,
                    armature=2e-5)
        d = np.asarray(axis_dir if i == 0 else (0, 0, -1), float)
        d /= np.linalg.norm(d)
        g(b, f"{name}{i}_g", CAP, [radius], [0, 0, 0], GREEN, mat="felt",
          fromto=list(np.r_[[0, 0, 0], d * length]))
        body, p = b, list(d * length)
    return body, p


def _webbed(body, name, pos, along, n, spread, length, width):
    """Flat webbed hand / foot: a felt pad with n digits fanned around ``along`` (body frame)."""
    along = np.asarray(along, float)
    along /= np.linalg.norm(along)
    side = np.cross([0, 0, 1.0], along)
    side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([0, 1.0, 0])
    up = np.cross(along, side)
    pos = np.asarray(pos, float)
    g(body, f"{name}_pad", ELL, [length * 0.55, length * 0.6, width * 0.9], pos + along * length * 0.35, GREEN,
      mat="felt", quat=mat2quat(np.column_stack([along, side, up])))
    for k in range(n):
        a = spread * (k - (n - 1) / 2) / max(n - 1, 1)
        d = np.cos(a) * along + np.sin(a) * side
        g(body, f"{name}_d{k}", CAP, [width], [0, 0, 0], GREEN, mat="felt",
          fromto=list(np.r_[pos + along * length * 0.3, pos + along * length * 0.3 + d * length]))


def add_body(spec: mujoco.MjSpec, prefix: str = "puppet"):
    """Mocap neck anchor + free torso (soft weld) + noodle arms and legs. Returns the anchor body name."""
    _materials(spec)
    w = spec.worldbody
    anchor = w.add_body(name=f"{prefix}_anchor", mocap=True)
    torso = w.add_body(name=f"{prefix}_torso", mass=0.06, inertia=[1.2e-4, 1.2e-4, 3e-5], explicitinertial=True)
    torso.add_freejoint(name=f"{prefix}_torso")
    # skinny torso hanging from the neck, slightly pear shaped
    g(torso, "torso", ELL, [0.03, 0.036, TORSO_LEN / 2], [0.0, 0, -TORSO_LEN / 2 - 0.01], GREEN, mat="felt")
    g(torso, "neck", CAP, [0.024], [0, 0, 0], GREEN, mat="felt", fromto=[0, 0, -0.01, 0, 0, 0.012])
    g(torso, "belly", ELL, [0.032, 0.038, 0.03], [0.004, 0, -TORSO_LEN + 0.012], GREEN, mat="felt")
    # petal collar: 11 points around the neck, pointing outward and down
    n = 11
    for i in range(n):
        a = 2 * np.pi * i / n
        radial = np.array([np.cos(a), np.sin(a), 0.0])
        d = radial * np.cos(np.deg2rad(32)) - np.array([0, 0, 1.0]) * np.sin(np.deg2rad(32))
        tangent = np.cross([0, 0, 1.0], radial)
        normal = np.cross(d, tangent)
        R = np.column_stack([d, tangent, normal])
        S = np.diag([0.052, 0.036, 0.004])  # petal length, base width, thickness
        verts = (R @ S @ np.array([[0, -0.5, 0], [0, 0.5, 0], [1, 0, 0], [0, -0.5, 1], [0, 0.5, 1],
                                   [1, 0, 1]]).T).T
        spec.add_mesh(name=f"collar_petal{i}", uservert=verts.ravel().tolist())
        torso.add_geom(name=f"collar{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"collar_petal{i}",
                       pos=list(radial * 0.026 + np.array([0, 0, -0.006])), material="collar", contype=0,
                       conaffinity=0, group=1, density=0)
    spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=f"{prefix}_anchor", name2=f"{prefix}_torso",
                      objtype=mujoco.mjtObj.mjOBJ_BODY, solref=[0.06, 0.6], data=[0] * 3 + [0] * 3 + [1, 0, 0, 0]
                      + [1])
    # noodle arms from the shoulders, long legs from the hips
    for s, side in [(1, "l"), (-1, "r")]:
        hand, p = _limb(spec, torso, f"arm_{side}", [0.0, 0.03 * s, -0.025], [0.065, 0.06], 0.0085,
                        (0.15, 0.35 * s, -1), damping=0.0006, stiffness=0.0012)
        _webbed(hand, f"hand_{side}", p, (0.25, 0.25 * s, -1), 4, np.deg2rad(75), 0.03, 0.0042)
        foot, p = _limb(spec, torso, f"leg_{side}", [0.01, 0.017 * s, -TORSO_LEN - 0.005], [0.075, 0.075], 0.0095,
                        (0.1, 0.12 * s, -1), damping=0.0008, stiffness=0.0015)
        _webbed(foot, f"foot_{side}", p, (1, 0.25 * s, -0.25), 3, np.deg2rad(60), 0.04, 0.0055)
    return f"{prefix}_anchor", f"{prefix}_torso"
