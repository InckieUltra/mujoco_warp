# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from math import cos
from typing import Optional

import warp as wp

from mujoco_warp._src import collision_driver
from mujoco_warp._src import constraint
from mujoco_warp._src import derivative
from mujoco_warp._src import island
from mujoco_warp._src import math
from mujoco_warp._src import passive
from mujoco_warp._src import sensor
from mujoco_warp._src import smooth
from mujoco_warp._src import solver
from mujoco_warp._src import util_misc
from mujoco_warp._src.support import xfrc_accumulate
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import TileSet
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.types import vec10f
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _next_position(
  # Model:
  opt_timestep: wp.array(dtype=float),
  jnt_type: wp.array(dtype=int),
  jnt_qposadr: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  # Data in:
  qpos_in: wp.array2d(dtype=float),
  qvel_in: wp.array2d(dtype=float),
  # In:
  qvel_scale_in: float,
  # Data out:
  qpos_out: wp.array2d(dtype=float),
):
  worldid, jntid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  jnttype = jnt_type[jntid]
  qpos_adr = jnt_qposadr[jntid]
  dof_adr = jnt_dofadr[jntid]
  qpos = qpos_in[worldid]
  qpos_next = qpos_out[worldid]
  qvel = qvel_in[worldid]

  if jnttype == JointType.FREE:
    qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])
    qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in

    qpos_new = qpos_pos + timestep * qvel_lin

    qpos_quat = wp.quat(
      qpos[qpos_adr + 3],
      qpos[qpos_adr + 4],
      qpos[qpos_adr + 5],
      qpos[qpos_adr + 6],
    )
    qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5]) * qvel_scale_in

    qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)

    qpos_next[qpos_adr + 0] = qpos_new[0]
    qpos_next[qpos_adr + 1] = qpos_new[1]
    qpos_next[qpos_adr + 2] = qpos_new[2]
    qpos_next[qpos_adr + 3] = qpos_quat_new[0]
    qpos_next[qpos_adr + 4] = qpos_quat_new[1]
    qpos_next[qpos_adr + 5] = qpos_quat_new[2]
    qpos_next[qpos_adr + 6] = qpos_quat_new[3]

  elif jnttype == JointType.BALL:
    qpos_quat = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])
    qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in

    qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)

    qpos_next[qpos_adr + 0] = qpos_quat_new[0]
    qpos_next[qpos_adr + 1] = qpos_quat_new[1]
    qpos_next[qpos_adr + 2] = qpos_quat_new[2]
    qpos_next[qpos_adr + 3] = qpos_quat_new[3]

  else:  # if jnt_type in (JointType.HINGE, JointType.SLIDE):
    qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr] * qvel_scale_in


@wp.kernel
def _next_velocity(
  # Model:
  opt_timestep: wp.array(dtype=float),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  qacc_in: wp.array2d(dtype=float),
  # In:
  qacc_scale_in: float,
  # Data out:
  qvel_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_scale_in * qacc_in[worldid, dofid] * timestep


# TODO(team): kernel analyzer array slice?
@wp.func
def _next_act(
  # Model:
  opt_timestep: float,  # kernel_analyzer: ignore
  actuator_dyntype: int,  # kernel_analyzer: ignore
  actuator_dynprm: vec10f,  # kernel_analyzer: ignore
  actuator_actrange: wp.vec2,  # kernel_analyzer: ignore
  # Data In:
  act_in: float,  # kernel_analyzer: ignore
  act_dot_in: float,  # kernel_analyzer: ignore
  # In:
  act_dot_scale: float,
  clamp: bool,
) -> float:
  # advance actuation
  if actuator_dyntype == DynType.FILTEREXACT:
    tau = wp.max(MJ_MINVAL, actuator_dynprm[0])
    act = act_in + act_dot_scale * act_dot_in * tau * (1.0 - wp.exp(-opt_timestep / tau))
  elif actuator_dyntype == DynType.USER:
    return act_in
  else:
    act = act_in + act_dot_scale * act_dot_in * opt_timestep

  # clamp to actrange
  if clamp:
    act = wp.clamp(act, actuator_actrange[0], actuator_actrange[1])

  return act


@wp.kernel
def _next_activation(
  # Model:
  opt_timestep: wp.array(dtype=float),
  actuator_dyntype: wp.array(dtype=int),
  actuator_actadr: wp.array(dtype=int),
  actuator_actnum: wp.array(dtype=int),
  actuator_actlimited: wp.array(dtype=bool),
  actuator_dynprm: wp.array2d(dtype=vec10f),
  actuator_actrange: wp.array2d(dtype=wp.vec2),
  # Data in:
  act_in: wp.array2d(dtype=float),
  act_dot_in: wp.array2d(dtype=float),
  # In:
  act_dot_scale: float,
  limit: bool,
  # Data out:
  act_out: wp.array2d(dtype=float),
):
  worldid, uid = wp.tid()
  opt_timestep_id = worldid % opt_timestep.shape[0]
  actuator_dynprm_id = worldid % actuator_dynprm.shape[0]
  actuator_actrange_id = worldid % actuator_actrange.shape[0]
  actadr = actuator_actadr[uid]
  actnum = actuator_actnum[uid]
  for j in range(actadr, actadr + actnum):
    act = _next_act(
      opt_timestep[opt_timestep_id],
      actuator_dyntype[uid],
      actuator_dynprm[actuator_dynprm_id, uid],
      actuator_actrange[actuator_actrange_id, uid],
      act_in[worldid, j],
      act_dot_in[worldid, j],
      act_dot_scale,
      limit and actuator_actlimited[uid],
    )
    act_out[worldid, j] = act


@wp.kernel
def _next_time(
  # Model:
  opt_timestep: wp.array(dtype=float),
  # Data in:
  nefc_in: wp.array(dtype=int),
  time_in: wp.array(dtype=float),
  nworld_in: int,
  naconmax_in: int,
  njmax_in: int,
  nacon_in: wp.array(dtype=int),
  ncollision_in: wp.array(dtype=int),
  # Data out:
  time_out: wp.array(dtype=float),
):
  worldid = wp.tid()
  time_out[worldid] = time_in[worldid] + opt_timestep[worldid % opt_timestep.shape[0]]
  nefc = nefc_in[worldid]

  if nefc > njmax_in:
    wp.printf("nefc overflow - please increase njmax to %u\n", nefc)

  if worldid == 0:
    ncollision = ncollision_in[0]
    if ncollision > naconmax_in:
      nconmax = int(wp.ceil(float(ncollision) / float(nworld_in)))
      wp.printf("broadphase overflow - please increase nconmax to %u or naconmax to %u\n", nconmax, ncollision)

    if nacon_in[0] > naconmax_in:
      nconmax = int(wp.ceil(float(nacon_in[0]) / float(nworld_in)))
      wp.printf("narrowphase overflow - please increase nconmax to %u or naconmax to %u\n", nconmax, nacon_in[0])


def _advance(m: Model, d: Data, qacc: wp.array, qvel: Optional[wp.array] = None):
  """Advance state and time given activation derivatives and acceleration."""
  # TODO(team): can we assume static timesteps?

  # advance activations
  wp.launch(
    _next_activation,
    dim=(d.nworld, m.nu),
    inputs=[
      m.opt.timestep,
      m.actuator_dyntype,
      m.actuator_actadr,
      m.actuator_actnum,
      m.actuator_actlimited,
      m.actuator_dynprm,
      m.actuator_actrange,
      d.act,
      d.act_dot,
      1.0,
      True,
    ],
    outputs=[d.act],
  )

  wp.launch(
    _next_velocity,
    dim=(d.nworld, m.nv),
    inputs=[m.opt.timestep, d.qvel, qacc, 1.0],
    outputs=[d.qvel],
  )

  # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
  qvel_in = qvel or d.qvel

  wp.launch(
    _next_position,
    dim=(d.nworld, m.njnt),
    inputs=[m.opt.timestep, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, qvel_in, 1.0],
    outputs=[d.qpos],
  )

  wp.launch(
    _next_time,
    dim=d.nworld,
    inputs=[m.opt.timestep, d.nefc, d.time, d.nworld, d.naconmax, d.njmax, d.nacon, d.ncollision],
    outputs=[d.time],
  )

  wp.copy(d.qacc_warmstart, d.qacc)


@wp.kernel
def _euler_damp_qfrc_sparse(
  # Model:
  opt_timestep: wp.array(dtype=float),
  dof_Madr: wp.array(dtype=int),
  dof_damping: wp.array2d(dtype=float),
  # Out:
  qM_integration_out: wp.array3d(dtype=float),
):
  worldid, tid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  adr = dof_Madr[tid]
  qM_integration_out[worldid, 0, adr] += timestep * dof_damping[worldid % dof_damping.shape[0], tid]


@cache_kernel
def _tile_euler_dense(tile: TileSet):
  @wp.kernel(module="unique", enable_backward=False)
  def euler_dense(
    # Model:
    opt_timestep: wp.array(dtype=float),
    dof_damping: wp.array2d(dtype=float),
    # Data in:
    qM_in: wp.array3d(dtype=float),
    efc_Ma_in: wp.array2d(dtype=float),
    # In:
    adr_in: wp.array(dtype=int),
    # Data out:
    qacc_out: wp.array2d(dtype=float),
  ):
    worldid, nodeid = wp.tid()
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    TILE_SIZE = wp.static(tile.size)

    dofid = adr_in[nodeid]
    M_tile = wp.tile_load(qM_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(dofid, dofid))
    damping_tile = wp.tile_load(dof_damping[worldid % dof_damping.shape[0]], shape=(TILE_SIZE,), offset=(dofid,))
    damping_scaled = damping_tile * timestep
    qm_integration_tile = wp.tile_diag_add(M_tile, damping_scaled)

    Ma_tile = wp.tile_load(efc_Ma_in[worldid], shape=(TILE_SIZE,), offset=(dofid,))
    L_tile = wp.tile_cholesky(qm_integration_tile)
    qacc_tile = wp.tile_cholesky_solve(L_tile, Ma_tile)
    wp.tile_store(qacc_out[worldid], qacc_tile, offset=(dofid))

  return euler_dense


@event_scope
def euler(m: Model, d: Data):
  """Euler integrator, semi-implicit in velocity."""
  # integrate damping implicitly
  if not (m.opt.disableflags & (DisableBit.EULERDAMP | DisableBit.DAMPER)):
    qacc = wp.empty((d.nworld, m.nv), dtype=float)
    if m.is_sparse:
      qM = wp.clone(d.qM)
      qLD = wp.empty((d.nworld, 1, m.nC), dtype=float)
      qLDiagInv = wp.empty((d.nworld, m.nv), dtype=float)
      wp.launch(
        _euler_damp_qfrc_sparse,
        dim=(d.nworld, m.nv),
        inputs=[m.opt.timestep, m.dof_Madr, m.dof_damping],
        outputs=[qM],
      )
      smooth.factor_solve_i(m, d, qM, qLD, qLDiagInv, qacc, d.efc.Ma)
    else:
      for tile in m.qM_tiles:
        wp.launch_tiled(
          _tile_euler_dense(tile),
          dim=(d.nworld, tile.adr.size),
          inputs=[m.opt.timestep, m.dof_damping, d.qM, d.efc.Ma, tile.adr],
          outputs=[qacc],
          block_dim=m.block_dim.euler_dense,
        )
    _advance(m, d, qacc)
  else:
    _advance(m, d, d.qacc)


def _rk_perturb_state(
  m: Model,
  d: Data,
  scale: float,
  qpos_t0: wp.array2d(dtype=float),
  qvel_t0: wp.array2d(dtype=float),
  act_t0: Optional[wp.array] = None,
):
  # position
  wp.launch(
    _next_position,
    dim=(d.nworld, m.njnt),
    inputs=[m.opt.timestep, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, qpos_t0, d.qvel, scale],
    outputs=[d.qpos],
  )

  # velocity
  wp.launch(
    _next_velocity,
    dim=(d.nworld, m.nv),
    inputs=[m.opt.timestep, qvel_t0, d.qacc, scale],
    outputs=[d.qvel],
  )

  # activation
  if m.na and act_t0 is not None:
    wp.launch(
      _next_activation,
      dim=(d.nworld, m.nu),
      inputs=[
        m.opt.timestep,
        m.actuator_dyntype,
        m.actuator_actadr,
        m.actuator_actnum,
        m.actuator_actlimited,
        m.actuator_dynprm,
        m.actuator_actrange,
        act_t0,
        d.act_dot,
        scale,
        False,
      ],
      outputs=[d.act],
    )


@wp.kernel
def _rk_accumulate_velocity_acceleration(
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  qacc_in: wp.array2d(dtype=float),
  # In:
  scale: float,
  # Data out:
  qvel_out: wp.array2d(dtype=float),
  qacc_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  qvel_out[worldid, dofid] += scale * qvel_in[worldid, dofid]
  qacc_out[worldid, dofid] += scale * qacc_in[worldid, dofid]


@wp.kernel
def _rk_accumulate_activation_velocity(
  # Data in:
  act_dot_in: wp.array2d(dtype=float),
  # In:
  scale: float,
  # Data out:
  act_dot_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()
  act_dot_out[worldid, actid] += scale * act_dot_in[worldid, actid]


def _rk_accumulate(
  m: Model,
  d: Data,
  scale: float,
  qvel_rk: wp.array2d(dtype=float),
  qacc_rk: wp.array2d(dtype=float),
  act_dot_rk: Optional[wp.array] = None,
):
  """Computes one term of 1/6 k_1 + 1/3 k_2 + 1/3 k_3 + 1/6 k_4."""
  wp.launch(
    _rk_accumulate_velocity_acceleration,
    dim=(d.nworld, m.nv),
    inputs=[d.qvel, d.qacc, scale],
    outputs=[qvel_rk, qacc_rk],
  )

  if m.na and act_dot_rk is not None:
    wp.launch(
      _rk_accumulate_activation_velocity,
      dim=(d.nworld, m.na),
      inputs=[d.act_dot, scale],
      outputs=[act_dot_rk],
    )


@event_scope
def rungekutta4(m: Model, d: Data):
  """Runge-Kutta explicit order 4 integrator."""
  # RK4 tableau
  A = [0.5, 0.5, 1.0]  # diagonal only
  B = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]

  qpos_t0 = wp.clone(d.qpos)
  qvel_t0 = wp.clone(d.qvel)
  qvel_rk = wp.zeros((d.nworld, m.nv), dtype=float)
  qacc_rk = wp.zeros((d.nworld, m.nv), dtype=float)

  if m.na:
    act_t0 = wp.clone(d.act)
    act_dot_rk = wp.zeros((d.nworld, m.na), dtype=float)
  else:
    act_t0 = None
    act_dot_rk = None

  _rk_accumulate(m, d, B[0], qvel_rk, qacc_rk, act_dot_rk)

  for i in range(3):
    a, b = float(A[i]), B[i + 1]
    _rk_perturb_state(m, d, a, qpos_t0, qvel_t0, act_t0)
    forward(m, d)
    _rk_accumulate(m, d, b, qvel_rk, qacc_rk, act_dot_rk)

  wp.copy(d.qpos, qpos_t0)
  wp.copy(d.qvel, qvel_t0)

  if m.na:
    wp.copy(d.act, act_t0)
    wp.copy(d.act_dot, act_dot_rk)

  _advance(m, d, qacc_rk, qvel_rk)


@event_scope
def implicit(m: Model, d: Data):
  """Integrates fully implicit in velocity."""
  if ~(m.opt.disableflags | ~(DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER)):
    if m.is_sparse:
      qDeriv = wp.empty((d.nworld, 1, m.nM), dtype=float)
      qLD = wp.empty((d.nworld, 1, m.nC), dtype=float)
    else:
      qDeriv = wp.empty(d.qM.shape, dtype=float)
      qLD = wp.empty(d.qM.shape, dtype=float)
    qLDiagInv = wp.empty((d.nworld, m.nv), dtype=float)
    derivative.deriv_smooth_vel(m, d, qDeriv)
    qacc = wp.empty((d.nworld, m.nv), dtype=float)
    smooth.factor_solve_i(m, d, qDeriv, qLD, qLDiagInv, qacc, d.efc.Ma)
    _advance(m, d, qacc)
  else:
    _advance(m, d, d.qacc)


@event_scope
def fwd_position(m: Model, d: Data, factorize: bool = True):
  """Position-dependent computations.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    factorize: Flag to factorize interia matrix.
  """
  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  smooth.camlight(m, d)
  smooth.flex(m, d)
  smooth.tendon(m, d)
  smooth.crb(m, d)
  smooth.tendon_armature(m, d)
  if factorize:
    smooth.factor_m(m, d)
  if m.opt.run_collision_detection:
    collision_driver.collision(m, d)
  constraint.make_constraint(m, d)
  # TODO(team): remove False after island features are more complete
  if False and not (m.opt.disableflags & DisableBit.ISLAND):
    island.island(m, d)
  smooth.transmission(m, d)


@wp.kernel
def _actuator_velocity(
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  moment_rownnz_in: wp.array2d(dtype=int),
  moment_rowadr_in: wp.array2d(dtype=int),
  moment_colind_in: wp.array2d(dtype=int),
  actuator_moment_in: wp.array2d(dtype=float),
  # Data out:
  actuator_velocity_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()

  rownnz = moment_rownnz_in[worldid, actid]
  rowadr = moment_rowadr_in[worldid, actid]

  vel = float(0.0)
  for i in range(rownnz):
    sparseid = rowadr + i
    colind = moment_colind_in[worldid, sparseid]
    vel += actuator_moment_in[worldid, sparseid] * qvel_in[worldid, colind]

  actuator_velocity_out[worldid, actid] = vel


@wp.kernel
def _tendon_velocity(
  # Model:
  ten_J_rownnz: wp.array(dtype=int),
  ten_J_rowadr: wp.array(dtype=int),
  ten_J_colind: wp.array(dtype=int),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  ten_J_in: wp.array2d(dtype=float),
  # Data out:
  ten_velocity_out: wp.array2d(dtype=float),
):
  worldid, tenid = wp.tid()

  velocity = float(0.0)
  rownnz = ten_J_rownnz[tenid]
  rowadr = ten_J_rowadr[tenid]
  for i in range(rownnz):
    sparseid = rowadr + i
    J = ten_J_in[worldid, sparseid]
    if J != 0.0:
      colind = ten_J_colind[sparseid]
      velocity += J * qvel_in[worldid, colind]

  ten_velocity_out[worldid, tenid] = velocity


@event_scope
def fwd_velocity(m: Model, d: Data):
  """Velocity-dependent computations."""
  wp.launch_tiled(
    _actuator_velocity,
    dim=(d.nworld, m.nu),
    inputs=[d.qvel, d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.actuator_moment],
    outputs=[d.actuator_velocity],
    block_dim=m.block_dim.actuator_velocity,
  )

  wp.launch(
    _tendon_velocity,
    dim=(d.nworld, m.ntendon),
    inputs=[m.ten_J_rownnz, m.ten_J_rowadr, m.ten_J_colind, d.qvel, d.ten_J],
    outputs=[d.ten_velocity],
  )

  smooth.com_vel(m, d)
  passive.passive(m, d)
  smooth.rne(m, d)
  smooth.tendon_bias(m, d, d.qfrc_bias)


from . import bezier_util

@wp.func
def fiber_length_curve(length: float, lmin: float = 0.4441, lmax: float = 1.8123, ltrans: float = 0.73) -> float:
  """Normalized muscle length-gain curve."""
  if length < lmin or length > lmax:
    return 0.0
  elif length <= ltrans:
    length = 0.4441 + (0.73 - 0.4441) * (length - lmin) / (ltrans - lmin)
  else:
    length = 0.73 + (1.8123 - 0.73) * (length - ltrans) / (lmax - ltrans)

  if length <= 0.444100:
    return 0.0
  elif length <= 0.587050:
    return bezier_util.calc_bezier(length, 0.444100, 0.469831, 0.471260, 0.478408, 0.484126, 0.587050,
                                   0.000000, -0.000000, -0.000000, 0.020261, 0.040522, 0.405224,)
  elif length <= 0.840000:
    return bezier_util.calc_bezier(length, 0.587050, 0.681707, 0.686966, 0.699613, 0.707002, 0.840000,
                                   0.405224, 0.740633, 0.759267, 0.784267, 0.790633, 0.905224,)
  elif length <= 1.0:
    return bezier_util.calc_bezier(length, 0.840000, 0.939000, 0.944500, 0.952500, 0.955000, 1.000000,
                                   0.905224, 0.990522, 0.995261, 1.000000, 1.000000, 1.000000,)
  elif length <= 1.431150:
    return bezier_util.calc_bezier(length, 1.000000, 1.067500, 1.071250, 1.092808, 1.110615, 1.431150,
                                   1.000000, 1.000000, 1.000000, 0.975000, 0.950000, 0.500000,)
  elif length <= 1.812300:
    return bezier_util.calc_bezier(length, 1.431150, 1.751685, 1.769493, 1.788550, 1.789800, 1.812300, 
                                   0.500000, 0.050000, 0.025000, 0.000000, 0.000000, 0.000000,)
  else:
    return 0.0

@wp.func
def fiber_velocity_curve(V: float, gain_fvmax: float) -> float:
  # y = gain_fvmax - 1.0
  # if V <= -1.0:
  #   FV = 0.0
  # elif V <= 0.0:
  #   FV = (V + 1.0) * (V + 1.0)
  # elif V <= y:
  #   FV = gain_fvmax - (y - V) * (y - V) / wp.max(MJ_MINVAL, y)
  # else:
  #   FV = gain_fvmax
  # return FV

  if V <= -1.0:
    return 0.0
  elif V <= -0.9:
    return bezier_util.calc_bezier(V, -1.000000, -0.975000, -0.962500, -0.937500, -0.925000, -0.900000,
                                   0.000000, -0.000000, -0.000000, 0.002620, 0.005239, 0.010478,)
  elif V <= 0.0:
    return bezier_util.calc_bezier(V, -0.900000, -0.523383, -0.335074, -0.110074, -0.073383, 0.000000,
                                   0.010478, 0.089405, 0.128868, 0.376248, 0.584165, 1.000000,)
  elif V <= 0.009353:
    return bezier_util.calc_bezier(V, 0.000000, 0.004793, 0.005319, 0.006161, 0.006477, 0.009353,
                                   1.000000, 1.027162, 1.030144, 1.042069, 1.051013, 1.132500,)
  elif V <= 0.900:
    return bezier_util.calc_bezier(V, 0.009353, 0.013031, 0.013434, 0.093593, 0.173347, 0.900000,
                                   1.132500, 1.236702, 1.248139, 1.271539, 1.283502, 1.392500,)
  elif V < 1.0:
    return bezier_util.calc_bezier(V, 0.900000, 0.941000, 0.945500, 0.954500, 0.959000, 1.000000,
                                   1.392500, 1.398650, 1.399325, 1.400000, 1.400000, 1.400000)
  else:
    return 1.400

@wp.func
def fiber_velocity_curve_inv(FV: float, gain_fvmax: float) -> float:
    y = gain_fvmax - 1.0
    if FV <= 0.0:
        V = -1.0  # 或更小的V，函数都返回0
    elif FV <= 1.0:
        V = FV ** 0.5 - 1.0
    elif FV < gain_fvmax:
        V = y - ((y * (gain_fvmax - FV)) ** 0.5)
    else:  # FV == gain_fvmax
        V = y
    return V

@wp.func
def fiber_bias(L: float, bias_lmax: float = 1.6, bias_fpmax: float = 1.0) -> float:
  # b = 0.5 * (1.0 + bias_lmax)
  # b = 1.6
  # if L <= 1.0:
  #   bias = 0.0
  # elif L <= b:
  #   x = (L - 1.0) / wp.max(MJ_MINVAL, b - 1.0)
  #   bias = -bias_fpmax * 0.5 * x * x
  #   bias = -bias_fpmax * x * x
  # else:
  #   x = (L - b) / wp.max(MJ_MINVAL, b - 1.0)
  #   bias = -bias_fpmax * (0.5 + x)
  #   bias = -bias_fpmax * (2.86*x + 1.0)
  # return bias

  if L > 1.0 and L <= bias_lmax:
    L = 1.0 + (L - 1.0) / (bias_lmax - 1.0) * 0.7
  elif L > bias_lmax:
    L = 1.7 + (L - bias_lmax)

  if L <= 1.0:
    return 0.0
  elif L <= 1.035:
    return - bezier_util.calc_bezier(L, 1.000000, 1.012250, 1.014875, 1.020125, 1.022750, 1.035000,
                                   0.000000, 0.000000, 0.000000, 0.000525, 0.001050, 0.003500, )
  elif L <= 1.7:
    return - bezier_util.calc_bezier(L, 1.035000, 1.273019, 1.324023, 1.423773, 1.472519, 1.700000,
                                   0.003500, 0.051104, 0.061305, 0.210780, 0.350054, 1.000000, )
  else:
    return - (2.86*L - 3.862)
  

@wp.func
def tendon_force_length_curve(LT: float, scale: float=1.0) -> float:
  # tendon length curve
  # x = LT
  # if LT <= 1.0:
  #   FLT = 0.0
  # elif LT <= 1.02524:
  #   LT -= 1.0
  #   FLT = -9420.2598878091 * (LT**3.0) + 760.9286249079 * (LT**2.0)
  # elif LT <= 1.0490000000:
  #   LT -= 1.02524
  #   FLT = -13559.1456147993 * (LT**3.0) + 644.2638306833 * (LT**2.0) + 20.4081632653 * LT + 0.3333333333
  # # elif LT<=1.049:
  # #   FLT = -5312.4225450119 * (LT**3.0) + 16614.0694676044*(LT**2.0) \
  # #     -17290.8713001730 * LT + 5989.2243775805
  # # elif x <= 1.0252:
  # #   x -= 1.0000
  # #   FLT = (-8605.061423823578)*x**3.0 + (799.6065388345157)*x**2.0 + \
  # #       (0.0)*x + (5.437753723332103e-55)
  # # elif x <= 1.0490:
  # #     x -= 1.0252
  # #     FLT = (-1707.3710957753804)*x**3.0 + (148.03128782259617)*x**2.0 + \
  # #         (23.918378744825436)*x + (0.3710317673645376)
  # else:
  #   FLT = 28.0612 * LT - 28.4361988
  # return FLT

  if LT <= 1.0:
    return 0.0
  elif LT <= 1.014477:
    return bezier_util.calc_bezier(LT, 1.000000, 1.001856, 1.002784, 1.006403, 1.009095, 1.014477,
                                   0.000000, 0.000000, 0.000000, 0.041667, 0.083333, 0.166667,) / scale
  elif LT <= 1.037121:
    return bezier_util.calc_bezier(LT, 1.014477, 1.019860, 1.022551, 1.028212, 1.031182, 1.037121,
                                   0.166667, 0.250000, 0.291667, 0.416667, 0.500000, 0.666667,) / scale
  else:
    return (28.06122449 * LT - 28.4361988) / scale

@wp.func
def fiber_length_deriv(length: float, lmin: float = 0.4441, lmax: float = 1.8123, ltrans: float = 0.73) -> float:
  """Normalized muscle length-gain curve."""
  # if (lmin > length) or (length > lmax):
  #     return 0.0

  # a = 0.5 * (lmin + 1.0)
  # b = 0.5 * (1.0 + lmax)

  # if length <= a:
  #     denom = max(MJ_MINVAL, a - lmin)
  #     return (length - lmin) / (denom * denom)
  # elif length <= 1.0:
  #     denom = max(MJ_MINVAL, 1.0 - a)
  #     return (1.0 - length) / (denom * denom)
  # elif length <= b:
  #     denom = max(MJ_MINVAL, b - 1.0)
  #     return -(length - 1.0) / (denom * denom)
  # else:
  #     denom = max(MJ_MINVAL, lmax - b)
  #     return -(lmax - length) / (denom * denom)

  if length < lmin or length > lmax:
    return 0.0
  elif length <= ltrans:
    length = 0.4441 + (0.73 - 0.4441) * (length - lmin) / (ltrans - lmin)
  else:
    length = 0.73 + (1.8123 - 0.73) * (length - ltrans) / (lmax - ltrans)

  if length <= 0.444100:
    return 0.0
  elif length <= 0.587050:
    return bezier_util.calc_bezier_deriv(length, 0.444100, 0.469831, 0.471260, 0.478408, 0.484126, 0.587050,
                                   0.000000, -0.000000, -0.000000, 0.020261, 0.040522, 0.405224,)
  elif length <= 0.840000:
    return bezier_util.calc_bezier_deriv(length, 0.587050, 0.681707, 0.686966, 0.699613, 0.707002, 0.840000,
                                   0.405224, 0.740633, 0.759267, 0.784267, 0.790633, 0.905224,)
  elif length <= 1.0:
    return bezier_util.calc_bezier_deriv(length, 0.840000, 0.939000, 0.944500, 0.952500, 0.955000, 1.000000,
                                   0.905224, 0.990522, 0.995261, 1.000000, 1.000000, 1.000000,)
  elif length <= 1.431150:
    return bezier_util.calc_bezier_deriv(length, 1.000000, 1.067500, 1.071250, 1.092808, 1.110615, 1.431150,
                                   1.000000, 1.000000, 1.000000, 0.975000, 0.950000, 0.500000,)
  elif length <= 1.812300:
    return bezier_util.calc_bezier_deriv(length, 1.431150, 1.751685, 1.769493, 1.788550, 1.789800, 1.812300, 
                                   0.500000, 0.050000, 0.025000, 0.000000, 0.000000, 0.000000,)
  else:
    return 0.0
  
@wp.func
def fiber_bias_deriv(L: float, bias_lmax: float = 1.6, bias_fpmax: float = 1.0) -> float:
  # b = 0.5 * (1.0 + bias_lmax)
  # b = 1.6
  # denom = max(MJ_MINVAL, b - 1.0)
  # if L <= 1.0:
  #     return 0.0
  # elif L <= b:
  #     x = (L - 1.0) / denom
  #     return -bias_fpmax * 2.0 * x / denom
  # else:
  #     return -2.86 * bias_fpmax / denom

  if L > 1.0 and L <= bias_lmax:
    L = 1.0 + (L - 1.0) / (bias_lmax - 1.0) * 0.7
  elif L > bias_lmax:
    L = 1.7 + (L - bias_lmax)

  if L <= 1.0:
    return 0.0
  elif L <= 1.035:
    return - bezier_util.calc_bezier_deriv(L, 1.000000, 1.012250, 1.014875, 1.020125, 1.022750, 1.035000,
                                   0.000000, 0.000000, 0.000000, 0.000525, 0.001050, 0.003500, )
  elif L <= 1.7:
    return - bezier_util.calc_bezier_deriv(L, 1.035000, 1.273019, 1.324023, 1.423773, 1.472519, 1.700000,
                                   0.003500, 0.051104, 0.061305, 0.210780, 0.350054, 1.000000, )
  else:
    return - 2.86
  
@wp.func
def tendon_force_length_curve_deriv(LT: float, scale: float = 1.0) -> float:
  # x = LT
  # if LT <= 1.0:
  #     return 0.0
  # elif LT <= 1.02524:
  #     LT_ = LT - 1.0
  #     return -28260.7796634273 * (LT_ ** 2.0) + 1521.8572498158 * LT_
  # elif LT <= 1.0490000000:
  #     LT_ = LT - 1.02524
  #     return (-40677.4368443979 * (LT_ ** 2.0)
  #             + 1288.5276613666 * LT_
  #             + 20.4081632653)
  # # elif LT<=1.049:
  # #   return -5312.4225450119*3.0 * (LT**2.0) + 16614.0694676044*2.0*LT \
  # #     -17290.8713001730
  # # elif x <= 1.0252:
  # #   x -= 1.0000
  # #   y = (-25815.18427147073)*x**2.0 + (1599.2130776690315)*x + \
  # #       (0.0)
  # # elif x <= 1.0490:
  # #     x -= 1.0252
  # #     y = (-5122.113287326141)*x**2.0 + (296.06257564519234)*x + \
  # #         (23.918378744825436)
  # else:
  #     return 28.0612
  # # return y

  if LT <= 1.0:
    return 0.0
  elif LT <= 1.014477:
    return bezier_util.calc_bezier_deriv(LT, 1.000000, 1.001856, 1.002784, 1.006403, 1.009095, 1.014477,
                                   0.000000, 0.000000, 0.000000, 0.041667, 0.083333, 0.166667,) / scale
  elif LT <= 1.037121:
    return bezier_util.calc_bezier_deriv(LT, 1.014477, 1.019860, 1.022551, 1.028212, 1.031182, 1.037121,
                                   0.166667, 0.250000, 0.291667, 0.416667, 0.500000, 0.666667,) / scale
  else:
    return 28.06122449 / scale

@wp.func
def partial_func(
  fce: float,
  lce: float,
  lce_h: float,
  Lce0: float,
  LceN: float,
  Lt0: float,
  LtN: float,
  FV: float,
  ctrl_act: float,
  peak_force: float,
  phi: float,
  active_lmin: float,
  active_ltrans: float,
  active_lmax: float,
  passive_lmax: float,
  bias_fpmax: float,
  cos_phi: float,
  tendon_softer: float,
) -> tuple[float, float, float, float, float, float, float, float]:
  dFL_dlceN = fiber_length_deriv(LceN, active_lmin, active_lmax, active_ltrans)
  if dFL_dlceN == 0.0:
    if LceN < 1.0:
      dFL_dlceN = 1e-2
    else:
      dFL_dlceN = -1e-2
  dbias_dlceN = fiber_bias_deriv(LceN, passive_lmax, bias_fpmax)
  dfce_dlce = (-dFL_dlceN * FV * ctrl_act * peak_force + dbias_dlceN * peak_force) / Lce0
  h_over_lce = lce_h / wp.max(MJ_MINVAL, lce)
  dphi_dlce = (-h_over_lce/lce) / wp.sqrt(1.0 - h_over_lce*h_over_lce)
  dcosPHI_dlce = -wp.sin(phi) * dphi_dlce
  dfceAT_dlce = dfce_dlce * cos_phi + fce * dcosPHI_dlce
  dlceAT_dlce = cos_phi - lce * wp.sin(phi) * dphi_dlce
  dfceAT_dlceAT = dfceAT_dlce / wp.max(MJ_MINVAL, dlceAT_dlce)
  # if abs(dfce_dlce) == 0.0:
  #   wp.printf("dfce_dlce == 0.0, fce: %f, dFL_dlceN: %f, V: %f, FV: %f, dbias_dlceN: %f, Lce0: %f, LceN: %f\n", 
  #             fce, dFL_dlceN, V, FV, dbias_dlceN, Lce0, LceN)
  dft_dLt = -tendon_force_length_curve_deriv(LtN, tendon_softer) * peak_force / Lt0
  dft_dlce = -dft_dLt 
  dferr_dlce = dfceAT_dlce - dft_dlce
  
  return dFL_dlceN, dbias_dlceN, dfce_dlce, dft_dLt, dft_dlce, dferr_dlce, dfceAT_dlce, dfceAT_dlceAT

@wp.func
def elastic_tendon_mtu(
  ctrl_act: float,
  length: float,
  velocity: float,
  lengthrange: wp.vec2,
  acc0: float,
  gainprm: vec10f,
  biasprm: vec10f,
) -> float:
  """Compute the force of an elastic tendon muscle-tendon unit (MTU)."""
  range_ = wp.vec2(gainprm[0], gainprm[1])
  peak_force = gainprm[2]
  gain_scale = gainprm[3]
  gain_lmin = gainprm[4]
  gain_lmax = gainprm[5]
  gain_vmax = gainprm[6]
  gain_fvmax = gainprm[8]
  bias_lmax =  gainprm[5]
  bias_fpmax = gainprm[7]
  bias_range_ = wp.vec2(gainprm[0], gainprm[1])

  active_lmin, active_ltrans, active_lmax = 0.4441, 0.73, 1.8123
  passive_lmax = 1.6
  ignore_elastic = 10.0
  if biasprm[4] != gainprm[4]:
  # if False:
    active_lmin, active_ltrans, active_lmax = biasprm[0], biasprm[1], biasprm[2]
    passive_lmax = biasprm[3]
    pennation_angle_opt = biasprm[4]
    ignore_elastic = biasprm[7]
  else:
    active_lmin, active_ltrans, active_lmax = 0.4441, 0.73, 1.8123
    passive_lmax = 1.6
    pennation_angle_opt = 0.0
  # pennation_angle_opt = 0.0
  ctrl_act = max(0.01, ctrl_act)
  beta = 0.01

  # ignore_elastic = 20.0
  tendon_softer = 1.0
  if abs(peak_force - 1575.06) < 0.5 or abs(peak_force - 3115.51) < 0.5 or abs(peak_force - 6194.84) < 0.5:
    # ignore_elastic = 10.0
    # peak_force = peak_force * 0.0
    tendon_softer = 1.0
  peak_force *= 0.5

  # optimum length
  Lce0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])
  Lce0bias = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, bias_range_[1] - bias_range_[0])
  # tendon slack length
  Lt0 = lengthrange[0] - Lce0 * range_[0]
  # wp.printf("lengthrange: %f,%f, normrange: %f,%f, lce0: %f, lt0: %f\n", 
  #           lengthrange[0], lengthrange[1], range_[0], range_[1], Lce0, Lt0)
  lce_h = Lce0 * wp.sin(pennation_angle_opt)

  if length < Lt0:
    # wp.printf("length: %f, Lt0: %f\n, peak_force: %f", length, Lt0, gainprm[2])
    return 0.0

  max_lce = wp.sqrt(length*length + lce_h*lce_h) - 1e-3
  min_lce = lce_h / wp.sin(wp.acos(0.1))

  # normalized length and velocity
  lt_ratio = float(0.01)
  lt = Lt0 * (1.0 + lt_ratio)
  LtN = lt / wp.max(MJ_MINVAL, Lt0)

  lce_w = length - lt
  lce_w = wp.clamp(lce_w, MJ_MINVAL, length)
  lce = wp.sqrt(lce_w * lce_w + lce_h * lce_h)
  lce = max(min_lce, lce)
  lce = min(max_lce, lce)
  lce_w = wp.sqrt(lce * lce - lce_h * lce_h)
  # lce = wp.clamp(lce, MJ_MINVAL, length)
  phi = wp.asin(lce_h / wp.max(MJ_MINVAL, lce))
  cos_phi = wp.cos(phi)

  LceN = lce / wp.max(MJ_MINVAL, Lce0)

  if ignore_elastic < 15.0:
    maxiter = 60
  else:
    maxiter = 0
    lt = Lt0
    LtN = lt / wp.max(MJ_MINVAL, Lt0)
    lce_w = length - lt
    lce_w = wp.clamp(lce_w, MJ_MINVAL, length)
    lce = wp.sqrt(lce_w * lce_w + lce_h * lce_h)
    lce = max(min_lce, lce)
    lce = min(max_lce, lce)
    lce_w = wp.sqrt(lce * lce - lce_h * lce_h)
    lt = length - lce_w
    LceN = lce / wp.max(MJ_MINVAL, Lce0)
    LtN = lt / wp.max(MJ_MINVAL, Lt0)
    phi = wp.asin(lce_h / wp.max(MJ_MINVAL, lce))
    cos_phi = wp.cos(phi)
    vce = velocity * cos_phi

  tol = 1e-6 * peak_force
  cur_iter = int(0)
  error = float(wp.inf)
  error_prev = float(0.0)

  V = float(0.0)

  # length curve
  FL = fiber_length_curve(LceN, active_lmin, active_lmax, active_ltrans)
  FLT = tendon_force_length_curve(LtN, tendon_softer)
  # FLT = fiber_bias(LtN, bias_lmax, bias_fpmax)
  bias = fiber_bias(LceN, passive_lmax, bias_fpmax)

  # velocity curve
  FV = fiber_velocity_curve(V, gain_fvmax)

  gain = -FL*FV
  fce = (gain * ctrl_act + bias) * peak_force + beta * V
  ft = -FLT * peak_force
  fce_at = fce * cos_phi
  error = fce_at - ft

  dFL_dlceN, dbias_dlceN, dfce_dlce, dft_dLt, dft_dlce, dferr_dlce, dfceAT_dlce, dfceAT_dlceAT = partial_func(
    fce, lce, lce_h, Lce0, LceN, Lt0, LtN, FV, ctrl_act, peak_force, phi, active_lmin,
    active_ltrans, active_lmax, passive_lmax, bias_fpmax, cos_phi, tendon_softer,
  )

  # update velocity
  if abs(dfce_dlce + dft_dLt) > MJ_MINVAL and LtN > 1.0:
    vt = velocity * dfceAT_dlceAT / (dfceAT_dlceAT + dft_dLt)
  else:
    vt = velocity
  vce = (velocity - vt) * cos_phi
  V = vce / (wp.max(MJ_MINVAL, Lce0) * gain_vmax)
  FV = fiber_velocity_curve(V, gain_fvmax)

  gain = -FL*FV
  fce = (gain * ctrl_act + bias) * peak_force + beta * V
  ft = -FLT * peak_force
  fce_at = fce * cos_phi
  error = fce_at - ft

  dFL_dlceN, dbias_dlceN, dfce_dlce, dft_dLt, dft_dlce, dferr_dlce, dfceAT_dlce, dfceAT_dlceAT = partial_func(
    fce, lce, lce_h, Lce0, LceN, Lt0, LtN, FV, ctrl_act, peak_force, phi, active_lmin,
    active_ltrans, active_lmax, passive_lmax, bias_fpmax, cos_phi, tendon_softer,
  )

  error_prev = error
  lce_prev = lce

  stuck_cnt = int(0)

  old_FL, old_FV, old_bias, old_FLT = FL, FV, bias, FLT
  old_error = error

  while cur_iter < maxiter:
    # wp.printf("%f\n", peak_force)

    h = float(1.0)
    ls_iter = int(0)

    while(True and ls_iter < 6):
      delta_lce = - h * error_prev / dferr_dlce

      lce = lce_prev + delta_lce
      # lce = max(MJ_MINVAL, lce)
      lce = max(min_lce, lce)
      lce = min(max_lce, lce)
      # lce = wp.clamp(lce, MJ_MINVAL, length)
      # lce = wp.max(lengthrange[0], wp.min(lengthrange[1], lce))

      LceN = lce / wp.max(MJ_MINVAL, Lce0)
      phi = wp.asin(lce_h / wp.max(MJ_MINVAL, lce))
      cos_phi = wp.cos(phi)
      lce_w = wp.sqrt(lce * lce - lce_h * lce_h)
      lt = length - lce_w
      if lt<0:
        wp.printf("lt < 0")
      LtN = lt / wp.max(MJ_MINVAL, Lt0)

      # length curve
      FL = fiber_length_curve(LceN, active_lmin, active_lmax, active_ltrans)
      FLT = tendon_force_length_curve(LtN, tendon_softer)
      # FLT = fiber_bias(LtN, bias_lmax, bias_fpmax)
      bias = fiber_bias(LceN, passive_lmax, bias_fpmax)

      V = vce / (wp.max(MJ_MINVAL, Lce0) * gain_vmax)
      # velocity curve
      FV = fiber_velocity_curve(V, gain_fvmax)
      gain = -FL*FV

      fce = (gain * ctrl_act + bias) * peak_force + beta * V
      fce_at = fce * cos_phi
      ft = -FLT * peak_force

      error = fce_at - ft
      # print("delta lce:", delta_lce, "delta lce:", lce-lce_prev, " -- error:", error)

      # if abs(delta_lce) <= MJ_MINVAL:
      #   wp.printf("delta error: %.15f\n", abs(error) - abs(error_prev))

      if h<1e-9 and abs(error) >= abs(error_prev):
        delta_lce = -wp.sign(delta_lce) * 1e-9
        lce = lce_prev + delta_lce
        # lce = max(MJ_MINVAL, lce)
        # lce = min(length, lce)
        lce = max(min_lce, lce)
        lce = min(max_lce, lce)
        LceN = lce / wp.max(MJ_MINVAL, Lce0)
        phi = wp.asin(lce_h / wp.max(MJ_MINVAL, lce))
        cos_phi = wp.cos(phi)
        lce_w = wp.sqrt(lce * lce - lce_h * lce_h)
        lt = length - lce_w
        if lt<0:
          wp.printf("lt < 0")
        LtN = lt / wp.max(MJ_MINVAL, Lt0)

        # length curve
        FL = fiber_length_curve(LceN, active_lmin, active_lmax, active_ltrans)
        FLT = tendon_force_length_curve(LtN, tendon_softer)
        bias = fiber_bias(LceN, passive_lmax, bias_fpmax)

        V = vce / wp.max(MJ_MINVAL, Lce0) / gain_vmax
        # velocity curve
        FV = fiber_velocity_curve(V, gain_fvmax)
        gain = -FL*FV

        fce = (gain * ctrl_act + bias) * peak_force + beta * V
        fce_at = fce * cos_phi
        ft = -FLT * peak_force

        error = fce_at - ft

      if h<1e-9 or abs(error) < abs(error_prev):
        break

      h *= 0.5

      ls_iter += 1

    cur_iter += 1

    not_change = (abs(error - error_prev) < MJ_MINVAL)

    error_prev = error
    lce_prev = lce

    if wp.abs(error) < tol or (abs(error)<5e-2) or not_change:
      stuck_cnt += 1
    else:
      stuck_cnt = 0
    
    if stuck_cnt > 4:
      break

    dFL_dlceN, dbias_dlceN, dfce_dlce, dft_dLt, dft_dlce, dferr_dlce, dfceAT_dlce, dfceAT_dlceAT = partial_func(
      fce, lce, lce_h, Lce0, LceN, Lt0, LtN, FV, ctrl_act, peak_force, phi, active_lmin,
      active_ltrans, active_lmax, passive_lmax, bias_fpmax, cos_phi, tendon_softer,
    )

    # update velocity
    if abs(dfce_dlce + dft_dLt) > MJ_MINVAL and LtN > 1.0:
      vt = velocity * dfceAT_dlceAT / (dfceAT_dlceAT + dft_dLt)
    else:
      vt = velocity
    vce = (velocity - vt) * cos_phi
    V = vce / wp.max(MJ_MINVAL, Lce0) / gain_vmax
    FV = fiber_velocity_curve(V, gain_fvmax)

    # print(f'iteration {cur_iter}, error {error}, lceN {LceN}, ltN {LtN}, derr_dlce {dferr_dlce}, h {h}, delta_lce {delta_lce}, delta_err {error - error_prev}')

    # if cur_iter > 55:
    # # if cur_iter>0 and abs(peak_force - 521.202) < 1e-2:
    #   wp.printf("iter: %d, error: %.15f, FL: %f, FV: %f, FLT: %f, V: %f, bias: %f, vce: %f, velocity: %f, vmax: %f\n",
    #             cur_iter, error, FL, FV, FLT, V, bias, vce, velocity, gain_vmax)
    #   wp.printf("iter: %d, error: %.15f, fce: %f, h: %.10f, dfce_dlce: %f, dft_dlce: %f, dft_dltN: %f, delta_lce_T: %f, delta_lce: %.15f, delta_err: %.15f, ltN: %f, length: %f, velocity: %f, lceN: %f, lce: %.15f, peak_force: %f\n\n", 
    #             cur_iter, error, fce, h, dfce_dlce, dft_dlce, dft_dLt * Lt0, -h*error/dferr_dlce, delta_lce, error-error_prev, LtN, length, velocity, LceN, lce, peak_force)
    #   # wp.printf("prm0: %f, prm1: %f, prm2: %f, prm3: %f, prm4: %f, prm5: %f, prm6: %f, prm7: %f, prm8: %f, prm9: %f\n\n",
    #   #           gainprm[0], gainprm[1], gainprm[2], gainprm[3], gainprm[4], gainprm[5], gainprm[6], gainprm[7], gainprm[8], gainprm[9])

  if ignore_elastic > 15.0:
    lce = length - Lt0
    lce = max(min_lce, lce)
    lce = min(max_lce, lce)
    LceN = lce / wp.max(MJ_MINVAL, Lce0)
    lce_w = wp.sqrt(lce * lce - lce_h * lce_h)
    phi = wp.asin(lce_h / wp.max(MJ_MINVAL, lce))
    cos_phi = wp.cos(phi)
    vce = velocity * cos_phi
    lt = length - lce_w
    LtN = lt / wp.max(MJ_MINVAL, Lt0)

  # length curve
  FL = fiber_length_curve(LceN, active_lmin, active_lmax, active_ltrans)
  FLT = tendon_force_length_curve(LtN, tendon_softer)
  bias = fiber_bias(LceN, passive_lmax, bias_fpmax)

  V = vce / (wp.max(MJ_MINVAL, Lce0) * gain_vmax)
  # velocity curve
  FV = fiber_velocity_curve(V, gain_fvmax)
  gain = -FL*FV

  # bias = 0.0
  fce = (gain * ctrl_act + bias) * peak_force + beta * V
  fceAT = fce * cos_phi
  ft = -FLT * peak_force
  old_error = error
  error = fceAT - ft

  # if error > 1.0 and ignore_elastic < 15.0:
  #   wp.printf("iter: %d, error: %f, peak force: %f, old_error: %f\n", cur_iter, error, gainprm[2], old_error)
  #   wp.printf("fl: %d, fv: %d, flt: %d, bias: %d\n", int(FL!=old_FL), int(FV!=old_FV), int(FLT!=old_FLT), int(bias!=old_bias))

  # if wp.isnan(((gain * ctrl_act + bias) * peak_force + beta * V) * cos_phi):
  #   # wp.printf("NaN detected in muscle force computation!\n")
  #   wp.printf("iters: %d, error: %.15f, err_prev: %f, fce: %f, h: %.10f, dfce_dlce: %f, dft_dlce: %f, dft_dltN: %f, ltN: %f, length: %f, velocity: %f, lceN: %f, lce: %f, peak_force: %f, phi: %f, cosphi: %f\n\n", 
  #             cur_iter, error, error_prev, fce, h, dfce_dlce, dft_dlce, dft_dLt * Lt0, LtN, length, velocity, LceN, lce, peak_force, phi, cos_phi)
  #   # wp.printf("gain4: %f, bias4: %f\n\n", gainprm[4], biasprm[4])

  return fceAT

@wp.kernel
def _actuator_force(
  # Model:
  na: int,
  opt_timestep: wp.array(dtype=float),
  actuator_dyntype: wp.array(dtype=int),
  actuator_gaintype: wp.array(dtype=int),
  actuator_biastype: wp.array(dtype=int),
  actuator_actadr: wp.array(dtype=int),
  actuator_actnum: wp.array(dtype=int),
  actuator_ctrllimited: wp.array(dtype=bool),
  actuator_forcelimited: wp.array(dtype=bool),
  actuator_actlimited: wp.array(dtype=bool),
  actuator_dynprm: wp.array2d(dtype=vec10f),
  actuator_gainprm: wp.array2d(dtype=vec10f),
  actuator_biasprm: wp.array2d(dtype=vec10f),
  actuator_actearly: wp.array(dtype=bool),
  actuator_ctrlrange: wp.array2d(dtype=wp.vec2),
  actuator_forcerange: wp.array2d(dtype=wp.vec2),
  actuator_actrange: wp.array2d(dtype=wp.vec2),
  actuator_acc0: wp.array2d(dtype=float),
  actuator_lengthrange: wp.array2d(dtype=wp.vec2),
  # Data in:
  act_in: wp.array2d(dtype=float),
  ctrl_in: wp.array2d(dtype=float),
  actuator_length_in: wp.array2d(dtype=float),
  actuator_velocity_in: wp.array2d(dtype=float),
  # In:
  dsbl_clampctrl: int,
  # Data out:
  act_dot_out: wp.array2d(dtype=float),
  actuator_force_out: wp.array2d(dtype=float),
):
  worldid, uid = wp.tid()

  actuator_ctrlrange_id = worldid % actuator_ctrlrange.shape[0]

  ctrl = ctrl_in[worldid, uid]

  if actuator_ctrllimited[uid] and not dsbl_clampctrl:
    ctrlrange = actuator_ctrlrange[actuator_ctrlrange_id, uid]
    ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])
  ctrl_act = ctrl

  act_first = actuator_actadr[uid]
  if na and act_first >= 0:
    act_last = act_first + actuator_actnum[uid] - 1
    dyntype = actuator_dyntype[uid]
    dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]

    if dyntype == DynType.INTEGRATOR:
      act_dot = ctrl
    elif dyntype == DynType.FILTER or dyntype == DynType.FILTEREXACT:
      act = act_in[worldid, act_last]
      act_dot = (ctrl - act) / wp.max(dynprm[0], MJ_MINVAL)
    elif dyntype == DynType.MUSCLE:
      dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]
      act = act_in[worldid, act_last]
      act_dot = util_misc.muscle_dynamics(ctrl, act, dynprm)
    elif dyntype == DynType.USER:
      act_dot = 0.0  # set by act_dyn_callback
    else:  # DynType.NONE
      act_dot = 0.0

    act_dot_out[worldid, act_last] = act_dot

    if actuator_actearly[uid]:
      if dyntype == DynType.INTEGRATOR or dyntype == DynType.NONE:
        act = act_in[worldid, act_last]

      ctrl_act = _next_act(
        opt_timestep[worldid % opt_timestep.shape[0]],
        dyntype,
        dynprm,
        actuator_actrange[worldid % actuator_actrange.shape[0], uid],
        act,
        act_dot,
        1.0,
        actuator_actlimited[uid],
      )
    else:
      ctrl_act = act_in[worldid, act_last]

  length = actuator_length_in[worldid, uid]
  velocity = actuator_velocity_in[worldid, uid]

  # gain
  gaintype = actuator_gaintype[uid]
  gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]

  gain = 0.0
  if gaintype == GainType.FIXED:
    gain = gainprm[0]
  elif gaintype == GainType.AFFINE:
    gain = gainprm[0] + gainprm[1] * length + gainprm[2] * velocity
  elif gaintype == GainType.MUSCLE:
    acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]
    lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]
    gain = util_misc.muscle_gain(length, velocity, lengthrange, acc0, gainprm)
  # GainType.USER: gain stays 0, modified by act_gain_callback

  # bias
  biastype = actuator_biastype[uid]
  biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]

  bias = 0.0  # BiasType.NONE or BiasType.USER (modified by act_bias_callback)
  if biastype == BiasType.AFFINE:
    bias = biasprm[0] + biasprm[1] * length + biasprm[2] * velocity
  elif biastype == BiasType.MUSCLE:
    acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]
    lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]
    bias = util_misc.muscle_bias(length, lengthrange, acc0, biasprm)

  if (abs(gainprm[0] - 0.5) < MJ_MINVAL) or (abs(gainprm[0] - biasprm[0]) < MJ_MINVAL and abs(gainprm[1] - biasprm[1]) < MJ_MINVAL):
  # if True:
    force = gain * ctrl_act + bias
    # wp.printf("peak force: %f\n", gainprm[2])
  else:
    force = elastic_tendon_mtu(ctrl_act, length, velocity, lengthrange, acc0, gainprm, biasprm)

  if actuator_forcelimited[uid]:
    forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], uid]
    force = wp.clamp(force, forcerange[0], forcerange[1])

  actuator_force_out[worldid, uid] = force


@wp.kernel
def _tendon_actuator_force(
  # Model:
  actuator_trntype: wp.array(dtype=int),
  actuator_trnid: wp.array(dtype=wp.vec2i),
  # Data in:
  actuator_force_in: wp.array2d(dtype=float),
  # Out:
  ten_actfrc_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()

  if actuator_trntype[actid] == TrnType.TENDON:
    tenid = actuator_trnid[actid][0]
    # TODO(team): only compute for tendons with force limits?
    wp.atomic_add(ten_actfrc_out[worldid], tenid, actuator_force_in[worldid, actid])


@wp.kernel
def _tendon_actuator_force_clamp(
  # Model:
  tendon_actfrclimited: wp.array(dtype=bool),
  tendon_actfrcrange: wp.array2d(dtype=wp.vec2),
  actuator_trntype: wp.array(dtype=int),
  actuator_trnid: wp.array(dtype=wp.vec2i),
  # In:
  ten_actfrc_in: wp.array2d(dtype=float),
  # Data out:
  actuator_force_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()

  if actuator_trntype[actid] == TrnType.TENDON:
    tenid = actuator_trnid[actid][0]
    if tendon_actfrclimited[tenid]:
      ten_actfrc = ten_actfrc_in[worldid, tenid]
      actfrcrange = tendon_actfrcrange[worldid % tendon_actfrcrange.shape[0], tenid]

      if ten_actfrc < actfrcrange[0]:
        actuator_force_out[worldid, actid] *= actfrcrange[0] / ten_actfrc
      elif ten_actfrc > actfrcrange[1]:
        actuator_force_out[worldid, actid] *= actfrcrange[1] / ten_actfrc


@wp.kernel
def _qfrc_actuator(
  # Data in:
  moment_rownnz_in: wp.array2d(dtype=int),
  moment_rowadr_in: wp.array2d(dtype=int),
  moment_colind_in: wp.array2d(dtype=int),
  actuator_moment_in: wp.array2d(dtype=float),
  actuator_force_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_actuator_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()

  rownnz = moment_rownnz_in[worldid, actid]
  rowadr = moment_rowadr_in[worldid, actid]

  for i in range(rownnz):
    sparseid = rowadr + i
    colind = moment_colind_in[worldid, sparseid]
    qfrc = actuator_moment_in[worldid, sparseid] * actuator_force_in[worldid, actid]
    wp.atomic_add(qfrc_actuator_out[worldid], colind, qfrc)


@wp.kernel
def _qfrc_actuator_gravcomp_limits(
  # Model:
  ngravcomp: int,
  jnt_actfrclimited: wp.array(dtype=bool),
  jnt_actgravcomp: wp.array(dtype=int),
  jnt_actfrcrange: wp.array2d(dtype=wp.vec2),
  dof_jntid: wp.array(dtype=int),
  # Data in:
  qfrc_gravcomp_in: wp.array2d(dtype=float),
  qfrc_actuator_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_actuator_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  jntid = dof_jntid[dofid]

  qfrc = qfrc_actuator_in[worldid, dofid]

  # actuator-level gravity compensation, skip if added as passive force
  if ngravcomp and jnt_actgravcomp[jntid]:
    qfrc += qfrc_gravcomp_in[worldid, dofid]

  # limits
  if jnt_actfrclimited[jntid]:
    frcrange = jnt_actfrcrange[worldid % jnt_actfrcrange.shape[0], jntid]
    qfrc = wp.clamp(qfrc, frcrange[0], frcrange[1])

  qfrc_actuator_out[worldid, dofid] = qfrc



@event_scope
def fwd_actuation(m: Model, d: Data):
  """Actuation-dependent computations."""
  if not m.nu or (m.opt.disableflags & DisableBit.ACTUATION):
    d.act_dot.zero_()
    d.qfrc_actuator.zero_()
    return

  wp.launch(
    _actuator_force,
    dim=(d.nworld, m.nu),
    inputs=[
      m.na,
      m.opt.timestep,
      m.actuator_dyntype,
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_actadr,
      m.actuator_actnum,
      m.actuator_ctrllimited,
      m.actuator_forcelimited,
      m.actuator_actlimited,
      m.actuator_dynprm,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_actearly,
      m.actuator_ctrlrange,
      m.actuator_forcerange,
      m.actuator_actrange,
      m.actuator_acc0,
      m.actuator_lengthrange,
      d.act,
      d.ctrl,
      d.actuator_length,
      d.actuator_velocity,
      m.opt.disableflags & DisableBit.CLAMPCTRL,
    ],
    outputs=[d.act_dot, d.actuator_force],
  )

  if m.callback.act_dyn:
    m.callback.act_dyn(m, d)
  if m.callback.act_gain:
    m.callback.act_gain(m, d)
  if m.callback.act_bias:
    m.callback.act_bias(m, d)

  if m.ntendon:
    # total actuator force at tendon
    ten_actfrc = wp.zeros((d.nworld, m.ntendon), dtype=float)
    wp.launch(
      _tendon_actuator_force,
      dim=(d.nworld, m.nu),
      inputs=[m.actuator_trntype, m.actuator_trnid, d.actuator_force],
      outputs=[ten_actfrc],
    )

    wp.launch(
      _tendon_actuator_force_clamp,
      dim=(d.nworld, m.nu),
      inputs=[m.tendon_actfrclimited, m.tendon_actfrcrange, m.actuator_trntype, m.actuator_trnid, ten_actfrc],
      outputs=[d.actuator_force],
    )

  # TODO(team): optimize performance
  d.qfrc_actuator.zero_()
  wp.launch(
    _qfrc_actuator,
    dim=(d.nworld, m.nu),
    inputs=[
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
      d.actuator_moment,
      d.actuator_force,
    ],
    outputs=[d.qfrc_actuator],
  )
  wp.launch(
    _qfrc_actuator_gravcomp_limits,
    dim=(d.nworld, m.nv),
    inputs=[
      m.ngravcomp,
      m.jnt_actfrclimited,
      m.jnt_actgravcomp,
      m.jnt_actfrcrange,
      m.dof_jntid,
      d.qfrc_gravcomp,
      d.qfrc_actuator,
    ],
    outputs=[d.qfrc_actuator],
  )


@wp.kernel
def _qfrc_smooth(
  # Data in:
  qfrc_applied_in: wp.array2d(dtype=float),
  qfrc_bias_in: wp.array2d(dtype=float),
  qfrc_passive_in: wp.array2d(dtype=float),
  qfrc_actuator_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_smooth_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  qfrc_smooth_out[worldid, dofid] = (
    qfrc_passive_in[worldid, dofid]
    - qfrc_bias_in[worldid, dofid]
    + qfrc_actuator_in[worldid, dofid]
    + qfrc_applied_in[worldid, dofid]
  )


@event_scope
def fwd_acceleration(m: Model, d: Data, factorize: bool = False):
  """Add up all non-constraint forces, compute qacc_smooth.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    factorize: Flag to factorize inertia matrix.
  """
  wp.launch(
    _qfrc_smooth,
    dim=(d.nworld, m.nv),
    inputs=[d.qfrc_applied, d.qfrc_bias, d.qfrc_passive, d.qfrc_actuator],
    outputs=[d.qfrc_smooth],
  )
  xfrc_accumulate(m, d, d.qfrc_smooth)

  if factorize:
    smooth.factor_solve_i(m, d, d.qM, d.qLD, d.qLDiagInv, d.qacc_smooth, d.qfrc_smooth)
  else:
    smooth.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)


@event_scope
def forward(m: Model, d: Data):
  """Forward dynamics."""
  energy = m.opt.enableflags & EnableBit.ENERGY

  fwd_position(m, d, factorize=False)
  d.sensordata.zero_()
  sensor.sensor_pos(m, d)
  if energy:
    if m.sensor_e_potential == 0:  # not computed by sensor
      sensor.energy_pos(m, d)
  else:
    d.energy.zero_()

  fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  if energy:
    if m.sensor_e_kinetic == 0:  # not computed by sensor
      sensor.energy_vel(m, d)

  if not (m.opt.disableflags & DisableBit.ACTUATION):
    if m.callback.control:
      m.callback.control(m, d)
  fwd_actuation(m, d)
  fwd_acceleration(m, d, factorize=True)

  solver.solve(m, d)
  sensor.sensor_acc(m, d)


@event_scope
def step(m: Model, d: Data):
  """Advance simulation."""
  # TODO(team): mj_checkPos
  # TODO(team): mj_checkVel
  forward(m, d)
  # TODO(team): mj_checkAcc

  if m.opt.integrator == IntegratorType.EULER:
    euler(m, d)
  elif m.opt.integrator == IntegratorType.RK4:
    rungekutta4(m, d)
  elif m.opt.integrator == IntegratorType.IMPLICITFAST:
    implicit(m, d)
  else:
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")


@event_scope
def step1(m: Model, d: Data):
  """Advance simulation in two phases: before input is set by user."""
  energy = m.opt.enableflags & EnableBit.ENERGY
  # TODO(team): mj_checkPos
  # TODO(team): mj_checkVel
  fwd_position(m, d)
  d.sensordata.zero_()
  sensor.sensor_pos(m, d)

  if energy:
    if m.sensor_e_potential == 0:  # not computed by sensor
      sensor.energy_pos(m, d)
  else:
    d.energy.zero_()

  fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  if energy:
    if m.sensor_e_kinetic == 0:  # not computed by sensor
      sensor.energy_vel(m, d)

  if not (m.opt.disableflags & DisableBit.ACTUATION):
    if m.callback.control:
      m.callback.control(m, d)


@event_scope
def step2(m: Model, d: Data):
  """Advance simulation in two phases: after input is set by user."""
  fwd_actuation(m, d)
  fwd_acceleration(m, d)
  solver.solve(m, d)
  sensor.sensor_acc(m, d)
  # TODO(team): mj_checkAcc

  # integrate with Euler or implicitfast
  # TODO(team): implicit
  if m.opt.integrator == IntegratorType.IMPLICITFAST:
    implicit(m, d)
  else:
    # note: RK4 defaults to Euler
    euler(m, d)


@wp.func
def elastic_tendon_mtu_analyse(
  ctrl_act: float,
  length: float,
  velocity: float,
  lengthrange: wp.vec2,
  acc0: float,
  gainprm: vec10f,
  biasprm: vec10f,
) -> float:
  """Compute the force of an elastic tendon muscle-tendon unit (MTU)."""
  range_ = wp.vec2(gainprm[0], gainprm[1])
  peak_force = gainprm[2]
  gain_scale = gainprm[3]
  gain_lmin = gainprm[4]
  gain_lmax = gainprm[5]
  gain_vmax = gainprm[6]
  gain_fvmax = gainprm[8]
  bias_lmax =  gainprm[5]
  bias_fpmax = gainprm[7]
  bias_range_ = wp.vec2(gainprm[0], gainprm[1])

  active_lmin, active_ltrans, active_lmax = 0.4441, 0.73, 1.8123
  passive_lmax = 1.6
  ignore_elastic = 10.0
  if biasprm[4] != gainprm[4]:
  # if False:
    active_lmin, active_ltrans, active_lmax = biasprm[0], biasprm[1], biasprm[2]
    passive_lmax = biasprm[3]
    pennation_angle_opt = biasprm[4]
    ignore_elastic = biasprm[7]
  else:
    active_lmin, active_ltrans, active_lmax = 0.4441, 0.73, 1.8123
    passive_lmax = 1.6
    pennation_angle_opt = 0.0
  # pennation_angle_opt = 0.0
  ctrl_act = max(0.01, ctrl_act)
  beta = 0.01

  ignore_elastic = 20.0

  if abs(peak_force - 1575.06) < 0.5 or abs(peak_force - 3115.51) < 0.5 or abs(peak_force - 6194.84) < 0.5:
    ignore_elastic = 10.0
    # peak_force = peak_force * 0.6667

  # optimum length
  Lce0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])
  Lce0bias = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, bias_range_[1] - bias_range_[0])
  # tendon slack length
  Lt0 = lengthrange[0] - Lce0 * range_[0]
  # wp.printf("lengthrange: %f,%f, normrange: %f,%f, lce0: %f, lt0: %f\n", 
  #           lengthrange[0], lengthrange[1], range_[0], range_[1], Lce0, Lt0)
  lce_h = Lce0 * wp.sin(pennation_angle_opt)

  if length < Lt0:
    cos_phi = wp.sqrt(1.0 - wp.pow(wp.sin(pennation_angle_opt), 2.0))
    lce_w = max(0.0, (length - Lt0))
    lceN = wp.sqrt(lce_w * lce_w + lce_h * lce_h) / wp.max(MJ_MINVAL, Lce0)
    return (0.0, velocity * cos_phi, 1.0, lceN, 0.0, lceN * Lce0, Lt0)

  max_lce = wp.sqrt(length*length + lce_h*lce_h) - 1e-3
  min_lce = lce_h / wp.sin(wp.acos(0.1))

  # normalized length and velocity
  lt_ratio = float(0.01)
  lt = Lt0 * (1.0 + lt_ratio)
  LtN = lt / wp.max(MJ_MINVAL, Lt0)

  lce_w = length - lt
  lce_w = wp.clamp(lce_w, MJ_MINVAL, length)
  lce = wp.sqrt(lce_w * lce_w + lce_h * lce_h)
  lce = max(min_lce, lce)
  lce = min(max_lce, lce)
  lce_w = wp.sqrt(lce * lce - lce_h * lce_h)
  # lce = wp.clamp(lce, MJ_MINVAL, length)
  phi = wp.asin(lce_h / wp.max(MJ_MINVAL, lce))
  cos_phi = wp.cos(phi)

  LceN = lce / wp.max(MJ_MINVAL, Lce0)

  if ignore_elastic < 15.0:
    maxiter = 200
  else:
    maxiter = 0
    lt = Lt0
    LtN = lt / wp.max(MJ_MINVAL, Lt0)
    lce_w = length - lt
    lce_w = wp.clamp(lce_w, MJ_MINVAL, length)
    lce = wp.sqrt(lce_w * lce_w + lce_h * lce_h)
    lce = min(max_lce, lce)
    lce = max(min_lce, lce)
    lce_w = wp.sqrt(lce * lce - lce_h * lce_h)
    lt = length - lce_w
    LceN = lce / wp.max(MJ_MINVAL, Lce0)
    LtN = lt / wp.max(MJ_MINVAL, Lt0)
    phi = wp.asin(lce_h / wp.max(MJ_MINVAL, lce))
    cos_phi = wp.cos(phi)
    vce = velocity * cos_phi

  tol = 1e-6 * peak_force
  cur_iter = int(0)
  error = float(wp.inf)
  error_prev = float(0.0)

  V = float(0.0)

  while cur_iter < maxiter and wp.abs(error) > tol and not (abs(error)<5e-2 and abs(error) - abs(error_prev)<MJ_MINVAL):
    LceN = lce / wp.max(MJ_MINVAL, Lce0)
    lt = length - lce_w
    LtN = lt / wp.max(MJ_MINVAL, Lt0)

    # length curve
    FL = fiber_length_curve(LceN, active_lmin, active_lmax, active_ltrans)
    FLT = tendon_force_length_curve(LtN)
    # FLT = fiber_bias(LtN, bias_lmax, bias_fpmax)
    bias = fiber_bias(LceN, passive_lmax, bias_fpmax)

    # velocity curve
    FV = fiber_velocity_curve(V, gain_fvmax)
    gain = -FL*FV

    fce = (gain * ctrl_act + bias) * peak_force + beta * V
    ft = -FLT * peak_force

    fce_at = fce * cos_phi

    error = fce_at - ft

    dFL_dlceN = fiber_length_deriv(LceN, active_lmin, active_lmax, active_ltrans)
    if dFL_dlceN == 0.0:
      if LceN < 1.0:
        dFL_dlceN = 1e-2
      else:
        dFL_dlceN = -1e-2
    dbias_dlceN = fiber_bias_deriv(LceN, passive_lmax, bias_fpmax)
    dfce_dlce = (-dFL_dlceN * FV * ctrl_act * peak_force + dbias_dlceN * peak_force) / Lce0
    h_over_lce = lce_h / wp.max(MJ_MINVAL, lce)
    dphi_dlce = (-h_over_lce/lce) / wp.sqrt(1.0 - h_over_lce*h_over_lce)
    dcosPHI_dlce = -wp.sin(phi) * dphi_dlce
    dfceAT_dlce = dfce_dlce * cos_phi + fce * dcosPHI_dlce
    # if abs(dfce_dlce) == 0.0:
    #   wp.printf("dfce_dlce == 0.0, fce: %f, dFL_dlceN: %f, V: %f, FV: %f, dbias_dlceN: %f, Lce0: %f, LceN: %f\n", 
    #             fce, dFL_dlceN, V, FV, dbias_dlceN, Lce0, LceN)
    dft_dLt = -tendon_force_length_curve_deriv(LtN) * peak_force / Lt0
    dft_dlce = -dft_dLt
    dferr_dlce = dfceAT_dlce - dft_dlce

    # update velocity
    if abs(dfce_dlce + dft_dLt) > MJ_MINVAL and LtN > 1.0:
      vt = velocity * dfceAT_dlce / (dfceAT_dlce + dft_dLt)
    else:
      vt = velocity

    vce = (velocity - vt) * cos_phi

    h = float(1.0)

    error_prev = error
    lce_prev = lce

    # if cur_iter==0 and abs(peak_force - 521.202) < 1e-2:
    #   wp.printf("iter: %d, error: %.15f, fce: %f, h: %.10f, dfce_dlce: %f, dft_dlce: %f, dft_dltN: %f, ltN: %f, length: %f, velocity: %f, lceN: %f, lce: %f, peak_force: %f\n\n", 
    #             cur_iter, error, fce, h, dfce_dlce, dft_dlce, dft_dLt * Lt0, LtN, length, velocity, LceN, lce, peak_force)

    while(True):
      delta_lce = - h * error_prev / dferr_dlce

      # if abs(delta_lce) < MJ_MINVAL:
      # if h<1e-9 or abs(delta_lce) < MJ_MINVAL:
      #   delta_lce = -wp.sign(delta_lce) * 1e-11 * peak_force
      #   h = 0.0

      # if h<1e-9 and abs(error) > abs(error_prev):
      #   lt_ratio *= 0.5
      #   lt = Lt0 * (1.0 + lt_ratio)
      #   delta_lce = (length - lt) - lce_prev

      # if dfce_dlce==0 and fce==0:
      #   delta_lce = (length-Lt0)-lce_prev

      lce = lce_prev + delta_lce
      # lce = max(MJ_MINVAL, lce)
      lce = max(min_lce, lce)
      lce = min(max_lce, lce)
      # lce = wp.clamp(lce, MJ_MINVAL, length)
      # lce = wp.max(lengthrange[0], wp.min(lengthrange[1], lce))

      LceN = lce / wp.max(MJ_MINVAL, Lce0)
      phi = wp.asin(lce_h / wp.max(MJ_MINVAL, lce))
      cos_phi = wp.cos(phi)
      lce_w = wp.sqrt(lce * lce - lce_h * lce_h)
      lt = length - lce_w
      if lt<0:
        wp.printf("lt < 0")
      LtN = lt / wp.max(MJ_MINVAL, Lt0)

      # length curve
      FL = fiber_length_curve(LceN, active_lmin, active_lmax, active_ltrans)
      FLT = tendon_force_length_curve(LtN)
      # FLT = fiber_bias(LtN, bias_lmax, bias_fpmax)
      bias = fiber_bias(LceN, passive_lmax, bias_fpmax)

      V = vce / wp.max(MJ_MINVAL, Lce0) / gain_vmax
      # velocity curve
      FV = fiber_velocity_curve(V, gain_fvmax)
      gain = -FL*FV

      fce = (gain * ctrl_act + bias) * peak_force + beta * V
      fce_at = fce * cos_phi
      ft = -FLT * peak_force

      error = fce_at - ft
      # print("delta lce:", delta_lce, "delta lce:", lce-lce_prev, " -- error:", error)

      # if abs(delta_lce) <= MJ_MINVAL:
      #   wp.printf("delta error: %.15f\n", abs(error) - abs(error_prev))

      if h<1e-9 and abs(error) >= abs(error_prev):
        delta_lce = -wp.sign(delta_lce) * 1e-11 * peak_force
        lce = lce_prev + delta_lce
        # lce = max(MJ_MINVAL, lce)
        # lce = min(length, lce)
        lce = max(min_lce, lce)
        lce = min(max_lce, lce)
        LceN = lce / wp.max(MJ_MINVAL, Lce0)
        lt = length - lce
        LtN = lt / wp.max(MJ_MINVAL, Lt0)

      if h<1e-9 or abs(error) < abs(error_prev):
        break

      h *= 0.5

    cur_iter += 1

    # print(f'iteration {cur_iter}, error {error}, lceN {LceN}, ltN {LtN}, derr_dlce {dferr_dlce}, h {h}, delta_lce {delta_lce}, delta_err {error - error_prev}')

    # if cur_iter > 56:
    # # if cur_iter>0 and abs(peak_force - 521.202) < 1e-2:
    #   wp.printf("iter: %d, error: %.15f, fce: %f, h: %.10f, dfce_dlce: %f, dft_dlce: %f, dft_dltN: %f, delta_lce_T: %f, delta_lce: %.15f, delta_err: %.15f, ltN: %f, length: %f, velocity: %f, lceN: %f, lce: %f, peak_force: %f\n\n", 
    #             cur_iter, error, fce, h, dfce_dlce, dft_dlce, dft_dLt * Lt0, -h*error/dferr_dlce, delta_lce, error-error_prev, LtN, length, velocity, LceN, lce, peak_force)
    #   # wp.printf("prm0: %f, prm1: %f, prm2: %f, prm3: %f, prm4: %f, prm5: %f, prm6: %f, prm7: %f, prm8: %f, prm9: %f\n\n",
    #   #           gainprm[0], gainprm[1], gainprm[2], gainprm[3], gainprm[4], gainprm[5], gainprm[6], gainprm[7], gainprm[8], gainprm[9])

    # if cur_iter >= 60:
    #   wp.printf("iter: %d, error: %.15f, h: %.10f, dfce_dlce: %f, dft_dlce: %f, delta_lce: %.15f, delta_err: %.15f, ltN: %f, length: %f, lceN: %f, peak_force: %f\n\n", 
    #             cur_iter, error, h, dfce_dlce, dft_dlce, delta_lce, error-error_prev, LtN, length, LceN, peak_force)

  # if cur_iter > 50:
  #     wp.printf("error: %f, peak_force: %f; ", error, peak_force)
  #     # wp.printf("iter: %d, error: %.15f, fce: %f, h: %.10f, dfce_dlce: %f, dfceAT_dlce: %f, dft_dlce: %f, dft_dltN: %f, delta_lce: %.15f, delta_err: %.15f, ltN: %f, tsl: %f, length: %f, lceN: %f, peak_force: %f\n\n", 
  #     #           cur_iter, error, fce, h, dfce_dlce, dfceAT_dlce, dft_dlce, dft_dLt * Lt0, delta_lce, error-error_prev, LtN, Lt0, length, LceN, peak_force)
      
  # length curve
  FL = fiber_length_curve(LceN, active_lmin, active_lmax, active_ltrans)
  FLT = tendon_force_length_curve(LtN)
  bias = fiber_bias(LceN, passive_lmax, bias_fpmax)

  V = vce / wp.max(MJ_MINVAL, Lce0) / gain_vmax
  # velocity curve
  FV = fiber_velocity_curve(V, gain_fvmax)
  gain = -FL*FV

  # if wp.isnan(((gain * ctrl_act + bias) * peak_force + beta * V) * cos_phi):
  #   wp.printf("NaN detected in muscle force computation!\n")
  #   wp.printf("iters: %d, error: %.15f, err_prev: %f, fce: %f, h: %.10f, dfce_dlce: %f, dft_dlce: %f, dft_dltN: %f, ltN: %f, length: %f, velocity: %f, lceN: %f, lce: %f, peak_force: %f, phi: %f, cosphi: %f\n\n", 
  #             cur_iter, error, error_prev, fce, h, dfce_dlce, dft_dlce, dft_dLt * Lt0, LtN, length, velocity, LceN, lce, peak_force, phi, cos_phi)
  #   wp.printf("gain4: %f, bias4: %f\n\n", gainprm[4], biasprm[4])

  if LtN < 0.8:
    print("Muscle analyse result -- iters:", cur_iter, "error:", error, "length:", length, "velocity:", velocity, "lceN:", LceN, "ltN:", LtN)

  fce = (gain * ctrl_act + bias) * peak_force + beta * V
  ft = - FLT * peak_force
  ft = fce * cos_phi

  return fce, vce, LtN, LceN, ft, lce, lt

def mujoco_mtu_analyse(
  ctrl_act: float,
  length: float,
  velocity: float,
  lengthrange: wp.vec2,
  acc0: float,
  gainprm: vec10f,
  biasprm: vec10f,
):
  """Analyze the elastic tendon muscle-tendon unit (MTU) model."""
  gain = util_misc.muscle_gain(length, velocity, lengthrange, acc0, gainprm)
  bias = util_misc.muscle_bias(length, lengthrange, acc0, biasprm)
  force = gain * ctrl_act + bias

  return force, velocity, 1.0, 1.0