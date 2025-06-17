#!/usr/bin/env python
# coding: utf-8

# In[1]:


from carvekit.api.high import HiInterface
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as Rot
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from tensorflow_probability.substrates import jax as tfp
import torch
import warp as wp
import pickle


import condorgmm
import condorgmm.warp_gmm as warp_gmm

# # Load video

# In[3]:


video_dir = condorgmm.get_root_path() / "assets/condorgmm_bucket/ramen_scene_graph.r3d"
video = condorgmm.data.R3DVideo(video_dir)
frame = video[0]


# # Segmentation

# In[4]:


HIINTERFACE = HiInterface(
    object_type="object",  # Can be "object" or "hairs-like".
    batch_size_seg=5,
    batch_size_matting=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
    matting_mask_size=2048,
    trimap_prob_threshold=220,  # 231,
    trimap_dilation=15,
    trimap_erosion_iters=20,
    fp16=False,
)


# In[5]:


input_image = Image.fromarray(frame.rgb)
obj_mask = np.array(HIINTERFACE([input_image])[0])[..., -1] > 0.5


# In[6]:


obj_num_points = 500
obj_indices = np.random.choice(obj_mask.sum(), size=obj_num_points, replace=False)
obj_cloud = condorgmm.xyz_from_depth_image(frame.depth, *frame.intrinsics)[
    obj_mask, :
].reshape(-1, 3)


# In[7]:


bg_num_points = 500
bg_mask = ~obj_mask
bg_indices = np.random.choice(bg_mask.sum(), size=bg_num_points, replace=False)
bg_cloud = condorgmm.xyz_from_depth_image(frame.depth, *frame.intrinsics)[
    bg_mask, :
].reshape(-1, 3)


# # Finding the table plane

# In[8]:


def ransac(key, data, fitter, scorer, num_samples, num_iters):
    def iteration(key):
        sample = jax.random.choice(key, data, shape=(num_samples,))
        fit = fitter(sample)
        score = scorer(sample, fit)
        return fit, score

    fits, scores = jax.vmap(iteration)(jax.random.split(key, num_iters))
    return fits[jnp.argmax(scores)]


# In[9]:


def plane_fitter(pts_homo):
    U, s, Vt = jnp.linalg.svd(pts_homo)
    return Vt[-1, :]


def plane_scorer(pts_homo, coeffs):
    return -jnp.linalg.norm(pts_homo @ coeffs)


bg_cloud_homo = jnp.hstack([bg_cloud, jnp.ones((len(bg_cloud), 1))])


# In[10]:


key = jax.random.key(0)
table_plane_coeffs = ransac(key, bg_cloud_homo, plane_fitter, plane_scorer, 5, 10)
residuals = jnp.abs(bg_cloud_homo @ table_plane_coeffs)


# In[11]:


def normalize(v):
    return v / jnp.linalg.norm(v)


# In[12]:


def plane_coeffs_to_point_upward_normal(coeffs):
    normal = coeffs[:3]
    point = (-coeffs[3] / (normal @ normal)) * normal
    normal *= -jnp.sign(normal[1])
    return point, normalize(normal)


# In[13]:


def plane_coeffs_to_plane_pose(coeffs):
    point, upward_normal = plane_coeffs_to_point_upward_normal(coeffs)
    x0, y0, z = jnp.array([[1.0, 0, 0], [0, 0, 1], upward_normal])
    y = normalize(y0 - z * (y0 @ z))
    x = normalize(x0 - y * (x0 @ y) - z * (x0 @ z))
    pose_matrix = jnp.eye(4).at[:3, :3].set(jnp.array([x, y, z]).T).at[:3, 3].set(point)
    return condorgmm.pose.Pose.from_matrix(pose_matrix)


# In[14]:


table_point, table_normal = plane_coeffs_to_point_upward_normal(table_plane_coeffs)
table_pose = plane_coeffs_to_plane_pose(table_plane_coeffs)


# # Initial GMM fitting

# In[15]:


obj_spatial_means = np.array(obj_cloud, dtype=np.float32)[obj_indices, :]
# obj_pose = condorgmm.Pose.from_translation(np.median(obj_spatial_means, axis=0))
obj_pose = condorgmm.Pose.from_pos_and_quat(
    np.median(obj_spatial_means, axis=0), table_pose.xyzw
)
obj_spatial_means = obj_pose.inv().apply(obj_spatial_means).astype(np.float32)
obj_rgb_means = np.array(frame.rgb[obj_mask, :].reshape(-1, 3), dtype=np.float32)[
    obj_indices, :
]
obj_log_rgb_scales = np.array(
    np.log(np.ones_like(obj_rgb_means) * 4.0), dtype=np.float32
)
obj_mask_warp = wp.array(obj_mask, dtype=wp.bool)
obj_log_spatial_scales = np.array(
    np.log(np.ones_like(obj_spatial_means) * 0.01), dtype=np.float32
)

obj_gmm = warp_gmm.gmm_warp_from_numpy(
    spatial_means=obj_spatial_means,
    rgb_means=obj_rgb_means,
    log_rgb_scales=obj_log_rgb_scales,
    log_spatial_scales=obj_log_spatial_scales,
    object_posquats=obj_pose.posquat[None, :],
)


# In[16]:


obj_warp_gmm_state = warp_gmm.initialize_state(gmm=obj_gmm, frame=frame)
obj_warp_gmm_state.hyperparams.window_half_width = 7
obj_warp_gmm_state.mask = obj_mask_warp
obj_warp_gmm_state.gmm.spatial_means.requires_grad = True
obj_warp_gmm_state.gmm.quaternions_imaginary.requires_grad = True
obj_warp_gmm_state.gmm.quaternions_real.requires_grad = True
obj_warp_gmm_state.gmm.rgb_means.requires_grad = True
obj_warp_gmm_state.gmm.log_spatial_scales.requires_grad = True
frame_warp = frame.as_warp()

warp_gmm.optimize_params(
    [
        obj_warp_gmm_state.gmm.spatial_means,
        obj_warp_gmm_state.gmm.log_spatial_scales,
        obj_warp_gmm_state.gmm.rgb_means,
        obj_warp_gmm_state.gmm.quaternions_imaginary,
        obj_warp_gmm_state.gmm.quaternions_real,
    ],
    frame_warp,
    obj_warp_gmm_state,
    num_timesteps=1000,
    lr=[5e-4, 1e-3, 1e-2, 1e-2, 1e-2],
    use_tqdm=True,
)


# In[17]:


bg_spatial_means = np.array(bg_cloud, dtype=np.float32)[bg_indices, :]
bg_object_pose = condorgmm.Pose.from_translation(np.median(bg_spatial_means, axis=0))
bg_spatial_means = bg_object_pose.inv().apply(bg_spatial_means).astype(np.float32)
bg_rgb_means = np.array(frame.rgb[bg_mask, :].reshape(-1, 3), dtype=np.float32)[
    bg_indices, :
]
bg_log_rgb_scales = np.array(np.log(np.ones_like(bg_rgb_means) * 4.0), dtype=np.float32)
bg_mask_warp = wp.array(bg_mask, dtype=wp.bool)
bg_log_spatial_scales = np.array(
    np.log(np.ones_like(bg_spatial_means) * 0.01), dtype=np.float32
)

bg_gmm = warp_gmm.gmm_warp_from_numpy(
    spatial_means=bg_spatial_means,
    rgb_means=bg_rgb_means,
    log_rgb_scales=bg_log_rgb_scales,
    log_spatial_scales=bg_log_spatial_scales,
    object_posquats=bg_object_pose.posquat[None, :],
)


# In[18]:


bg_warp_gmm_state = warp_gmm.initialize_state(gmm=bg_gmm, frame=frame)
bg_warp_gmm_state.hyperparams.window_half_width = 7
bg_warp_gmm_state.mask = bg_mask_warp
bg_warp_gmm_state.gmm.spatial_means.requires_grad = True
bg_warp_gmm_state.gmm.quaternions_imaginary.requires_grad = True
bg_warp_gmm_state.gmm.quaternions_real.requires_grad = True
bg_warp_gmm_state.gmm.rgb_means.requires_grad = True
bg_warp_gmm_state.gmm.log_spatial_scales.requires_grad = True
frame_warp = frame.as_warp()

warp_gmm.optimize_params(
    [
        bg_warp_gmm_state.gmm.spatial_means,
        bg_warp_gmm_state.gmm.log_spatial_scales,
        bg_warp_gmm_state.gmm.rgb_means,
        bg_warp_gmm_state.gmm.quaternions_imaginary,
        bg_warp_gmm_state.gmm.quaternions_real,
    ],
    frame_warp,
    bg_warp_gmm_state,
    num_timesteps=3000,
    lr=[5e-4, 1e-3, 1e-2, 1e-2, 1e-2],
    use_tqdm=True,
)


# # Pose factorization

# In[19]:


def rot_axis_angle(vec, rotated_vec):
    vec, rotated_vec = map(normalize, (vec, rotated_vec))
    cross_prod = jnp.cross(vec, rotated_vec)
    angle = jnp.atan2(jnp.linalg.norm(cross_prod), jnp.dot(vec, rotated_vec))
    axis = normalize(cross_prod)
    return axis, angle


# In[20]:


def factor_pose_through_plane(pose, plane_pose):
    normal = Rot.from_quat(plane_pose.xyzw).as_matrix()[:, 2]
    tran, rot = pose.pos, Rot.from_quat(pose.xyzw)
    rotated_normal = rot.apply(normal)
    slack_rot_axis, slack_rot_angle = rot_axis_angle(normal, rotated_normal)
    slack_rot = Rot.from_rotvec(slack_rot_angle * slack_rot_axis)
    planar_rot = slack_rot.inv() * rot
    slack_tran = (
        jnp.dot(tran, rotated_normal) / jnp.dot(normal, rotated_normal)
    ) * normal
    planar_tran = slack_rot.inv().apply(tran - slack_tran)
    slack_pose = condorgmm.Pose.from_pos_and_quat(slack_tran, slack_rot.as_quat())
    planar_pose = condorgmm.Pose.from_pos_and_quat(planar_tran, planar_rot.as_quat())
    return slack_pose, planar_pose


# In[21]:


def planar_pose_to_tran_angle(planar_pose, plane_pose):
    tran, rot = planar_pose.pos, Rot.from_quat(planar_pose.xyzw)
    rot_vec = Rot.as_rotvec(rot)
    _, angle = normalize(rot_vec), jnp.linalg.norm(rot_vec)
    plane_rot_mat = Rot.from_quat(plane_pose.xyzw).as_matrix()
    planar_tran_full = plane_rot_mat.T @ tran
    # assert jnp.linalg.norm(jnp.cross(plane_rot_mat[:, 2], axis)) < 1e-4
    # assert jnp.abs(planar_tran_full[2]) < 1e-6
    # XXX align axis with plane normal and fix angle
    return planar_tran_full[:2], angle


# # Scene Graph Prior

# In[22]:


def gaussian_vmf_logpdf(pose, pose_mean, pos_var, quat_concentration):
    pos, quat = pose.pos.astype(jnp.float32), pose.xyzw.astype(jnp.float32)
    pos_mean, quat_mean = (
        pose_mean.pos.astype(jnp.float32),
        pose_mean.xyzw.astype(jnp.float32),
    )
    pos_distr = tfp.distributions.MultivariateNormalDiag(pos_mean, jnp.full(3, pos_var))
    quat_distr = tfp.distributions.VonMisesFisher(quat_mean, quat_concentration)
    return pos_distr.log_prob(pos) + quat_distr.log_prob(quat)


def uniform_vmf_logpdf(pose, uniform_volume, vmf_mean, vmf_concentration):
    quat = pose.xyzw
    quat_distr = tfp.distributions.VonMisesFisher(vmf_mean, vmf_concentration)
    return -jnp.log(uniform_volume) + quat_distr.log_prob(quat)


def gaussian_vm_logpdf(
    tran, angle, tran_mean, tran_var, angle_mean, angle_concentration
):
    tran_distr = tfp.distributions.MultivariateNormalDiag(
        tran_mean, jnp.full(2, tran_var)
    )
    angle_distr = tfp.distributions.VonMises(angle_mean, angle_concentration)
    return tran_distr.log_prob(tran) + angle_distr.log_prob(angle)


# In[23]:


FLOATING_POSE_MEAN = condorgmm.Pose.identity()
FLOATING_GAUSS_VAR = 5.0
FLOATING_QUAT_CONCENTRATION = 0.0


def floating_prior(obj_pose):
    return gaussian_vmf_logpdf(
        obj_pose, FLOATING_POSE_MEAN, FLOATING_GAUSS_VAR, FLOATING_QUAT_CONCENTRATION
    )


# In[24]:


ON_TABLE_TRAN_MEAN = jnp.zeros(2)
ON_TABLE_TRAN_VAR = 40.0
ON_TABLE_ANGLE_MEAN = 0.0
ON_TABLE_ANGLE_CONCENTRATION = 0.0
ON_TABLE_SLACK_POSE_MEAN = condorgmm.Pose.identity()
ON_TABLE_SLACK_POS_VAR = 0.1
ON_TABLE_SLACK_QUAT_CONCENTRATION = 1.0


def on_table_prior(obj_pose):
    obj_pose_relative_to_table = obj_pose @ table_pose.inv()
    relative_slack_pose, relative_projected_pose = factor_pose_through_plane(
        obj_pose_relative_to_table, table_pose
    )
    tran, angle = planar_pose_to_tran_angle(relative_projected_pose, table_pose)
    projected_pose_logpdf = gaussian_vm_logpdf(
        tran,
        angle,
        ON_TABLE_TRAN_MEAN,
        ON_TABLE_TRAN_VAR,
        ON_TABLE_ANGLE_MEAN,
        ON_TABLE_ANGLE_CONCENTRATION,
    )
    slack_pose_logpdf = gaussian_vmf_logpdf(
        relative_slack_pose,
        ON_TABLE_SLACK_POSE_MEAN,
        ON_TABLE_SLACK_POS_VAR,
        ON_TABLE_SLACK_QUAT_CONCENTRATION,
    )
    return projected_pose_logpdf + slack_pose_logpdf


# # Inference

# In[25]:


scene_gmm = warp_gmm.gmm_warp.concatenate_gmms(
    [obj_warp_gmm_state.gmm, bg_warp_gmm_state.gmm]
)

scene_warp_gmm_state = warp_gmm.initialize_state(gmm=scene_gmm, frame=frame)
scene_warp_gmm_state.hyperparams.window_half_width = 7


# In[26]:


pose_learning_rates = wp.array(
    [0.002, 0.002, 0.002, 0.0004, 0.0004, 0.0004, 0.0004], dtype=wp.float32
)

slacks_over_time = []
planars_over_time = []
obj_poses_over_time = []

for i in range(len(video))[::5]:
    print(".", end="")
    frame = video[i]
    frame_warp = frame.as_warp()
    scene_warp_gmm_state.gmm.object_posquats.requires_grad = True
    scene_warp_gmm_state.gmm.camera_posquat.requires_grad = True
    scene_warp_gmm_state.gmm.rgb_means.requires_grad = True
    _ = warp_gmm.optimize_params(
        [
            scene_warp_gmm_state.gmm.camera_posquat,
            scene_warp_gmm_state.gmm.object_posquats,
            scene_warp_gmm_state.gmm.rgb_means,
        ],
        frame_warp,
        scene_warp_gmm_state,
        200,
        [pose_learning_rates, pose_learning_rates, 1e-1],
        storing_stuff=False,
    )

    gmm = scene_warp_gmm_state.gmm
    obj_pose = condorgmm.Pose(gmm.object_posquats.numpy()[0])
    obj_pose_relative_to_table = obj_pose @ table_pose.inv()
    relative_slack_pose, relative_projected_pose = factor_pose_through_plane(
        obj_pose_relative_to_table, table_pose
    )
    slacks_over_time.append(relative_slack_pose)
    planars_over_time.append(relative_projected_pose)
    obj_poses_over_time.append(obj_pose)


# # Debugging Plots

# In[28]:


floating_lls = jnp.array([floating_prior(pose) for pose in obj_poses_over_time])
on_table_lls = jnp.array([on_table_prior(pose) for pose in obj_poses_over_time])
full_lls = jnp.array([floating_lls, on_table_lls])


# In[29]:


Zs = jax.scipy.special.logsumexp(full_lls, axis=0)
floating_probs, on_table_probs = jnp.exp(full_lls - Zs)


# In[30]:


def draw_graph(ax, edge_strength):
    NODE_SIZE = 1000
    NODE_DIST = 1.4
    pos = {"table": (0, -NODE_DIST), "cup": (0, NODE_DIST)}
    node_colors = {"table": "#f0f0f0", "cup": "#ffcccb"}
    edge_color = mpl.colormaps["Greys"](edge_strength)
    G = nx.DiGraph()
    G.add_node("table")
    G.add_node("cup")
    G.add_edge("table", "cup")

    for node, color in node_colors.items():
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=[node],
            node_shape="o",
            node_size=NODE_SIZE,
            node_color=color,
        )
    if edge_strength > 0.5:
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_color,
            ax=ax,
            arrowstyle="-|>",
            arrowsize=20,
            node_size=NODE_SIZE,
            width=2,
        )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight="bold")


# In[31]:
KEY_FRAMES = [0, 30, 35, 58, 60]
subsampled_frame_idxs = list(range(len(video))[::5])
N = len(KEY_FRAMES)


import pickle
results = {"pngs": [
    Image.open(f"PNGs/sgkf{i}.png") for i in range(len(KEY_FRAMES))
], "video_data": [video[subsampled_frame_idxs[fidx]].rgb for fidx in KEY_FRAMES],
    "floating_probs": floating_probs,
    "on_table_probs": on_table_probs,
    "key_frames": KEY_FRAMES,
    "subsampled_frame_idxs": subsampled_frame_idxs,
}


condorgmm_bucket_dir = condorgmm.get_root_path() / "assets/condorgmm_bucket"
# Save results dictionary to pickle file
results_path = condorgmm_bucket_dir / "scene_graph_results.pkl"
with open(results_path, "wb") as f:
    pickle.dump(results, f)

# Load results dictionary from pickle file
with open(results_path, "rb") as f:
    results = pickle.load(f)
    


fig = plt.figure(figsize=(15, 8))
gs = GridSpec(4, N, figure=fig)
key_frames = results["key_frames"]
subsampled_frame_idxs = results["subsampled_frame_idxs"]
N = len(key_frames)
fig_axes = [[fig.add_subplot(gs[i, j]) for j in range(N)] for i in range(3)]

plot_ax = fig.add_subplot(gs[3, :])
# plot_ax.vlines(KEY_FRAMES, 0, 1)

floating_probs = results["floating_probs"]
on_table_probs = results["on_table_probs"]
plot_ax.plot(floating_probs, label="probability of floating")
plot_ax.plot(on_table_probs, label="probability of being on table")
plot_ax.legend(loc="center right")
plot_ax.set_xlabel("Frame")
plot_ax.set_ylabel("Probability")

video_data = results["video_data"]
for i, fidx in enumerate(key_frames):
    fig_axes[0][i].imshow(video_data[i])
    fig_axes[0][i].set_xticks([])
    fig_axes[0][i].set_yticks([])
    fig_axes[2][i].set_xticks([])
    fig_axes[2][i].set_yticks([])
    fig_axes[1][i].set_xlim(-0.1, 0.1)
    fig_axes[1][i].set_ylim(-2, 2)
    draw_graph(fig_axes[1][i], on_table_probs[fidx])
    fig_axes[2][i].imshow(results["pngs"][i])


plt.tight_layout()
plt.savefig("scene_graph_fig.pdf")



from IPython import embed
embed()

# In[ ]:
