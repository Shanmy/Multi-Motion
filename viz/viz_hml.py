import imageio
import numpy as np
import torch
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from pdb import set_trace as st
 
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]
]
 
# ------------------------------ 2D Image Drawing --------------------------- #
def get_limit_from_joints(joints):
    '''
    get the limit of the joints
 
    inputs:
    -------
    joints: (n, 3)
 
    return:
    -------
    x_limit: (2,)
 
    y_limit: (2,)
 
    z_limit: (2,)
 
    '''
    assert len(joints.shape) == 2
 
    x_limit = joints[:, 0].min(), joints[:, 0].max()
    y_limit = joints[:, 1].min(), joints[:, 1].max()
    z_limit = joints[:, 2].min(), joints[:, 2].max()
 
    return x_limit, y_limit, z_limit
 
 
def set_axes_equal(x_limits, y_limits, z_limits, ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
 
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
 
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
 
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])
 
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
 
 
def draw_skeleton(
    joints, kinematic_chain, limits=None, y_is_up=True, caption=""
):
    ''' Draw the first frame of the motion
    # joints: (n_frames, n_joints, 3)
    # kinematic_chain: [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], ...]
    '''
 
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(caption)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
 
    if limits is None:
        x_limit, y_limit, z_limit = get_limit_from_joints(joints)
    else:
        x_limit, y_limit, z_limit = limits
 
    set_axes_equal(x_limit, y_limit, z_limit, ax)
 
    for chain in kinematic_chain:
        for i in range(len(chain) - 1):
            j1, j2 = chain[i], chain[i + 1]
            ax.plot(
                [joints[j1, 0], joints[j2, 0]],
                [joints[j1, 1], joints[j2, 1]],
                [joints[j1, 2], joints[j2, 2]],
                color='b', marker='o', markersize=2, linestyle='-'
            )
 
    for i, joint in enumerate(joints):
        ax.text(joint[0], joint[1], joint[2], f'{i}')
    
 
    if y_is_up:
        ax.view_init(elev=100, azim=-90)
 
    # plt.savefig('./preprocess/skeleton.png')
    fig.canvas.draw()
    data = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
 
    # Reshape the data into an image
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
 
    # clean the figure
    plt.close()
 
    return image
 
 
def draw_skeleton_with_parent(joints, parent):
    # trimmed_joints: [35, 3]
    trimmed_joints = joints[:len(parent)]
    trimmed_joints = trimmed_joints.data.cpu().numpy()
 
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
 
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Skeleton')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
 
    x_limit, y_limit, z_limit = get_limit_from_joints(trimmed_joints)
 
    set_axes_equal(x_limit, y_limit, z_limit, ax)
 
    for child, parent in enumerate(parent):
        if parent == -1:
            continue
        ax.plot(
            [trimmed_joints[child, 0], trimmed_joints[parent, 0]],
            [trimmed_joints[child, 1], trimmed_joints[parent, 1]],
            [trimmed_joints[child, 2], trimmed_joints[parent, 2]],
            color='b', marker='o', markersize=2, linestyle='-'
        )
 
    # for chain in kinematic_chain:
    #     for i in range(len(chain) - 1):
    #         j1, j2 = chain[i], chain[i + 1]
    #         ax.plot([joints[j1, 0], joints[j2, 0]],
    #                 [joints[j1, 1], joints[j2, 1]],
    #                 [joints[j1, 2], joints[j2, 2]],
    #                 color='b', marker='o', markersize=2, linestyle='-')
 
    for i, joint in enumerate(trimmed_joints):
        ax.text(joint[0], joint[1], joint[2], f'{i}')
 
    # ax.view_init(elev=100, azim=-90)
    # plt.savefig('./skeleton.png')
    fig.canvas.draw()
    data = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
 
    # Reshape the data into an image
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
 
    return image
 
 
def draw_skeleton_animation(
    frame_joints, save_gif_path: str, verbose: bool = False, caption: str = ""
):
    '''
    draw the skeleton
 
    inputs:
    -------
    frame_joints: (n_frames, n_joints, 3)
        joints positions accross frames
    save_gif_path
        the file path to save the gif
    '''
    images = []
 
    limits = get_limit_from_joints(frame_joints.reshape(-1, 3))
 
    if verbose:
        iterable = tqdm(
            frame_joints, desc=f"generating '{save_gif_path}'",
        )
    else:
        iterable = frame_joints
 
    for joints in iterable:
        joint_image = draw_skeleton(
            joints, t2m_kinematic_chain, limits=limits, caption=caption
        )
        images.append(joint_image)
 
    imageio.mimsave(save_gif_path, images)
    
def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)
def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask
    
def recover_root_rot_pos(data: torch.Tensor):
    '''
    Recover global angle and positions for rotation dataset.
    all the operations are done in pytorch
    inputs:
    -------
    data (..., 263)
    return:
    -------
    root_rot_velocity (B, seq_len, 1)
    root_linear_velocity (B, seq_len, 2)
    root_y (B, seq_len, 1)
    ric_data (B, seq_len, (joint_num - 1)*3)
    rot_data (B, seq_len, (joint_num - 1)*6)
    local_velocity (B, seq_len, joint_num*3)
    foot contact (B, seq_len, 4)
    '''
    # TODO clear the docstring and add dimension description
    # rot_vel : [...]
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    # Get Y-axis rotation from rotation velocity
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    # Add Y-axis rotation to root position
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos
def recover_from_ric(data, joints_num: int):
    '''
    recover global positions from rotation invariant coordinates
    inputs:
    -------
    data (..., 263)
        the data from mdm
    joints_num
        the number of joints, include global one
    '''
    # TODO addd dimension description
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    # add Y-axis rotation to local joints
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions
    )
    # add root XZ to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    # concate root and joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


if __name__ == "__main__":
    
    # draw one sample
    sample = torch.from_numpy(np.load("/home/s9053168/code/multi-pose-diffusion/diffusion/evaluations/exp.npy"))
    sample = recover_from_ric(sample, 22)
    draw_skeleton_animation(sample, f"mdm-2/exp.gif", caption="")
    breakpoint()
    
    # draw from traine model
    sample = torch.from_numpy(np.load("/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mdm-test/default_config_mdm_4_gpu/23-10-08-05-37-18/samples/samples_64x263x1x1.npy"))
    #sample = torch.from_numpy(np.load("/home/s9053168/code/motion-diffusion-model/save/samples/samples_64x263x1x80.npy"))
    mean = np.load("/home/s9053168/code/HumanML3D/HumanML3D/Mean.npy").reshape(263, 1)
    std = np.load("/home/s9053168/code/HumanML3D/HumanML3D/Std.npy").reshape(263, 1)
    
    conds = []
    with open("/home/s9053168/code/motion-diffusion-model/dataset/HumanML3D/test.txt", 'r') as file:
        for _ in range(sample.shape[0]):
            line = file.readline().strip()
            with open(f"/home/s9053168/code/motion-diffusion-model/dataset/HumanML3D/texts/{line}.txt", 'r') as t:
                conds.append(t.readline().strip().split('#')[0])
        
    for i in range(sample.shape[0]):
        ssample = sample[i, :, 0, :]
        ssample = ssample * std + mean
        
        ssample = recover_from_ric(ssample.permute([1, 0]), 22)
        draw_skeleton_animation(ssample, f"mdm-2/{i:02d}.gif", caption=conds[i])