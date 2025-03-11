from dpvo.lietorch import SO3
from dpvo.lietorch import SE3
import math
import torch

def JacobianR(omega: torch.Tensor) -> torch.Tensor:
    """
    计算给定旋转向量 omega 在 SO(3) 群中的右雅可比矩阵。
    
    参数：
    omega -- 三维向量，表示旋转增量

    返回：
    右雅可比矩阵(3x3)
    """
    # 确保 omega 是一个三维向量
    assert omega.shape == (3,), "omega should be a 3-dimensional vector"

    # Use SO3 to represent rotation
    so3_element = SO3.exp(omega)
    rotation_matrix = so3_element.matrix()

    # 计算右雅可比矩阵
    # 右雅可比矩阵 R'(omega) 是关于 omega 的导数，类似于旋转矩阵关于小增量的导数
    # 我们可以通过对 SO3 的指数映射进行求导来获得雅可比矩阵
    epsilon = 1e-6  # 计算数值导数时的微小增量
    omega_plus = omega + epsilon * torch.eye(3)[:, 0]  # 在每个方向上进行增量
    so3_plus = SO3.exp(omega_plus)
    rotation_matrix_plus = so3_plus.matrix()
    
    # 数值求导得到雅可比矩阵
    # 注意这里输出还是4 * 4的，只不过平移部分为0
    jacobian = (rotation_matrix_plus - rotation_matrix) / epsilon
    
    return jacobian

def hat(omega: torch.Tensor) -> torch.Tensor:
    """
    将一个三维旋转向量 omega 转换为一个 3x3 的反对称矩阵。

    参数:
    omega -- 一个三维向量，表示旋转向量 (3x1)

    返回:
    3x3 的反对称矩阵
    """
    assert omega.shape == (3,), "输入的 omega 必须是一个三维向量"
    
    # 创建一个 3x3 的反对称矩阵
    omega_hat = torch.zeros((3, 3), dtype=omega.dtype, device=omega.device)
    
    omega_hat[0, 1] = -omega[2]
    omega_hat[0, 2] = omega[1]
    omega_hat[1, 0] = omega[2]
    omega_hat[1, 2] = -omega[0]
    omega_hat[2, 0] = -omega[1]
    omega_hat[2, 1] = omega[0]
    
    return omega_hat

# 工具函数的定义
def skewSymmetric(v):
    """计算一个向量的反对称矩阵"""
    return torch.tensor([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

def Qleft(q):
    """四元数左乘操作，返回左乘矩阵"""
    return torch.tensor([[q[0], -q[1], -q[2], -q[3]],
                         [q[1], q[0], q[3], -q[2]],
                         [q[2], -q[3], q[0], q[1]],
                         [q[3], q[2], -q[1], q[0]]])

def Qright(q):
    """四元数右乘操作，返回右乘矩阵"""
    return torch.tensor([[q[0], q[1], q[2], q[3]],
                         [-q[1], q[0], q[3], -q[2]],
                         [-q[2], -q[3], q[0], q[1]],
                         [-q[3], q[2], -q[1], q[0]]])


def deltaQ(theta: torch.Tensor):
    """Compute half theta

    Args:
        theta (torch.Tensor): input. Vector3d

    Returns:
        torch.Tensor: half q
    """
    half_theta = theta / 2.0
    
    w = 1.0
    x, y, z =half_theta
    
    dq = torch.tensor([w, x, y, z], dtype=torch.float)
    return dq

def A2q(axis_angle):
    # Quaternion from axis-angle representation
    angle = torch.norm(axis_angle)
    if angle > 1e-6:
        axis = axis_angle / angle
        return torch.cos(angle / 2) + axis * torch.sin(angle / 2)
    else:
        return torch.tensor([1.0, 0.0, 0.0, 0.0])
    
def q2R(q: torch.Tensor):
    w, x, y, z = q

    R = torch.zeros(3, 3)
    
    R[0, 0] = 1 - 2 * (y**2 + z**2)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)

    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x**2 + z**2)
    R[1, 2] = 2 * (y * z - w * x)

    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x**2 + y**2)

    return R

def R2q(R: torch.Tensor):
    w = torch.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0

    x = (R[2, 1] - R[1, 2]) / (4.0 * w)
    y = (R[0, 2] - R[2, 0]) / (4.0 * w)
    z = (R[1, 0] - R[0, 1]) / (4.0 * w)

    return torch.tensor([w, x, y, z])

def normalize_quaternion(q):
    # Normalize quaternion
    norm = torch.norm(q)
    return q / norm if norm > 0 else q

def NormalizeR(R :torch.Tensor):
    # R to q to normalized q to R
    q = R2q(R)
    q = normalize_quaternion(q)
    return q2R(q)