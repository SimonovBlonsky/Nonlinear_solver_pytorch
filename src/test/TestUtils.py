import Utils
import torch

# 测试代码
def testJacobianR():
    omega = torch.randn(3)  # 随机生成一个三维旋转向量
    jacobian = Utils.JacobianR(omega)
    print("Omega: ",omega)
    print("右雅可比矩阵：")
    print(jacobian)
    
def testHat():
    omega = torch.tensor([1.0, 2.0, 3.0])  # 示例旋转向量
    omega_hat = Utils.hat(omega)
    print("旋转向量 omega 的 hat 矩阵：")
    print(omega_hat)

if __name__ == "__main__":
    testHat()