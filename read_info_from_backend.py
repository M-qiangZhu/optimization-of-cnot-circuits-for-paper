# -*- coding: utf-8 -*-

"""
    @Author kungfu
    @Date 2023/10/30 23:54
    @Describe 
    @Version 1.0
"""

from qiskit import IBMQ

# 配置 IBM Quantum Experience 帐户
IBMQ.save_account("86d1f23f0a57fefbd11e508ebf722ef6237238f854de5326079f82119b78904647bc56227ee39f6bf4433ba4f251d34e2a0afc12e60436f22ab3d1b8f9ebbafe")

# 加载 IBMQ 后端
IBMQ.load_account()

# 获取 "quito" 设备
provider = IBMQ.get_provider(hub='ibm-q')
quito = provider.get_backend("ibmq_quito")

# 获取设备的参数信息
properties = quito.properties()

# 打印所有参数信息
print(properties)
