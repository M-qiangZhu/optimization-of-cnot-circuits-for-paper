# -*- coding: utf-8 -*-

"""
    @Author kungfu
    @Date 2023/6/4 14:32
    @Describe 
    @Version 1.0
"""


from ezQpy import Account

cir = Account.convert_qasm_to_qcis_from_file("./4gt5_75_synthetic.qasm")
print(cir)

