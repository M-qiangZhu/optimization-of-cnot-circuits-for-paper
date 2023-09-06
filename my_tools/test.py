from itertools import combinations
from itertools import permutations

class Cat:
    def __init__(self, name):
        print("这是初始化方法")
        self.name = name

    def eat(self):
        print(f"{self.name} 爱吃鱼")


def my_sum(a, b):
    b = a + b
    return b


if __name__ == '__main__':
    tom = Cat("Tom")
    print(tom.name)
    tom.eat()

    vertex = {0: False, 3: True, 1: False, 4: False}
    edges = [(0, 1), (1, 3), (3, 4)]
    vertex = sorted(vertex.items(), key=lambda x: x[0])

    print(vertex)

    # while len(vertex) != 0:
    #     val, is_steiner_point = vertex[0]
    #     print(f"val = {val} : {is_steiner_point}")
    #     vertex.pop(0)

    target_edges = [(0, 1), (1, 3)]
    # for e in target_edges:
    #     edges.remove(e)
    l = [edges.remove(e) for e in target_edges]
    print(l)

    a = 1
    b = 2
    my_sum(a, b)
    print(a)
    print(b)



    """
    所谓排列，就是指从给定个数的元素中取出指定个数的元素进行排序。
    组合则是指从给定个数的元素中仅仅取出指定个数的元素，不考虑排序。
    """
    # 组合 从10个数里面挑4个
    data = list(combinations([i for i in range(2, 5)], 1))
    print(f"共有{len(data)}中选法")
    print(data)
    print(type(data))
    # 排列 从5个数里面挑3个
    data = list(permutations([i for i in range(2, 5)], 2))
    print(f"共有{len(data)}中选法")
    print(data)

    l1 = [1, 2, 3]
    l2 = l1
    l2.append(4)
    print(l1)
