# -*- coding: utf-8 -*-

"""
    @Author kungfu
    @Date 2024/3/27 00:44
    @Describe 
    @Version 1.0
"""

class TreeNode:
    def __init__(self, value, level, path):
        self.value = value
        self.level = level
        self.path = path  # 当前节点的路径，表示从根到当前节点的列表
        self.left = None
        self.right = None

def build_tree(level, path, max_level):
    if level == max_level:
        return None  # 达到最大层，停止构建

    # 创建当前层的节点
    node = TreeNode(path[-1] if path else 0, level, path)

    # 递归构建左右子树
    node.left = build_tree(level + 1, path + [0], max_level) if level < max_level else None
    node.right = build_tree(level + 1, path + [1], max_level) if level < max_level else None

    return node  # 返回当前节点


def print_tree(node, indent=""):
    if node:
        # 打印节点及其路径
        print(f"{indent}{node.value} (Path: {'->'.join(map(str, node.path))})")
        # 递归打印左子树
        print_tree(node.left, indent + "  ")
        # 递归打印右子树
        print_tree(node.right, indent + "  ")

def dfs_search(node, target):
    if node is None:
        return None

    # 如果路径与目标匹配，则找到目标
    if node.path == target:
        return node.path

    # 继续深度优先搜索
    left_result = dfs_search(node.left, target)
    if left_result:
        return left_result

    right_result = dfs_search(node.right, target)
    if right_result:
        return right_result

    return None




if __name__ == '__main__':

    # 构建解空间树
    root = build_tree(0, [], 5)  # 构建树的根节点
    print_tree(root)
    # 搜索目标列表
    target = [0, 1, 0, 1]
    result = dfs_search(root, target)
    print(result)

