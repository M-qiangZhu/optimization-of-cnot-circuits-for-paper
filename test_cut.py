# -*- coding: utf-8 -*-

"""
    @Author kungfu
    @Date 2024/3/20 22:48
    @Describe 
    @Version 1.0
"""


class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        # 初始化树节点
        self.value = value  # 节点的值
        self.left = left  # 左子节点
        self.right = right  # 右子节点


def generate_full_binary_tree(n):
    # 递归生成高度为n+1的满二叉树
    if n == 0:
        return TreeNode(0)

    root = TreeNode(0)
    root.left = generate_full_binary_tree(n - 1)
    root.left.value = 0
    root.right = generate_full_binary_tree(n - 1)
    root.right.value = 1

    return root


def print_tree(node, level=0):
    if node is not None:
        print_tree(node.right, level + 1)
        print(' ' * 4 * level + '->', node.value)
        print_tree(node.left, level + 1)


def depth_first_search_pruning(node, level=0):
    if node is None:
        return

    # 打印当前节点及其所在层级
    print(f"Node {node.value} at level {level}")

    # 如果节点的值为1（非根节点），则剪枝
    if node.value != 0 and level > 0:  # 根节点层级为0
        return

    # 递归遍历左子节点，层级加1
    depth_first_search_pruning(node.left, level + 1)
    # 递归遍历右子节点，层级加1
    depth_first_search_pruning(node.right, level + 1)

def set_child_values(node, target_level, node_index, left_value, right_value, current_level=0):
    # 如果当前节点为空或已超过目标层级，则直接返回
    if node is None or current_level > target_level:
        return

    # 如果达到目标层级的前一层
    if current_level == target_level - 1:
        # 遍历到指定的节点
        if node_index == 0:
            # 设置左孩子的值
            if node.left is not None:
                node.left.value = left_value
            else:
                node.left = TreeNode(left_value)

            # 设置右孩子的值
            if node.right is not None:
                node.right.value = right_value
            else:
                node.right = TreeNode(right_value)
            return

    # 递归遍历左右子树，同时将节点索引除以2（二叉树的性质）
    set_child_values(node.left, target_level, node_index // 2, left_value, right_value, current_level + 1)
    set_child_values(node.right, target_level, node_index // 2, left_value, right_value, current_level + 1)



if __name__ == '__main__':
    # 生成一个高度为3的满二叉树（n=2）
    tree = generate_full_binary_tree(3)
    print_tree(tree)
    depth_first_search_pruning(tree)  # 执行深度优先搜索并打印节点所在层级

    # 使用示例
    tree = generate_full_binary_tree(3)  # 生成高度为4的满二叉树
    set_child_values(tree, 2, 1, 9, 8)  # 假设我们要设置第2层，第1个节点的左右孩子值分别为9和8
    print_tree(tree)
    # set_child_values(tree,0,1,1,2)
    # set_child_values(tree,1,2,4,7)
    # set_child_values(tree,2,3,11,8)
    # print_tree(tree)

