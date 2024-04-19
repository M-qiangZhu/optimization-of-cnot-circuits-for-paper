class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

# class TreeNode:
#     def __init__(self, value=0, left=None, right=None):
#         self.value = value  # 节点的值
#         self.left = left  # 左子节点
#         self.right = right  # 右子节点


def generate_full_binary_tree(n):
    # Base case: when n is 0, return a leaf node
    if n == 0:
        return TreeNode(0)

    # Create the current node
    root = TreeNode(0)

    # Recursively create the left subtree with 0 as value
    root.left = generate_full_binary_tree(n - 1)
    root.left.value = 0

    # Recursively create the right subtree with 1 as value
    root.right = generate_full_binary_tree(n - 1)
    root.right.value = 1

    return root


# Function to print the tree for visualization
def print_tree(node, level=0):
    if node is not None:
        print_tree(node.right, level + 1)
        print(' ' * 4 * level + '->', node.value)
        print_tree(node.left, level + 1)


def depth_first_search(node, depth=0):
    if node is not None:
        # 打印当前节点的值和深度
        print('Depth:', depth, 'Value:', node.value)

        # 首先遍历左子树
        depth_first_search(node.left, depth + 1)

        # 然后遍历右子树
        depth_first_search(node.right, depth + 1)


# def depth_first_search_pruning(node):
#     # Check if the current node is None
#     if node is None:
#         return
#
#     # Visit the node (print its value)
#     print(node.value, end=' ')
#
#     # If the node is not the root and its value is not 0, return (prune)
#     if node.value != 0 and node != tree:  # Assuming 'tree' is the root node
#         return
#
#     # Recursively visit the left and right children
#     depth_first_search_pruning(node.left)
#     depth_first_search_pruning(node.right)

from collections import deque  # 导入deque，用于实现队列功能





# def generate_full_binary_tree(n):
#     if n == 0:
#         return TreeNode(0)  # 基础情况，当树的层数为0时，返回一个值为0的节点
#
#     root = TreeNode(0)  # 创建根节点
#     root.left = generate_full_binary_tree(n - 1)  # 递归创建左子树
#     root.left.value = 0  # 设置左子节点的值为0
#     root.right = generate_full_binary_tree(n - 1)  # 递归创建右子树
#     root.right.value = 1  # 设置右子节点的值为1
#
#     return root  # 返回根节点


def breadth_first_search_pruning(root):
    if root is None:
        return  # 如果根节点为空，直接返回

    queue = deque([root])  # 初始化队列，并将根节点加入队列
    while queue:
        node = queue.popleft()  # 从队列中取出一个节点

        print(node.value, end=' ')  # 访问当前节点

        # 如果节点是根节点或其值为0，则将其子节点加入队列
        if node == root or node.value == 0:
            if node.left:
                queue.append(node.left)  # 将左子节点加入队列
            if node.right:
                queue.append(node.right)  # 将右子节点加入队列





def breadth_first_search_pruning(node):
    if node is None:
        return

    queue = deque([node])
    while queue:
        current_node = queue.popleft()
        print(current_node.value, end=' ')

        if current_node.value != 0 and current_node != node:  # Assuming 'node' is the root node
            continue

        if current_node.left:
            queue.append(current_node.left)
        if current_node.right:
            queue.append(current_node.right)







if __name__ == '__main__':
    # sum = 0
    # for i in range(1, 16):
    #     sum += 400 * pow((1 + 0.15), -i)
    #     print(f"t = {i} : {sum}")
    # print(sum)
    #
    # a1 = 1 / 1.15
    # q = 1 / 1.15
    # n = 15
    #
    # sn = a1 * (1 - pow(q, n)) / (1 - q)
    # print(400 * sn)

    # Generate and print a full binary tree of height 3 (n+1 where n=2)
    n = 3
    tree = generate_full_binary_tree(n)
    print_tree(tree)

    # 使用此方法遍历上面创建的树
    # depth_first_search(tree)

    # Perform depth-first search with pruning on the tree
    # depth_first_search_pruning(tree)

    # breadth_first_search_pruning(tree)

    # 生成高度为3的满二叉树（n=2）
    # tree = generate_full_binary_tree(3)
    breadth_first_search_pruning(tree)  # 对树进行按层遍历并剪枝




