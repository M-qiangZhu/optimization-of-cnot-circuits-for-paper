class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


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
    n = 4
    tree = generate_full_binary_tree(n)
    print_tree(tree)


