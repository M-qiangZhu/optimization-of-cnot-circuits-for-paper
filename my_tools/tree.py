class Node:

    def __init__(self, val, is_steiner_point=False, left_child=None, right_child=None, parent=None):
        self.val = val
        self.is_steiner_point = is_steiner_point
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent

    def insert_left_child(self, val, is_steiner_point, **kwargs):
        if self.left_child is None:
            tmp = Node(val, is_steiner_point, **kwargs)
            tmp.parent = self
            self.left_child = tmp
        else:
            tmp = Node(val, is_steiner_point, **kwargs)
            self.left_child.parent = tmp
            tmp.left_child = self.left_child
            tmp.parent = self
            self.left_child = tmp

    def insert_right_child(self, val, is_steiner_point, **kwargs):
        if self.right_child is None:
            tmp = Node(val, is_steiner_point, **kwargs)
            tmp.parent = self
            self.right_child = tmp
        else:
            tmp = Node(val, is_steiner_point, **kwargs)
            self.right_child.parent = tmp
            tmp.right_child = self.right_child
            tmp.parent = self
            self.right_child = tmp

    # def get_root(self):
    #     return self.val

    # def set_root_val(self, val):
    #     self.val = val

    def get_value(self):
        if self is not None:
            return self.val
        return None

    def get_left_child(self):
        return self.left_child

    def get_right_child(self):
        return self.right_child


class Tree:

    def __init__(self, v: dict, e: list[tuple], r: int):
        self.vertex = v
        self.edges = e
        self.root = r

    def gen_tree(self):
        # 设置 值: 是否是Steiner点 查询字典
        global root_val, root_is_steiner_point
        v_dic = self.vertex
        edges = self.edges
        self.vertex = sorted(self.vertex.items(), key=lambda x: x[0])
        print(f"self.vertex : {self.vertex}")
        # 记录根节点
        for t in self.vertex:
            if t[0] == self.root:
                root_val, root_is_steiner_point = t[0], t[1]
                break
        # root_val, root_is_steiner_point= self.vertex[0]
        root = Node(root_val, root_is_steiner_point)
        # tmp_node = root
        # self.vertex.pop(0)
        # while len(self.vertex) != 0:
        #     # 从当前顶点列表里取出一个顶点, 如果在树中能找到这个顶点,则处理它, 如果找不到, 就找下一个点
        #     n = len(self.vertex)
        #     for i in range(n):
        #         v = self.vertex[i]
        #         vertex_val = v[0]
        #     # for v in self.vertex:
        #     #     vertex_val = v[0]
        #         # 将 travel_p 定位到 当前值对应的顶点处
        #         travel_p = find_node(root, vertex_val)
        #         if travel_p is not None:
        #             # print(f"当前tmp_node为 : {travel_p.val}")
        #             # print(f"vertex_val : {vertex_val}")
        #             # 在边集合中找包含当前节点的边
        #             target_edges = [edge for edge in self.edges if vertex_val in edge]
        #             # print(f"target_edges : {target_edges}")
        #             # 遍历待处理的边集合
        #             for edge in target_edges:
        #                 # 获取待加入树的顶点值
        #                 if edge[0] == vertex_val:
        #                     next_node_val = edge[1]
        #                 else:
        #                     next_node_val = edge[0]
        #                 # print(f"next_node_val = {next_node_val}")
        #                 # 根据值, 查询是否是Steiner点
        #                 is_steiner_point = v_dic.get(next_node_val)
        #                 # # 根据键值对生成节点
        #                 # next_node = Node(next_node_val, is_steiner_point)
        #                 # 判断当前节点 travel_p 是否有左右孩子, 哪个没有往哪儿插
        #                 if travel_p.get_left_child() is None:
        #                     travel_p.insert_left_child(next_node_val, is_steiner_point)
        #                 else:
        #                     travel_p.insert_right_child(next_node_val, is_steiner_point)
        #             # 从顶点列表删除当前顶点和Steiner点标志
        #             self.vertex.remove(v)
        #             # 从边集合删除target_edges中的边
        #             for e in target_edges:
        #                 self.edges.remove(e)
        #             # print(f"self.edges : {self.edges}")
        #             # print(f"vertex : {self.vertex}")
        #             # print("-" * 10)
        #         else:
        #             continue
        #     n = n - 1

        # 遍历边集合
        generator(v_dic, root, edges)
        print(root)

        return root


def generator(v_dic, node, edges_list):
    if len(edges_list) == 0:
        return
    tar_edges = []
    for e in edges_list:
        if node.val in e:
            tar_edges.append(e)
    # 默认插入到左子树
    if len(tar_edges) != 0:
        edge = tar_edges.pop()
        # 获取待加入树的顶点值
        if edge[0] == node.val:
            next_node_val = edge[1]
        else:
            next_node_val = edge[0]
        # 根据值, 查询是否是Steiner点
        is_steiner_point = v_dic.get(next_node_val)
        # 根据键值对生成节点
        node.insert_left_child(next_node_val, is_steiner_point)
        # 从边集合中删除已经加入到树中的边
        edges_list.remove(edge)

        generator(v_dic, node.left_child, edges_list)
    if len(tar_edges) != 0:
        edge = tar_edges.pop()
        # 获取待加入树的顶点值
        if edge[0] == node.val:
            next_node_val = edge[1]
        else:
            next_node_val = edge[0]
        # 根据值, 查询是否是Steiner点
        is_steiner_point = v_dic.get(next_node_val)
        node.insert_right_child(next_node_val, is_steiner_point)
        # 从边集合中删除已经加入到树中的边
        edges_list.remove(edge)
        generator(v_dic, node.right_child, edges_list)
    return node


# 先序遍历
def find_node(node, val):
    # print(f"node.val : {node.val}")
    if node is None:
        return
    if node.val == val:
        return node
    if node.left_child is not None:
        return find_node(node.left_child, val)
    if node.right_child is not None:
        return find_node(node.right_child, val)
    return None


# 向下消元Steiner点置1
# def


if __name__ == '__main__':
    # vertex = {0: False, 3: True, 1: False, 4: False}
    vertex = {1: False, 2: False, 3: False, 4: True}
    # vertex = {0: True, 1: False, 2: False, 3: False, 4: False}
    edges = [(1, 2), (1, 3), (3, 4)]
    tree = Tree(vertex, edges, 4)
    root = tree.gen_tree()
    print(root.get_value())  # 0
