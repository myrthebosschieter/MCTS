import random
import matplotlib.pyplot as plt
import numpy as np


class Node:
    def __init__(self, value=float("-inf"), left=None, right=None, parent=None, meta=None):
        """
        Value also becomes reward for parent nodes
        """
        self.value = value
        self.left = left
        self.right = right
        self.parent = parent
        self.meta = meta

    """
    Is last node in tree
    """

    def is_terminal(self):
        return self.left is None and self.right is None


class MCSTSCalculator:
    def __init__(self, depth=12, c_constant=0.5, random_values=None):
        self.is_done = False
        self.c_constant = c_constant
        self.depth = depth
        total_nodes = 2 ** self.depth
        if random_values is None:
            self.random_values = list(np.random.uniform(0, 100, total_nodes))
        else:
            self.random_values = random_values
        self.root_btree = Node()

    def generate_btree(self):
        self.node_recursive_insert(self.root_btree, 1)

    def node_recursive_insert(self, node, current_depth):
        if current_depth == self.depth:
            node.value = self.random_values.pop()
            return
        if node.left is None:
            node.left = Node()
            node.left.parent = node
            self.node_recursive_insert(node.left, current_depth + 1)
        if node.right is None:
            node.right = Node()
            node.right.parent = node
            self.node_recursive_insert(node.right, current_depth + 1)

    def traverse(self, node):
        if self.children_are_visited(node):
            best_ucb_child = self.pick_ucb_child(node)
            return self.traverse(best_ucb_child)

        if node.is_terminal() or (node.left.meta is None and node.right.meta is None):
            # return this node when nothing is visited
            # or we reached a terminal node
            return node

        # this only work with max 2 children
        if node.left.meta is None:
            return node.left
        return node.right

    def rollout(self, node):
        while node.is_terminal() == False:
            node = self.pick_random_child(node)

        return node

    def backpropagate(self, node):
        self.update_node_stats(node)

        # we reached the root
        if node is self.root_btree:
            return

        self.backpropagate(node.parent)

    def update_node_stats(self, node):
        if node.meta is None:
            node.meta = NodeStats()

        # no need to sum the childs as it is always + 1
        node.meta.visit()

        # for the terminal node we will use the value as the reward
        if node.is_terminal():
            node.meta.reward = node.value
            return

        # reward of this node is the sum of its children's rewards
        children_reward_sum = 0
        if node.left is not None and node.left.meta is not None:
            children_reward_sum = children_reward_sum + node.left.meta.reward
        if node.right is not None and node.right.meta is not None:
            children_reward_sum = children_reward_sum + node.right.meta.reward

        node.meta.reward = children_reward_sum

    def pick_random_child(self, node):
        if bool(random.getrandbits(1)):
            return node.left

        return node.right

    def children_are_visited(self, node):
        return node.is_terminal() is False and node.left.meta is not None and node.right.meta is not None

    def pick_ucb_child(self, node):
        if node.is_terminal():
            raise Exception("This shouldn't happen, snowcap reached ground")

        ucb_left = self.get_node_ucb(node.left)
        ucb_right = self.get_node_ucb(node.right)

        if ucb_left == ucb_right:
            print('Same ucb score, insane!, picking a random child')
            return self.pick_random_child(node)

        if ucb_left > ucb_right:
            return node.left
        return node.right

    def get_node_ucb(self, node):
        N = node.parent.meta.visits
        n = node.meta.visits
        x = node.meta.reward / n

        return x + (self.c_constant * np.sqrt(np.log(N) / n))

    def iterate(self):
        node_from_traverse = self.traverse(self.root_btree)

        random_terminal_node_from_rollout = self.rollout(node_from_traverse)
        self.backpropagate(random_terminal_node_from_rollout)

    def get_best_child(self):
        return self.find_best_child(self.root_btree)

    def find_best_child(self, node):
        if node.is_terminal():
            return node

        node_left_visits = 0
        node_right_visits = 0
        if node.left.meta is not None:
            node_left_visits = node.left.meta.visits
        if node.right.meta is not None:
            node_right_visits = node.right.meta.visits

        if node_left_visits > node_right_visits:
            return self.find_best_child(node.left)
        return self.find_best_child(node.right)


class NodeStats:
    def __init__(self, reward=0, visits=0):
        self.reward = reward
        self.visits = visits

    def visit(self):
        self.visits = self.visits + 1


class TreePrinter:
    def __init__(self, value_space, space_between, depth, tree_root, calculator):
        self.tree_root = tree_root
        self.value_space = value_space
        self.space_between = space_between
        self.depth = depth
        self.calculator = calculator

    def print_tree(self):
        for i in range(self.depth):
            self.print_layer(i + 1)

    def print_layer(self, layer):
        nodes = self.get_nodes_for_layer(layer, 1, self.tree_root)
        width = len(nodes)
        layer_width = self.total_width() - (self.value_space * self.space_between * width)
        padding = int(layer_width / 2.5)
        print('[layer:', layer, ']', padding * ' ', end='')
        for node in nodes:
            reward = node.value
            visits = 0
            ucb = float("-inf")
            if node.meta is not None:
                reward = node.meta.reward
                visits = node.meta.visits
            if node.meta is not None and node.parent is not None:
                ucb = self.calculator.get_node_ucb(node)
            print("%.2f" % reward, '/', visits, '/', ucb, ' ' * self.space_between, end='', sep='')
        print(padding * ' ')

    def total_width(self):
        return self.space_between * self.value_space * (2 ** (self.depth - 1))

    def get_nodes_for_layer(self, layer, current_layer, node):
        nodes = []
        if current_layer == layer:
            nodes.append(node)
            return nodes

        nodes_left = self.get_nodes_for_layer(layer, current_layer + 1, node.left)
        nodes_right = self.get_nodes_for_layer(layer, current_layer + 1, node.right)

        return nodes_left + nodes_right


"""
Print tree every iteration
"""
# depth = 3
# random_values = list(np.random.uniform(0, 100, 2 ** 4))
#
# calculator = MCSTSCalculator(c_constant=0.5, depth=depth, random_values=random_values)
# calculator.generate_btree()
#
# for i in range(5):
#     print('Iteration', i)
#     calculator.iterate()
#     printer = TreePrinter(5, 2, calculator.depth, calculator.root_btree, calculator)
#     printer.print_tree()

"""
Plotting different c constants
"""
different_c_values = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2,5]
different_reward_values = []
depth = 12
random_values = list(np.random.uniform(0, 100, 2 ** depth))

for c_value in different_c_values:
    rewards = []
    for j in range(0, 100):
        random_list = random_values.copy()
        calculator = MCSTSCalculator(c_constant=c_value, depth=depth, random_values=random_list)
        calculator.generate_btree()

        for i in range(50):
            calculator.iterate()

        rewards.append(calculator.get_best_child().value)

    avg_reward = sum(rewards) / len(rewards)
    print('Avg reward for c value:', c_value, '=', avg_reward)
    different_reward_values.append(avg_reward)

figure, ax = plt.subplots()
ax.plot(different_c_values, different_reward_values)
ax.set(xlabel='c value (c)', ylabel='reward (r)', title='mean reward (100 runs) / c', ylim=(0, 102), xlim=(0, 5))
ax.grid()
ax.plot(different_c_values, [min(random_values)] * len(different_c_values))
ax.plot(different_c_values, [max(random_values)] * len(different_c_values))
ax.legend(("reward", min(random_values), max(random_values)))

plt.show()

"""
Plotting different amount of iterations
"""
iterations = list(range(10, 51))
different_reward_values = []

for iteration in iterations:
    rewards = []
    for j in range(0, 100):
        random_list = random_values.copy()
        calculator = MCSTSCalculator(c_constant=0.5, depth=depth, random_values=random_list)
        calculator.generate_btree()

        for i in range(iteration):
            calculator.iterate()

        rewards.append(calculator.get_best_child().value)

    avg_reward = sum(rewards) / len(rewards)
    print('Avg reward for iteration:', iteration, '=', avg_reward)
    different_reward_values.append(avg_reward)

figure, ax = plt.subplots()
ax.plot(iterations, different_reward_values)
ax.set(xlabel='iteration count (i)', ylabel='reward (r)', title='mean reward (100 runs) / i', ylim=(0, 102))
ax.grid()
ax.plot(iterations, [min(random_values)] * len(iterations))
ax.plot(iterations, [max(random_values)] * len(iterations))
ax.legend(("reward", min(random_values), max(random_values)))

plt.show()