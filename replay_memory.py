import random
import numpy as np
from collections import namedtuple
from typing import List, Tuple


class ReplayMemory(object):
  def __init__(self, capacity: int) -> None:
    self.capacity = capacity
    self.sum_tree = Sumtree(capacity=capacity)
    self.transition = namedtuple("Transition",
                                 field_names=["prev_state", "action",
                                              "reward", "curr_state",
                                              "done"])
    # Hyperparameter that we use to avoid some experiences
    # to have 0 probability of being taken
    self.err = 0.01
    # Hyperparameter that we use to make a tradeoff between
    # taking only exp with high priority and sampling randomly
    self.alpha = 0.6
    # importance-sampling, from initial value increasing to 1
    self.beta = 0.4
    self.beta_increment_per_sampling = 0.001
    # clipped abs error
    self.absolute_error_upper = 1.0

  def __len__(self) -> int:
    return self.sum_tree.n_entries

  def _get_priority(self, abs_err: np.float) -> np.float:
    # If the max priority = 0 we can't put priority = 0 since
    # this exp will never have a chance to be selected
    # so we use a minimum priority
    abs_errs = np.abs(abs_err) + self.err
    clipped_errors = np.minimum(abs_errs, self.absolute_error_upper)
    return clipped_errors ** self.alpha

  def update(self, tree_idx: int, abs_err: np.float) -> None:
    priority = self._get_priority(abs_err=abs_err)
    self.sum_tree.update(tree_idx=tree_idx, priority=priority)

  def push(self, abs_err: np.float, prev_state: np.ndarray, action: int,
           reward: int, curr_state: np.ndarray, done: bool) -> None:
    new_exp = self.transition(prev_state, action, reward, curr_state, done)
    priority = self._get_priority(abs_err=abs_err)
    self.sum_tree.add(priority, new_exp)

  def sample(self, batch_size: int) -> Tuple[List, List, List]:
    """
      return a sample of with length of batch size.
      batch contains transitions (experience).
      indices and is_weights are lists of floats.
    """
    batch = []
    indices = []
    priorities = []

    priority_segment = self.sum_tree.total_priority / batch_size
    self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

    for i in range(batch_size):
      a = priority_segment * i
      b = priority_segment * (i+1)
      uniform_sample = random.uniform(a, b)

      idx, priority, data = self.sum_tree.get(uniform_sample)
      indices.append(idx)
      priorities.append(priority)
      batch.append(data)

    # p(j)
    sampling_probabilities = priorities / self.sum_tree.total_priority
    # IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
    is_weights = np.power(self.sum_tree.n_entries * sampling_probabilities,
                          -self.beta)
    is_weights /= is_weights.max()

    return batch, indices, is_weights.tolist()


class Sumtree(object):
  def __init__(self, capacity: int) -> None:
    """
      A binary tree data structure where the value of a node is equal 
      to sum of the nodes present in its left subtree and right subtree.
    """
    self.capacity = capacity
    self.n_entries = 0
    self.data_pointer = 0

    # remember we are in a binary node (each node has max 2 children)
    # so 2x size of leaf (capacity) - 1 (root node)
    self.tree = np.zeros(2*capacity-1)
    # contains the experiences (so the size of data is capacity)
    self.data = np.zeros(capacity, dtype=object)

  def add(self, priority: int, new_data: namedtuple) -> None:
    """
      store the priority and experience
    """
    tree_idx = self.data_pointer + self.capacity - 1
    self.data[self.data_pointer] = new_data
    self.update(tree_idx, priority)

    self.data_pointer += 1
    # If we're above the capacity, you go back to first index (we overwrite)
    if self.data_pointer >= self.capacity:
      self.data_pointer = 0

    if self.n_entries < self.capacity:
      self.n_entries += 1

  def update(self, tree_idx: int, priority: int) -> None:
    """
      update the priority
    """
    change = priority - self.tree[tree_idx]
    self.tree[tree_idx] = priority

    # update the parent nodes
    while True:
      tree_idx = (tree_idx-1) // 2
      self.tree[tree_idx] += change
      if tree_idx == 0:
        break

  def get(self, value: int) -> Tuple[int, int, int]:
    """
      get the leaf index, priority value of that leaf, and experience associated with that index
    """
    parent_idx = 0
    left_child_idx = 2 * parent_idx + 1

    while left_child_idx < len(self.tree):
      left_child_idx = 2 * parent_idx + 1
      right_child_idx = left_child_idx + 1

      # downward search, always search for a higher priority node
      if value < self.tree[left_child_idx]:
        parent_idx = left_child_idx
      else:
        value -= self.tree[left_child_idx]
        parent_idx = right_child_idx

    data_idx = parent_idx - self.capacity + 1
    return parent_idx, self.tree[parent_idx], self.data[data_idx]

  @property
  def total_priority(self) -> np.float64:
    """
      find the sum of nodes by returning the root node
    """
    return self.tree[0]
