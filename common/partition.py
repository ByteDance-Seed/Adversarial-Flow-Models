"""
Partition utility functions.
"""

from heapq import heappush, heappop
from typing import Any, List


def partition_by_size(data: List[Any], size: int) -> List[List[Any]]:
    """
    Partition a list by size.
    When indivisible, the last group contains fewer items than the target size.

    Examples:
        - data: [1,2,3,4,5]
        - size: 2
        - return: [[1,2], [3,4], [5]]
    """
    assert size > 0
    return [data[i : (i + size)] for i in range(0, len(data), size)]


def partition_by_groups(data: List[Any], groups: int) -> List[List[Any]]:
    """
    Partition a list by groups.
    When indivisible, some groups may have more items than others.

    Examples:
        - data: [1,2,3,4,5]
        - groups: 2
        - return: [[1,3,5], [2,4]]
    """
    assert groups > 0
    return [data[i::groups] for i in range(groups)]


def partition_by_groups_weighted(items, weights, n_groups):
    """
    Partition a list by groups balanced by weights.
    
    Examples:
        - data: [1, 2, 3, 4, 5, 6, 7, 8]
        - weights: [30, 20, 100, 90, 10, 100, 10, 10]
        - groups: 4
        - return: [[3], [6], [4], [1, 2, 5, 7, 8]]
    """
    # initialize groups as min-heap: (total_weight, group_index, items_list)
    groups = [(0, i, []) for i in range(n_groups)]
    
    # sort items by descending weight
    for item, w in sorted(zip(items, weights), key=lambda x: -x[1]):
        total, i, group = heappop(groups)  # get group with smallest total weight
        group.append(item)
        heappush(groups, (total + w, i, group))
    
    # sort back by group index
    groups.sort(key=lambda x: x[1])
    return [g for _, _, g in groups]


def shift_list(data: List[Any], n: int) -> List[Any]:
    """
    Rotate a list by n elements.

    Examples:
        - data: [1,2,3,4,5]
        - n: 3
        - return: [4,5,1,2,3]
    """
    return data[(n % len(data)) :] + data[: (n % len(data))]
