from typing import List


def check_list_lengths(lists: List[List]) -> None:
    """
    Check if input lists are all of same length.
    Parameters
    ----------
    lists : List of lists of different possible types.
    """
    it = iter(lists)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError("Not all lists have same length!")
