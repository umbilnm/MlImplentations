from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
@dataclass
class Node:
    """
    Represents a Trie node.

    Attributes
    ----------
    children : Dict[str, "Node"]
        Dictionary of child nodes.
    is_end : bool
        Indicates if this node is an ending character of a stored string.
    value : float
        Value associated with the stored string.
    """

    children: Dict[str, "Node"] = field(default_factory=dict)
    is_end: bool = False
    value: float = 0


@dataclass
class Trie:
    """
    A Trie data structure for efficient string manipulation and retrieval.

    Attributes
    ----------
    root : Node
        The root node of the Trie.
    """
    root: Node = field(default_factory=Node)

    def add_query(self, query: str, value: float) -> None:
        """
        Adds a single query string to the Trie.

        Parameters
        ----------
        query : str
            The string to be added to the Trie.
        value : float
            The value associated with the query.
        """
        node = self.root
        for char in query:
            if char not in node.children:
                node.children[char] = Node()
            node = node.children[char]
        node.value = value
        node.is_end = True

    def remove_query(self, query: str) -> None:
        """
        Removes a single query string from the Trie.

        Parameters
        ----------
        query : str
            The string to be removed from the Trie.

        Raises
        ------
        Exception:
            If the query is not found in the Trie.

        >>> raise Exception(f"Query {query} not found!")
        """
        node = self.root
        passed_nodes = [node]
        for char in query:
            if char not in node.children:
                raise Exception(f"Query {query} not found!")
            node = node.children[char]
            passed_nodes.append(node)
        if not node.is_end:
            raise Exception(f"Query {query} not found!")
        node.is_end = False
        node.value = None

        for i in range(len(passed_nodes)-1, -1, -1):
            if not node.is_end or node.children:
                break
            node = passed_nodes[i-1] 
            del node[i-1].children[query[i]]
         
    def clear(self) -> None:
        """Clears all the entries in the Trie."""
        self.root = Node()
    
    def _find_node(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def _collect_suffixes(self, node, prefix):
        suffixes = []
        if node.is_end:
            suffixes.append((node.value, prefix))
        for char, child in node.children:
            suffixes.extend(self.suffixes(child, prefix+char))
        return suffixes
    
    def suffixes(
        self,
        prefix: str,
    ) -> List[Tuple[float, str]]:
        """
        Returns all suffixes of the given prefix.

        Notes
        -----
        Here by suffix we mean string prefix + suffix.

        Parameters
        ----------
        prefix : str
            The prefix string.

        Returns
        -------
        List[Tuple[float, str]]
            List of (value, suffix) pairs.

        Examples
        --------
        Given queries: "apple", "app", "application", "triple"

        >>> trie = Trie()
        >>> trie.add_query("apple", 1.0)
        >>> trie.add_query("app", 2.0)
        >>> trie.add_query("application", 3.0)
        >>> trie.add_query("triple", 4.0)
        >>> trie.suffixes("app")
        [(3.0, 'application'), (2.0, 'app'), (1.0, 'apple')]
        """
        suffixes = []
        node = self._find_node(prefix)
        suffixes = self._collect_suffixes(node, prefix)
        return suffixes

    def _count_queries(self, node):
        count = 0
        if node.is_end:
            count +=1    
        for char, child in node.children:
            count += self._count_queries(child)
        return count
    
    def count_queries(self) -> int:
        """
        Returns the number of queries stored in the Trie.

        Returns
        -------
        int
            The number of queries stored in the Trie.
        """
        return self._count_queries(self.root)
