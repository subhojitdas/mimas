import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diags
]

def winner(board: Tuple[int, ...]) -> int:
    """Return +1 if X won, -1 if O won, 0 otherwise."""
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return +1
        if s == -3:
            return -1
    return 0

def is_terminal(board: Tuple[int, ...]) -> bool:
    return winner(board) != 0 or all(v != 0 for v in board)

def legal_moves(board: Tuple[int, ...]) -> List[int]:
    return [i for i, v in enumerate(board) if v == 0]

def apply_move(board: Tuple[int, ...], player: int, move: int) -> Tuple[Tuple[int, ...], int]:
    b = list(board)
    b[move] = player
    return (tuple(b), -player)

def terminal_value(board: Tuple[int, ...]) -> float:
    """
    From X's perspective:
      +1 if X wins, -1 if O wins, 0 draw
    """
    w = winner(board)
    if w == +1:
        return 1.0
    if w == -1:
        return -1.0
    return 0.0

@dataclass
class Node:
    board: Tuple[int, ...]
    player: int  # player to move at this node
    parent: Optional["Node"] = None
    parent_move: Optional[int] = None

    children: Dict[int, "Node"] = field(default_factory=dict)  # move -> child
    untried_moves: List[int] = field(default_factory=list)

    visits: int = 0
    value_sum: float = 0.0  # accumulated rollout values (from root player's perspective)

    def __post_init__(self):
        if not self.untried_moves:
            self.untried_moves = legal_moves(self.board)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def best_child_uct(self, c: float) -> "Node":
        """
        Pick child that maximizes UCT score.
        Note: value_sum is stored from the ROOT player's perspective.
        """
        assert self.visits > 0, "UCT undefined for unvisited parent"

        best_score = -1e9
        best = None
        lnN = math.log(self.visits)

        for move, child in self.children.items():
            # Avoid division by zero: if child.visits == 0, treat as +inf UCT
            if child.visits == 0:
                score = float("inf")
            else:
                mean_value = child.value_sum / child.visits
                score = mean_value + c * math.sqrt(lnN / child.visits)

            if score > best_score:
                best_score = score
                best = child

        return best

    def expand(self) -> "Node":
        """Expand by taking one untried move and adding resulting child node."""
        move = self.untried_moves.pop()
        new_board, new_player = apply_move(self.board, self.player, move)
        child = Node(board=new_board, player=new_player, parent=self, parent_move=move)
        self.children[move] = child
        return child

def rollout_random(board: Tuple[int, ...], player: int) -> float:
    b, p = board, player
    while not is_terminal(b):
        m = random.choice(legal_moves(b))
        b, p = apply_move(b, p, m)
    return terminal_value(b)

def mcts_search(root_board: Tuple[int, ...], root_player: int, iters: int = 5000, c: float = 1.4) -> int:
    root = Node(board=root_board, player=root_player)
    root_sign = 1.0 if root_player == +1 else -1.0

    for _ in range(iters):
        node = root

        # 1) Selection
        while node.is_fully_expanded() and node.children and not is_terminal(node.board):
            node = node.best_child_uct(c)

        # 2) Expansion
        if not is_terminal(node.board) and not node.is_fully_expanded():
            node = node.expand()

        # 3) Simulation
        result_x = rollout_random(node.board, node.player)  # value from X perspective
        result_root = root_sign * result_x                  # value from root player perspective

        # 4) Backpropagation
        while node is not None:
            node.visits += 1
            node.value_sum += result_root
            node = node.parent

    # Pick move with most visits (robust choice)
    best_move, best_child = max(root.children.items(), key=lambda kv: kv[1].visits)
    return best_move

# -------------------------
# Demo: play vs MCTS
# -------------------------

def print_board(board: Tuple[int, ...]):
    def sym(v): return "X" if v == 1 else ("O" if v == -1 else ".")
    for r in range(3):
        print(" ".join(sym(board[3*r + c]) for c in range(3)))
    print()

def play_game():
    board = (0,) * 9
    player = +1  # X starts
    human = -1   # you play O

    while not is_terminal(board):
        print_board(board)
        if player == human:
            moves = legal_moves(board)
            print("Your moves:", moves)
            m = int(input("Choose move (0-8): "))
            if m not in moves:
                print("Illegal move.")
                continue
            board, player = apply_move(board, player, m)
        else:
            m = mcts_search(board, player, iters=3000, c=1.4)
            print("MCTS plays:", m)
            board, player = apply_move(board, player, m)

    print_board(board)
    w = winner(board)
    if w == 1:
        print("X wins")
    elif w == -1:
        print("O wins")
    else:
        print("Draw")

if __name__ == "__main__":
    play_game()



