# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array


class GameState(NamedTuple):
    color: Array = jnp.int32(0)
    # 9x9 board
    # [[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
    #  [ 9, 10, 11, 12, 13, 14, 15, 16, 17],
    #  [18, 19, 20, 21, 22, 23, 24, 25, 26],
    #  [27, 28, 29, 30, 31, 32, 33, 34, 35],
    #  [36, 37, 38, 39, 40, 41, 42, 43, 44],
    #  [45, 46, 47, 48, 49, 50, 51, 52, 53],
    #  [54, 55, 56, 57, 58, 59, 60, 61, 62],
    #  [63, 64, 65, 66, 67, 68, 69, 70, 71],
    #  [72, 73, 74, 75, 76, 77, 78, 79, 80]]
    board: Array = -jnp.ones(81, jnp.int32)  # -1 (empty), 0, 1 
    winner: Array = jnp.int32(-1)
    round : Array = jnp.int32(0)


class Game:
    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:

        board2d = state.board.reshape(9, 9)

        num_filled = (board2d[:, action] >= 0).sum()
        board2d = board2d.at[8 - num_filled, action].set(state.color)
        won = ((board2d.flatten()[IDX] == state.color).all(axis=1)).any()
        winner = jax.lax.select(won, state.color, -1)

        state = state._replace(
            board=board2d.flatten(),
            winner=winner,
            round=state.round + 1,
            color=1 - state.color,
        )

        state = jax.lax.cond(
            (jnp.mod(state.round, jnp.int32(3)) == 0) & jnp.logical_not(won),
            lambda state: _rotate_update_state(state),
            lambda state: state,
            state
        )
        return state 


    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        def make(turn):
            return state.board.reshape(9, 9) == turn

        turns = jax.lax.select(color == 0, jnp.int32([0, 1]), jnp.int32([1, 0]))
        return jnp.stack(jax.vmap(make)(turns), -1)

    def legal_action_mask(self, state: GameState) -> Array:
        board2d = state.board.reshape(9, 9)
        return (board2d >= 0).sum(axis=0) < 9

    def is_terminal(self, state: GameState) -> Array:
        board2d = state.board.reshape(9, 9)
        return (state.winner >= 0) | jnp.all((board2d >= 0).sum(axis=0) == 9)

    def rewards(self, state: GameState) -> Array:
        return jnp.select(
            jnp.array([state.winner == 0, state.winner == 1]),
            jnp.array([[1.0, -1.0], [-1.0, 1.0]]),
            jnp.array([0.0, 0.0])
        )

def _check_winner(board: Array, player: Array) -> Array:
    won = ((board.flatten()[IDX] == player).all(axis=1)).any()
    return won

def _rotate_update_state(state: GameState) -> GameState:
    board = state.board.reshape(9, 9)
    board = _rotate_and_apply_gravity(board)
    won_0 = _check_winner(board, 0)
    won_1 = _check_winner(board, 1)
    winner = jnp.select(jnp.array([won_0 & won_1, won_0, won_1]), jnp.array([2, 0, 1]), -1)
    return state._replace(
        winner=winner,
        board=board.flatten(),
    )

def _apply_gravity_column_sort(col : Array) -> Array:
    mask = (col == -1)
    order = jnp.argsort(mask, stable=True, descending=True)
    final = col[order]
    return final

_apply_gravity_board_sort = jax.vmap(_apply_gravity_column_sort, in_axes=-1, out_axes=-1)

def _rotate_and_apply_gravity(board: Array) -> Array:
    rotated_board = jnp.rot90(board, -1)
    gravity_board = _apply_gravity_board_sort(rotated_board)
    return gravity_board

def _make_win_cache():
    idx = []
    # Vertical
    for i in range(4):
        for j in range(9):
            a = i * 9 + j
            idx.append([a, a + 9, a + 18, a + 27, a + 36])
    # Horizontal
    for i in range(9):
        for j in range(4):
            a = i * 9 + j
            idx.append([a, a + 1, a + 2, a + 3, a + 4])

    # Diagonal
    for i in range(4):
        for j in range(4):
            a = i * 9 + j
            idx.append([a, a + 10, a + 20, a + 30, a + 40])
    for i in range(4):
        for j in range(4, 9):
            a = i * 9 + j
            idx.append([a, a + 8, a + 16, a + 24, a + 32])
    return jnp.int32(idx)


IDX = _make_win_cache()
