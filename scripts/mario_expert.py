"""
This the primary class for the Mario Expert agent. It contains the logic for the Mario Expert agent to play the game and choose actions.

Your goal is to implement the functions and methods required to enable choose_action to select the best action for the agent to take.

Original Mario Manual: https://www.thegameisafootarcade.com/wp-content/uploads/2017/04/Super-Mario-Land-Game-Manual.pdf
"""

import json
import logging
import random
import cv2
from mario_environment import MarioEnvironment
from pyboy.utils import WindowEvent
import numpy as np

class KnowledgeBase:
    def __init__(self):
        self.rules = {
            'goomba_below': self.rule_goomba_below,
            'falling_goomba': self.rule_falling_goomba,
            'enemy_ahead': self.rule_enemy_ahead,
            'barrier_ahead': self.rule_barrier_ahead,
            'gap_ahead': self.rule_gap_ahead,
            'powerup_above': self.rule_powerup_above,
            'path_clear': self.rule_path_clear,
        }

    def rule_goomba_below(self, facts):
        goombas_below = facts['goombas_below']
        if goombas_below.size > 0 and facts['distance_to_goomba_below'] is not None:
            distance_to_goomba_below = facts['distance_to_goomba_below']
            goomba_x = goombas_below[0][1]

            if distance_to_goomba_below <= 3 and facts['mario_x_min'] <= goomba_x <= facts['mario_x_max']:
                if distance_to_goomba_below == 1:
                    return WindowEvent.PRESS_BUTTON_A  # Stomp the Goomba
                return WindowEvent.PRESS_ARROW_DOWN  # Wait for Goomba to get closer
        return None

    def rule_falling_goomba(self, facts):
        if facts['falling_goombas']:
            return WindowEvent.PRESS_ARROW_LEFT  # Slow down or stop
        return None

    def rule_enemy_ahead(self, facts):
        if facts['is_enemy_ahead'] and facts['distance_to_enemy'] <= 5:
            return WindowEvent.PRESS_BUTTON_A  # Jump over the enemy
        return None

    def rule_barrier_ahead(self, facts):
        if facts['is_barrier_ahead'] and not facts['next_tile_clear']:
            return WindowEvent.PRESS_BUTTON_A  # Jump over the barrier
        return None

    def rule_gap_ahead(self, facts):
        if facts['gap_ahead'] and facts['distance_to_gap'] is not None:
            if facts['mario_height'] <= 5 and facts['distance_to_gap'] == 4:
                return WindowEvent.PRESS_BUTTON_A  # Jump over gap from a higher platform
            elif facts['mario_height'] > 5 and facts['distance_to_gap'] <= 1:
                return WindowEvent.PRESS_BUTTON_A  # Jump over gap at ground level
        return None

    def rule_powerup_above(self, facts):
        if facts['is_powerup_above']:
            return WindowEvent.PRESS_BUTTON_A  # Jump to get the power-up
        return None

    def rule_path_clear(self, facts):
        if facts['next_tile_clear']:
            return WindowEvent.PRESS_ARROW_RIGHT  # Move right
        return None

class InferenceEngine:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def evaluate(self, facts):
        # Apply each rule from the knowledge base
        for rule_name, rule_function in self.knowledge_base.rules.items():
            action = rule_function(facts)
            if action:
                return action

        return WindowEvent.PRESS_ARROW_RIGHT  # Default action


class MarioController(MarioEnvironment):
    """
    The MarioController class represents a controller for the Mario game environment.

    You can build upon this class all you want to implement your Mario Expert agent.

    Args:
        act_freq (int): The frequency at which actions are performed. Defaults to 10.
        emulation_speed (int): The speed of the game emulation. Defaults to 0.
        headless (bool): Whether to run the game in headless mode. Defaults to False.
    """

    def __init__(
        self,
        act_freq: int = 10,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:
        super().__init__(
            act_freq=act_freq,
            emulation_speed=emulation_speed,
            headless=headless,
        )

        self.act_freq = act_freq

        # Example of valid actions based purely on the buttons you can press
        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        self.valid_actions = valid_actions
        self.release_button = release_button

    def run_action(self, action: int) -> None:
        """
        This is a very basic example of how this function could be implemented

        As part of this assignment your job is to modify this function to better suit your needs

        You can change the action type to whatever you want or need just remember the base control of the game is pushing buttons
        """

        action_index = self.valid_actions.index(action)
        self.pyboy.send_input(action)
        for _ in range(self.act_freq):
            self.pyboy.tick()
        self.pyboy.send_input(self.release_button[action_index])

class MarioExpert:
    """
    The MarioExpert class represents an expert agent for playing the Mario game.

    Edit this class to implement the logic for the Mario Expert agent to play the game.

    Do NOT edit the input parameters for the __init__ method.

    Args:
        results_path (str): The path to save the results and video of the gameplay.
        headless (bool, optional): Whether to run the game in headless mode. Defaults to False.
    """

    def __init__(self, results_path: str, headless=False):
        self.results_path = results_path
        self.environment = MarioController(headless=headless)
        self.knowledge_base = KnowledgeBase()
        self.inference_engine = InferenceEngine(self.knowledge_base)
        self.video = None
    
    def gather_facts(self):
        game_area = np.array(self.environment.game_area())

        # Find Mario's position
        mario_positions = np.argwhere(game_area == 1)
        if mario_positions.size == 0:
            return None  # If Mario's position is not found, we skip this step

        mario_min_row = mario_positions[:, 0].min()
        mario_max_row = mario_positions[:, 0].max()
        mario_min_col = mario_positions[:, 1].min()
        mario_max_col = mario_positions[:, 1].max()

        # Gather essential facts about the environment
        goomba_positions = np.argwhere(game_area == 15)

        goombas_ahead = goomba_positions[(goomba_positions[:, 0] >= mario_min_row) & 
                                         (goomba_positions[:, 0] <= mario_max_row) & 
                                         (goomba_positions[:, 1] > mario_max_col)]
        is_enemy_ahead = len(goombas_ahead) > 0
        distance_to_enemy = np.min(goombas_ahead[:, 1] - mario_max_col) if is_enemy_ahead else None

        goombas_below = goomba_positions[(goomba_positions[:, 0] > mario_max_row) & 
                                         (goomba_positions[:, 1] >= mario_min_col) & 
                                         (goomba_positions[:, 1] <= mario_max_col)]
        distance_to_goomba_below = np.min(goombas_below[:, 0] - mario_max_row) if len(goombas_below) > 0 else None

        goombas_above = goomba_positions[goomba_positions[:, 0] < mario_min_row]
        falling_goombas = len(goombas_above) > 0

        is_barrier_ahead = np.any(game_area[mario_min_row:mario_max_row + 2, mario_max_col + 1:mario_max_col + 2] == 14) or \
                           np.any(game_area[mario_min_row:mario_max_row + 2, mario_max_col + 1:mario_max_col + 2] == 10) or \
                           np.any(game_area[mario_min_row:mario_max_row + 2, mario_max_col + 1:mario_max_col + 2] == 12)

        # Check if there's a gap ahead
        ground_row = game_area.shape[0] - 1
        gap_ahead = False
        distance_to_gap = None

        for col in range(mario_max_col + 1, game_area.shape[1]):
            if game_area[ground_row, col] == 0 and np.all(game_area[mario_max_row + 1:, col] == 0):
                gap_ahead = True
                distance_to_gap = col - mario_max_col
                break

        is_powerup_above = np.any(game_area[mario_min_row - 1:mario_min_row, mario_min_col:mario_max_col + 2] == 13)

        next_tile_clear = np.all(game_area[mario_min_row:mario_max_row + 1, mario_max_col + 1:mario_max_col + 2] == 0)

        # Collect facts
        facts = {
            'mario_positions': mario_positions,
            'mario_x_min': mario_min_col,
            'mario_x_max': mario_max_col,
            'mario_height': mario_min_row,
            'is_enemy_ahead': is_enemy_ahead,
            'distance_to_enemy': distance_to_enemy,
            'is_barrier_ahead': is_barrier_ahead,
            'is_powerup_above': is_powerup_above,
            'next_tile_clear': next_tile_clear,
            'falling_goombas': falling_goombas,
            'goombas_below': goombas_below,
            'distance_to_goomba_below': distance_to_goomba_below,
            'gap_ahead': gap_ahead,
            'distance_to_gap': distance_to_gap,
        }
        return facts

    def choose_action(self):
        # print(self.environment.get_x_position())
        facts = self.gather_facts()
        if not facts:
            return None # no action if facts can't be generated
        action = self.inference_engine.evaluate(facts)
        return action

    def step(self):
        """
        Modify this function as required to implement the Mario Expert agent's logic.

        This is just a very basic example
        """
        action = self.choose_action()
        if action is not None:
            self.environment.run_action(action)

    def play(self):
        """
        Do NOT edit this method.
        """
        self.environment.reset()

        frame = self.environment.grab_frame()
        height, width, _ = frame.shape

        self.start_video(f"{self.results_path}/mario_expert.mp4", width, height)

        while not self.environment.get_game_over():
            frame = self.environment.grab_frame()
            self.video.write(frame)

            self.step()

        final_stats = self.environment.game_state()
        logging.info(f"Final Stats: {final_stats}")

        with open(f"{self.results_path}/results.json", "w", encoding="utf-8") as file:
            json.dump(final_stats, file)

        self.stop_video()

    def start_video(self, video_name, width, height, fps=30):
        """
        Do NOT edit this method.
        """
        self.video = cv2.VideoWriter(
            video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    def stop_video(self) -> None:
        """
        Do NOT edit this method.
        """
        self.video.release()
