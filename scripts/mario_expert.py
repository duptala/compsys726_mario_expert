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
            'enemy_ahead': self.is_enemy_ahead,
            'barrier_ahead': self.is_barrier_ahead,
            'powerup_nearby': self.is_powerup_nearby,
            'path_clear': self.is_path_clear,
        }

    def is_enemy_ahead(self, facts):
        return facts['is_enemy_ahead']
     
    def is_barrier_ahead(self, facts):
        return facts['is_barrier_ahead']

    def is_powerup_nearby(self, facts):
        return facts['is_powerup_nearby']

    def is_path_clear(self, facts):
        return facts['next_tile_clear'] == 0

class InferenceEngine:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def evaluate(self, facts):
        # Rule 1: If an enemy is ahead, jump
        if self.knowledge_base.rules['enemy_ahead'](facts):
            return WindowEvent.PRESS_BUTTON_A  # JUMP action

        # Rule 2: If a barrier is ahead, jump
        if self.knowledge_base.rules['barrier_ahead'](facts):
            return WindowEvent.PRESS_BUTTON_A  # JUMP action

        # Rule 3: If a power-up is nearby and the path is clear, move towards it
        if self.knowledge_base.rules['powerup_nearby'](facts) and self.knowledge_base.rules['path_clear'](facts):
            return WindowEvent.PRESS_ARROW_UP  # MOVE RIGHT

        # Rule 4: If the path is clear, keep moving right
        if self.knowledge_base.rules['path_clear'](facts):
            return WindowEvent.PRESS_ARROW_RIGHT  # MOVE RIGHT

        # Default action: Move right if unsure
        return WindowEvent.PRESS_ARROW_RIGHT


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
        emulation_speed: int = 1,
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
        state = self.environment.game_state()
        game_area = np.array(self.environment.game_area())
        mario_pose = self.environment.get_mario_pose()
        x_position = self.environment.get_x_position()

        # Find Mario's position (looking for the [1,1] block)
        mario_positions = np.argwhere(game_area == 1)

        # Check if Mario's position is found
        if mario_positions.size == 0:
            # Mario is likely dead or the game has ended, return a 'game over' state
            facts = {
                'state': state,
                'game_over': True,  # Indicate that the game is over
            }
            return facts

        # Mario usually occupies two rows, get the min and max row and column
        mario_min_row = mario_positions[:, 0].min()
        mario_max_row = mario_positions[:, 0].max()
        mario_min_col = mario_positions[:, 1].min()
        mario_max_col = mario_positions[:, 1].max()

        # Check the tiles in front of Mario (same rows, next columns)
        next_tiles = game_area[mario_min_row:mario_max_row + 1, mario_max_col + 1:mario_max_col + 2]

        # Check for enemies, barriers, and power-ups
        is_enemy_ahead = np.any(next_tiles == 15)
        is_barrier_ahead = np.any(next_tiles == 14) or np.any(next_tiles == 10)
        is_powerup_on_top = np.any(game_area[mario_min_row - 1:mario_max_row, mario_min_col:mario_max_col + 2] == 13)
        next_tile_clear = np.all(next_tiles == 0)

        # Collect facts about the current situation
        facts = {
            'state': state,
            'game_area': game_area,
            'mario_pose': mario_pose,
            'x_position': x_position,
            'is_enemy_ahead': is_enemy_ahead,
            'is_barrier_ahead': is_barrier_ahead,
            'is_powerup_nearby': is_powerup_nearby,
            'next_tile_clear': next_tile_clear,
            'game_over': False,  # Game is still running
        }
        return facts

    def choose_action(self):
        print(self.environment.get_x_position())
        facts = self.gather_facts()
        action = self.inference_engine.evaluate(facts)
        return action

    def step(self):
        """
        Modify this function as required to implement the Mario Expert agent's logic.

        This is just a very basic example
        """
        # Choose an action - button press or other...
        action = self.choose_action()

        # If the action is None (e.g., when Mario is dead), don't attempt to run it
        if action is not None:
            # Run the action on the environment
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
