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
            'powerup_above': self.is_powerup_above,
            'path_clear': self.is_path_clear,
            'multiple_goombas_ahead': self.are_multiple_goombas_ahead,
            'falling_goombas': self.are_goombas_falling,
        }

    def is_enemy_ahead(self, facts):
        return facts['is_enemy_ahead']
     
    def is_barrier_ahead(self, facts):
        return facts['is_barrier_ahead']

    def is_powerup_nearby(self, facts):
        return facts['is_powerup_nearby']

    def is_powerup_above(self, facts):
        return facts['is_powerup_above']

    def is_path_clear(self, facts):
        return facts['next_tile_clear'] == 0

    def are_multiple_goombas_ahead(self, facts):
        return facts['is_enemy_ahead'] and facts['distance_to_enemy'] is not None and facts['distance_to_enemy'] < 3

    def are_goombas_falling(self, facts):
        return facts['falling_goombas'] and facts['goombas_above_below_count'] > 0

class InferenceEngine:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.just_stepped_back = False  # Initialize just_stepped_back to track if Mario just stepped back

    def evaluate(self, facts):
        mario_positions = facts.get('mario_positions', None)  # Use mario_positions from facts
        
        if mario_positions is None or len(mario_positions) == 0:
            print("Mario is likely dead or the game has ended.")
            return WindowEvent.PRESS_BUTTON_START 
        
        # Calculate Mario's height based on the minimum row index (lower row number = higher on screen)
        mario_height = mario_positions[:, 0].min()

        # Rule 1: If there's a Goomba falling from above, slow down (or stop)
        if facts['falling_goombas']:
            print("Slowing down due to falling Goombas.")
            return WindowEvent.PRESS_ARROW_LEFT  # Slow down or stop

        # Rule 2: If there's a Goomba on the same level and close, jump over it
        if facts['is_enemy_ahead'] and facts['distance_to_enemy'] is not None and facts['distance_to_enemy'] <= 5:
            print("Jumping over Goomba.")
            return WindowEvent.PRESS_BUTTON_A  # JUMP action

        # Rule 3: If there's a barrier directly in front, jump to clear it
        if facts['is_barrier_ahead'] and not facts['next_tile_clear']:
            print("Jumping over barrier.")
            return WindowEvent.PRESS_BUTTON_A  # JUMP action

        # Rule 4: If there's a gap ahead, decide whether to step back or jump
        if facts['gap_ahead'] and facts['distance_to_gap'] is not None:
            if mario_height <= 5 and facts['distance_to_gap'] == 4:
                # Mario is high up and should jump from a greater distance
                print("Jumping over gap from a higher platform.")
                return WindowEvent.PRESS_BUTTON_A  # JUMP action to clear the gap
            elif mario_height > 5 and facts['distance_to_gap'] <= 2:
                # Mario is at ground level (or close to it) and should jump when closer
                print("Jumping over gap at ground level.")
                return WindowEvent.PRESS_BUTTON_A  # JUMP action to clear the gap

        # Rule 5: If there's a power-up above, jump and then wait
        if self.knowledge_base.rules['powerup_above'](facts):
            print("Jumping to get power-up.")
            return WindowEvent.PRESS_BUTTON_A  # JUMP action to get the power-up

        # Rule 6: If the path is clear, keep moving right
        if facts['next_tile_clear']:
            print("Path is clear, moving right.")
            self.just_stepped_back = False  # Reset stepping back tracker
            return WindowEvent.PRESS_ARROW_RIGHT  # MOVE RIGHT

        # Default action: Move right if unsure
        print("Default action: moving right.")
        self.just_stepped_back = False  # Reset stepping back tracker
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
        state = self.environment.game_state()
        game_area = np.array(self.environment.game_area())
        x_position = self.environment.get_x_position()

        # Find Mario's position (looking for the [1,1] block)
        mario_positions = np.argwhere(game_area == 1)

        print(f"Mario Positions: {mario_positions}")

        # Check if Mario's position is found
        if mario_positions.size == 0:
            print("Mario is likely dead or the game has ended.")
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

        print(f"Mario Position (min row/col): ({mario_min_row}, {mario_min_col})")

        # Detect Goombas directly ahead of Mario
        goomba_positions = np.argwhere(game_area == 15)
        print(f"Detected Goomba Positions: {goomba_positions}")

        goombas_ahead = goomba_positions[(goomba_positions[:, 0] >= mario_min_row) & 
                                        (goomba_positions[:, 0] <= mario_max_row) & 
                                        (goomba_positions[:, 1] > mario_max_col)]
        is_enemy_ahead = len(goombas_ahead) > 0
        distance_to_enemy = np.min(goombas_ahead[:, 1] - mario_max_col) if is_enemy_ahead else None

        # Determine if any Goombas are falling from above
        goombas_above = goomba_positions[goomba_positions[:, 0] < mario_min_row]
        falling_goombas = len(goombas_above) > 0

        # Check for barriers directly in front of Mario (same level)
        is_barrier_ahead = np.any(game_area[mario_min_row:mario_max_row + 2, mario_max_col + 1:mario_max_col + 2] == 14) or \
                        np.any(game_area[mario_min_row:mario_max_row + 2, mario_max_col + 1:mario_max_col + 2] == 10) or \
                        np.any(game_area[mario_min_row:mario_max_row + 2, mario_max_col + 1:mario_max_col + 2] == 12)

        print(f"Is Barrier Ahead: {is_barrier_ahead}")
        
        # Check if there's a gap in front of Mario
        ground_row = game_area.shape[0] - 1  # The last row is the ground
        gap_ahead = False
        distance_to_gap = None

        for col in range(mario_max_col + 1, game_area.shape[1]):
            if game_area[ground_row, col] == 0 and np.all(game_area[mario_max_row + 1:, col] == 0):
                gap_ahead = True
                distance_to_gap = col - mario_max_col
                break

        print(f"Is Gap Ahead: {gap_ahead}, Distance to Gap: {distance_to_gap}")
        
        # Check for power-up directly above Mario
        is_powerup_nearby = np.any(game_area[mario_min_row - 1:mario_min_row, mario_min_col:mario_max_col + 2] == 13)
        is_powerup_above = (378 <= x_position <= 385) and np.any(game_area[mario_min_row - 1:mario_min_row, mario_min_col:mario_max_col + 2] == 13)

        print(f"Is Powerup Nearby: {is_powerup_nearby}, Is Powerup Above: {is_powerup_above}")
        
        # Check if the immediate path ahead is clear
        next_tile_clear = np.all(game_area[mario_min_row:mario_max_row + 1, mario_max_col + 1:mario_max_col + 2] == 0)

        print(f"Is Next Tile Clear: {next_tile_clear}")
        print(game_area)

        # Collect facts about the current situation
        facts = {
            'state': state,
            'game_area': game_area,
            'mario_positions': mario_positions,  # Store Mario's positions in the facts
            'x_position': x_position,
            'is_enemy_ahead': is_enemy_ahead,
            'distance_to_enemy': distance_to_enemy,
            'is_barrier_ahead': is_barrier_ahead,
            'is_powerup_nearby': is_powerup_nearby,
            'is_powerup_above': is_powerup_above,
            'next_tile_clear': next_tile_clear,
            'game_over': False,  # Game is still running
            'falling_goombas': falling_goombas,
            'goombas_same_level_count': len(goombas_ahead),
            'gap_ahead': gap_ahead,
            'distance_to_gap': distance_to_gap,  # New fact for distance to gap
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
