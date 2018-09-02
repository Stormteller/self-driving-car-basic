from reinforcement.agent import Agent


# State: {image, speed, throttle, steer}
class Environment:
    def __init__(self):
        self.actions = [(-0.1, -0.1), (-0.1, 0), (-0.1, 0.1),
                        (0, -0.1), (0, 0), (0, 0.1),
                        (0.1, -0.1), (0.1, 0), (0.1, 0.1)] # steer and throttle

        self.state_image_shape = (66, 200, 3)
        self.action_size = 9   # Combinations of Steer:(-0.1, 0, 0.1) and Throttle: (-0.1, 0, 0.1)
        self.agent = Agent(self.state_image_shape, self.action_size)

    def reward(self, state, action):

        return 0

    def start(self):
        pass