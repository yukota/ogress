"""
ogress.
"""
from urandom import random
from ucollections import OrderedDict
from math import isclose
import sys

from mini4wd import Machine
from microMLP import MicroMLP


machine = Machine()

# settin of machine
machine.setTireSize(31)
machine.setGainKp(0.1)
machine.setGainKi(0.02)
machine.setGainKd(0.02)


# utils
class Blink:
    def __init__(self, timing):
        self._timing = timing
        self._count = 0
        self._on = True

    def step(self):
        self._count += 1
        if self._count > self._timing:
            self._count = 0
            self._on = not self._on
        return self._on


# Q
class Q:
    class Example:
        def __init__(self, state, past_state, action, reward):
            self.state = state
            self.past_state = past_state
            self.action = action
            self.reward = reward

    """
    action 0 speed up
    action 1 speed keep
    action 2 speed down
    """
    STATE_NUM = 10
    ACTION_NUM = 3

    def __init__(self):
        self._exapmles = []
        self._target_rpm = 0
        self._episode = 0


        self._previous_observation = self._get_observation()
        self._previous_state = self._get_state(self._previous_observation)

        # create NN
        self._mlp = MicroMLP.Create(neuronsByLayers=[self.STATE_NUM, 3, self.ACTION_NUM],
                                    activationFuncName=MicroMLP.ACTFUNC_RELU,
                                    layersAutoConnectFunction=MicroMLP.LayersFullConnect,
                                    useBiasValue=1.0)

    def step(self):
        if self._episode < sys.maxsize:
            self._episode += 1

        # TODO: crash detector
        crashed = False
        # stop wheel
        if crashed:
            machine.setRpm(0)
        else:
            action = self.get_action(self._previous_state, self._episode)
            if action == 0:
                self._target_rpm += 10
            elif action == 2:
                self._target_rpm -= 10

            machine.setRpm(self._target_rpm)

        observation = self._get_observation()
        new_state = self._get_state(observation)
        reward = self._get_reward(observation, self._previous_observation)
        new_observation = observation

        # add learning example
        self._add_example(new_state, self._past_state, action, reward)

        self._previous_state = new_state
        self._previous_observation = new_observation



    def get_action(self, state, episode):

        eps = 0.5 * (1 / episode + 1)
        if eps < random():
            # suit act
            next_action = self._mpl.QLearningPredictBestActionIndex(state)
        else:
            next_action = random.choice([0, 1, 2])
        return next_action

    def get_observation(self):
        # wait for new sensor data
        machine.grab()
        ax = machine.getAx()
        ay = machine.getAy()
        az = machine.getAz()
        roll = machine.getRoll()
        pitch = machine.getPitch()
        yaw = machine.getYaw()
        rpm = machine.getRpm()
        voltage = machine.getVbat()
        ma = machine.getMotorCurrent()
        observation = OrderedDict([("ax", ax), ("ax", ay), ("az", az), ("roll", roll), ("pitch", pitch), ("yaw", yaw), ("rpm", rpm),
                                   ("voltage", voltage), ("ma", ma)])
        return observation

    def _get_reward(self, observation, previous_observation):
        """
        speed
        """
        reward = 0
        if observation["rpm"] > previous_observation["rpm"]:
            reward += 1.0
        elif isclose(["rpm"],  previous_observation["rpm"]):
            reward += .5
        return reward

    def _get_state(self, observtion):
        return list(observtion.values())

    def _add_example(self, state, past_state, action, reward):
        example = Q.Example(state, past_state, action, reward)
        self._exapmles.append(example)


q = Q()


class LearningState:

    def __init__(self):
        self._invoke_next = False

        self._running = False
        self._crashed = False

        self._led_blink = Blink(100)


    def pre_loop(self, sw0, sw1):
        if self._running:
            if sw0:
                self._invoke_next = True
            elif sw1:
                self._crashed = True
        else:
            if sw1:
                self._running = True
                self._crashed = False

    def loop(self):
        if not self._running:
            return
        # start running

        # blink in running
        if self._led_blink.step():
            machine.led(0x2)
        else:
            machine.led(0x0)

        q.step()

    def post_loop(self):
        pass

    def invoke_next(self):
        if self._invoke_next:
            return True
        else:
            return False

    def next_state(self):
        pass


class InitialState:
    def __init__(self):
        self._invoke_next = False
        machine.led(0x1)

    def pre_loop(self, sw0, sw1):
        if sw0:
            self._invoke_next = True

    def loop(self):
        pass

    def post_loop(self):
        pass

    def invoke_next(self):
        if self._invoke_next:
            return True
        else:
            return False

    def next_state(self):
        return LearningState()


class ModeState:
    def __init__(self):
        self._state = InitialState()

    def start_loop(self):
        while True:
            # check sw
            sw0, sw1 = self._check_sw()
            self._state.pre_loop(sw0, sw1)
            self._state.loop()
            self._state.post_loop()
            if self._state.invoke_next():
                self._state = self._state.next_state()

    def _check_sw(self):
        # left
        sw0 = machine.sw(0)
        # right
        sw1 = machine.sw(1)
        return sw0, sw1


def main():
    print("Start ogress")
    mode_state = ModeState()
    mode_state.start_loop()


main()
