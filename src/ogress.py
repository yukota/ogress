"""
ogress.
"""
from urandom import random, randint, choice

from mini4wd import Machine
from microMLP import MicroMLP
from pyb import millis, elapsed_millis

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
    STATE_NUM = 9
    HIDDEN_NUM = 12
    ACTION_NUM = 3

    # State max num
    AX_MAX = 10
    AY_MAX = 10
    AZ_MAX = 10
    ROLL_MAX = 10
    PITCH_MAX = 10
    YAW_MAX = 10
    RPM_MAX = 5000
    VOLTAGE_MAX = 3000
    MA_MAX = 2000

    AX_MIN = 0
    AY_MIN = 0
    AZ_MIN = 0
    ROLL_MIN = 0
    PITCH_MIN = 0
    YAW_MIN = 0
    RPM_MIN = 0
    VOLTAGE_MIN = 0
    MA_MIN = 0

    def __init__(self):
        self._examples = []
        self._target_rpm = self.RPM_MAX // 2
        self._episode = 0
        self._first_step = True

        self._previous_observation = None
        self._previous_state = None

        self._state_max = {
            "ax": self.AX_MAX, "ay": self.AY_MAX, "az": self.AZ_MAX, "roll": self.ROLL_MAX, "pitch": self.PITCH_MAX,
            "yaw": self.YAW_MAX, "rpm": self.RPM_MAX, "voltage": self.VOLTAGE_MAX, "ma": self.MA_MAX}

        self._state_min = {
            "ax": self.AX_MIN, "ay": self.AY_MIN, "az": self.AZ_MIN, "roll": self.ROLL_MIN, "pitch": self.PITCH_MIN,
            "yaw": self.YAW_MIN, "rpm": self.RPM_MIN, "voltage": self.VOLTAGE_MIN, "ma": self.MA_MIN}

        # create NN
        self._mlp = MicroMLP.Create(neuronsByLayers=[self.STATE_NUM, self.HIDDEN_NUM, self.ACTION_NUM],
                                    activationFuncName=MicroMLP.ACTFUNC_RELU,
                                    layersAutoConnectFunction=MicroMLP.LayersFullConnect,
                                    useBiasValue=1.0)

    def step(self):

        if self._first_step:
            self._target_rpm = self.RPM_MAX // 2
            machine.setRpm(self._target_rpm)
            self._previous_observation = self._get_observation()
            self._previous_state = self._get_state(self._previous_observation)
            self._first_step = False
            return

        if self._episode < 10000000:
            self._episode += 1

        # TODO: crash detector
        crashed = False
        # stop wheel
        if crashed:
            machine.setRpm(0)
            return
        else:
            action = self._get_action(self._previous_state, self._episode)
            if action == 0:
                self._target_rpm += 10
            elif action == 2:
                self._target_rpm -= 10
            self._target_rpm = max(0, self._target_rpm)
            print("target rpm: {}".format(self._target_rpm))
            machine.setRpm(self._target_rpm)


        observation = self._get_observation()
        new_state = self._get_state(observation)
        reward = self._get_reward(observation, self._previous_observation)
        print("reward: {}".format(reward.AsAnalogSignal))
        new_observation = observation

        # add learning example
        self._add_example(new_state, self._previous_state, action, reward)

        if len(self._examples) > 50:
            # randomness for dqn
            self._learn()

        self._previous_state = new_state
        self._previous_observation = new_observation

    def _get_action(self, state, episode):
        eps = 0.9 * (1 / episode * 0.001 + 1)
        if eps < random():
            # suit act
            next_action = self._mlp.QLearningPredictBestActionIndex(state)
            if next_action is None:
                next_action = choice([0, 1, 2])
        else:
            next_action = choice([0, 1, 2])
        return next_action

    def _get_observation(self):
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
        observation = {
            "ax": ax, "ay": ay, "az": az, "roll": roll, "pitch": pitch,
            "yaw": yaw, "rpm": rpm, "voltage": voltage, "ma": ma}
        return observation

    def _get_reward(self, observation, previous_observation):
        """
        speed
        """
        reward = 0
        if observation["rpm"] > previous_observation["rpm"]:
            reward += 1.0
        elif abs(observation["rpm"] - previous_observation["rpm"]) < 10:
            reward += .5
        nn_reward = MicroMLP.NNValue.FromAnalogSignal(reward)
        return nn_reward

    def _get_state(self, observation):
        # update max values
        for k, v in observation.items():
            if self._state_max[k] < v:
                self._state_max[k] = v
                print("max " + k + " to " + str(v))

            if self._state_min[k] > v:
                self._state_min[k] = v
                print("min " + k + " to " + str(v))

        state = [MicroMLP.NNValue(self._state_min['ax'], self._state_max['ax'], observation['ax']),
                 MicroMLP.NNValue(self._state_min['ay'], self._state_max['ay'], observation['ay']),
                 MicroMLP.NNValue(self._state_min['az'], self._state_max['az'], observation['az']),
                 MicroMLP.NNValue(self._state_min['roll'], self._state_max['roll'], observation['roll']),
                 MicroMLP.NNValue(self._state_min['pitch'], self._state_max['pitch'], observation['pitch']),
                 MicroMLP.NNValue(self._state_min['yaw'], self._state_max['yaw'], observation['yaw']),
                 MicroMLP.NNValue(self._state_min['rpm'], self._state_max['rpm'], observation['rpm']),
                 MicroMLP.NNValue(self._state_min['voltage'], self._state_max['voltage'], observation['voltage']),
                 MicroMLP.NNValue(self._state_min['ma'], self._state_max['ma'], observation['ma'])
                 ]
        return state

    def _add_example(self, state, past_state, action, reward):
        example = Q.Example(state, past_state, action, reward)
        self._examples.append(example)

    def _learn(self):
        index = randint(0, len(self._examples) - 1)
        example = self._examples.pop(index)
        is_success = self._mlp.QLearningLearnForChosenAction(example.state, example.reward, example.past_state,
                                                             example.action)
        if not is_success:
            print("Failed to learn")


q = Q()


class LearningState:

    def __init__(self):
        machine.led(0x1)
        self._invoke_next = False

        self._running = False

        self._led_blink = Blink(10)

    def pre_loop(self, sw0, sw1):
        if self._running:
            if sw0:
                # Stop
                pass
        else:
            if sw0:
                # Start
                self._running = True

    def loop(self):
        if not self._running:
            machine.led(0x1)
            machine.setRpm(0)
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


class ModeState:
    def __init__(self):
        self._state = LearningState()

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
