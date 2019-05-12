"""
ogress.
"""
from mini4wd import Machine

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


class LearningState:
    def __init__(self):
        machine.led(0x2)

        self._running = False
        self._led_blink = Blink(100)

        # self._agent = Agent()
        self._target_rpm = 0

    def pre_loop(self, sw0, sw1):
        if sw0:
            self.next_state()

        elif sw1:
            self._running = True

    def loop(self):
        if not self._running:
            return
        # start running

        # blink in running
        if self._led_blink.step():
            machine.led(0x2)
        else:
            machine.led(0x0)

        # up = self._agent.step(self._target_rpm)
        up = True
        if up:
            self._target_rpm += 100
        else:
            self._target_rpm -= 100

        machine.setRpm(self._target_rpm)

    def post_loop(self):
        pass

    def invoke_next(self):
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
