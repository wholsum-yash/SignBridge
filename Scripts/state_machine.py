class StateMachine:
    """
    Controls when a prediction is allowed to be emitted.

    INPUT:
        has_hand (bool)
        stable_pred (int or None)

    OUTPUT:
        emitted_word (int or None)
    """

    def __init__(self, cooldown_frames=15):
        self.state = "IDLE"
        self.cooldown_counter = 0
        self.COOLDOWN_FRAMES = cooldown_frames

    def update(self, has_hand, stable_pred):
        """
        Main update loop

        Args:
            has_hand (bool): whether hand is detected
            stable_pred (int or None): output from stabilizer

        Returns:
            emitted_word (int or None)
        """

        # IDLE STATE
        if self.state == "IDLE":
            if has_hand:
                self.state = "DETECTING"

        # DETECTING STATE 
        elif self.state == "DETECTING":
            if not has_hand:
                self.state = "IDLE"

            elif stable_pred is not None:
                self.state = "CONFIRMED"
                return stable_pred

        # CONFIRMED STATE 
        elif self.state == "CONFIRMED":
            self.state = "COOLDOWN"
            self.cooldown_counter = self.COOLDOWN_FRAMES

        # COOLDOWN STATE
        elif self.state == "COOLDOWN":
            self.cooldown_counter -= 1

            if self.cooldown_counter <= 0:
                if has_hand:
                    self.state = "DETECTING"
                else:
                    self.state = "IDLE"

        return None

    # optional debug
    def get_state(self):
        return self.state
