class StateMachine:
    def __init__(self, cooldown_frames=8, hand_presence_frames=3):
        self.state = "IDLE"
        self.cooldown_counter = 0
        self.COOLDOWN_FRAMES = cooldown_frames

        # NEW: require stable hand presence
        self.hand_counter = 0
        self.HAND_FRAMES = hand_presence_frames

    def update(self, has_hand, stable_pred):

        # STATE: IDEL 
        if self.state == "IDLE":
            if has_hand:
                self.hand_counter += 1
            else:
                self.hand_counter = 0

            # Only enter detection if hand is stable
            if self.hand_counter >= self.HAND_FRAMES:
                self.state = "DETECTING"

        # STATE: DETECTING 
        elif self.state == "DETECTING":
            if not has_hand:
                self.state = "IDLE"
                self.hand_counter = 0

            elif stable_pred is not None:
                self.state = "CONFIRMED"
                return stable_pred

        # STATE: CONFIRMED 
        elif self.state == "CONFIRMED":
            self.state = "COOLDOWN"
            self.cooldown_counter = self.COOLDOWN_FRAMES

        # STATE: COOLDOWN 
        elif self.state == "COOLDOWN":
            self.cooldown_counter -= 1

            if self.cooldown_counter <= 0:
                self.state = "IDLE"
                self.hand_counter = 0

        return None

    def get_state(self):
        return self.state
