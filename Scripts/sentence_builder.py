class SentenceBuilder:
    """
    Multi-line sentence builder with expiration
    """

    def __init__(self, max_pause_frames=20, expiry_frames=300):
        self.lines = []              # list of sentence lines
        self.current_line = []      # active sentence
        self.pause_counter = 0
        self.no_hand_counter = 0

        self.MAX_PAUSE = max_pause_frames
        self.EXPIRY_FRAMES = expiry_frames  # ~10 sec @30fps

    def update(self, emitted_word, has_hand, actions):

        # add word
        if emitted_word is not None:
            word = actions[emitted_word]

            if len(self.current_line) == 0 or self.current_line[-1] != word:
                self.current_line.append(word)

            self.pause_counter = 0
            self.no_hand_counter = 0

        # no hand
        elif not has_hand:
            self.pause_counter += 1
            self.no_hand_counter += 1

            # finalize line after pause
            if self.pause_counter >= self.MAX_PAUSE:
                if len(self.current_line) > 0:
                    line = " ".join(self.current_line)
                    self.lines.append(line)
                    self.current_line.clear()
                    self.pause_counter = 0

        else:
            self.no_hand_counter = 0

        # expiry
        if self.no_hand_counter >= self.EXPIRY_FRAMES:
            self.lines.clear()
            self.current_line.clear()
            self.no_hand_counter = 0

        return None

    def get_display_lines(self):
        # include active line at bottom
        lines = self.lines.copy()

        if len(self.current_line) > 0:
            lines.append(" ".join(self.current_line))

        return lines[-5:]  # show last 5 lines max
