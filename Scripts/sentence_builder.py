class SentenceBuilder:
    """
    Builds sentences from emitted words
    """

    def __init__(self, max_pause_frames=20):
        self.sentence = []
        self.pause_counter = 0
        self.MAX_PAUSE = max_pause_frames

    def update(self, emitted_word, has_hand, actions):
        """
        Args:
            emitted_word (int or None)
            has_hand (bool)
            actions (list of labels)

        Returns:
            finalized_sentence (str or None)
        """

        # ADD WORD 
        if emitted_word is not None:
            word = actions[emitted_word]

            # prevent duplicates
            if len(self.sentence) == 0 or self.sentence[-1] != word:
                self.sentence.append(word)

            # reset pause counter
            self.pause_counter = 0

        # NO HAND 
        elif not has_hand:
            self.pause_counter += 1

            # finalize sentence after pause
            if self.pause_counter >= self.MAX_PAUSE:
                if len(self.sentence) > 0:
                    final = " ".join(self.sentence)
                    self.sentence.clear()
                    self.pause_counter = 0
                    return final

        return None

    def get_current_sentence(self):
        return " ".join(self.sentence)
