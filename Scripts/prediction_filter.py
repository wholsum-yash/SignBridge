from collections import deque

class Stabilizer:
    """
    Prediction Stabilization Layer
    Converts noisy frame-by-frame model predictions into stable outputs.

    INPUT:
        - class index (int)
        - confidence (float)

    OUTPUT:
        - confirmed class index OR None
    """

    def __init__(
        self,
        maxlen=10,
        conf_threshold=0.6,
        score_threshold=5.0
    ):
        """
        maxlen: size of temporal buffer
        conf_threshold: minimum confidence to consider a prediction
        score_threshold: total weighted confidence required to confirm a class
        """
        self.buffer = deque(maxlen=maxlen)
        self.conf_threshold = conf_threshold
        self.score_threshold = score_threshold

    # Adding new prediction
    def update(self, pred_class, confidence):
        """ Add a new prediction to the buffer """

        if pred_class is None:
            return

        self.buffer.append({
            "class": pred_class,
            "confidence": confidence
        })

    #  Filter weak signals
    def _get_valid_predictions(self):
        """ Keep only predictions above confidence threshold """

        return [
            p for p in self.buffer
            if p["confidence"] >= self.conf_threshold
        ]

    #  Weighted scoring
    def _compute_scores(self, valid_preds):
        """ Aggregate confidence scores per class """

        class_scores = {}

        for p in valid_preds:
            cls = p["class"]
            conf = p["confidence"]

            if cls not in class_scores:
                class_scores[cls] = 0.0

            class_scores[cls] += conf

        return class_scores

    # Final decision
    def get_output(self):
        """ Returns:  confirmed class index OR None """

        valid_preds = self._get_valid_predictions()

        if not valid_preds:
            return None

        class_scores = self._compute_scores(valid_preds)

        # Find best class
        best_class = max(class_scores, key=class_scores.get) # pyright: ignore[reportCallIssue, reportArgumentType]

        best_score = class_scores[best_class]

        # Stability condition
        if best_score >= self.score_threshold:
            return best_class

        return None

    #  Reset buffer (rare use)
    def reset(self):
        """ Clears buffer manually (avoid using frequently) """
        self.buffer.clear()

    # Debug helper (optional)
    def debug(self):
        """ Returns internal buffer for inspection """
        return list(self.buffer)
