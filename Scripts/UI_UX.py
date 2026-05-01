import cv2


def get_conf_color(conf):
    if conf >= 0.8:
        return (0, 255, 0)
    elif conf >= 0.6:
        return (0, 255, 255)
    else:
        return (0, 0, 255)


def draw_banner(frame, text, x, y, w, h, bg_color, text_color, scale, thickness):
    overlay = frame.copy()

    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)

    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(
        frame,
        text,
        (x + 15, y + int(h * 0.65)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

    return frame


def draw_ui(frame, word, sentence, confidence, state):
    h, w, _ = frame.shape

    conf_color = get_conf_color(confidence if confidence is not None else 0)

    # word
    if word:
        banner_width = min(220, max(90, 60 + len(word) * 12))

        frame = draw_banner(
            frame,
            word,
            30,
            30,
            banner_width,
            45,
            (30, 30, 30),
            conf_color,
            0.95,
            3
        )

    # sentence stack
    if sentence:

        line_height = 28
        padding = 15

        num_lines = len(sentence)
        box_height = padding * 2 + line_height * num_lines

        y_start = h - box_height - 20

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (30, y_start),
            (w - 30, y_start + box_height),
            (20, 20, 20),
            -1
        )

        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # draw bottom-up
        for i, line in enumerate(reversed(sentence)):
            y = y_start + box_height - padding - (i * line_height)

            cv2.putText(
                frame,
                line,
                (45, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    # confidence bar
    if confidence is not None:
        bar_max_width = 150
        bar_width = int(bar_max_width * confidence)

        cv2.rectangle(frame, (30, 120), (30 + bar_max_width, 135), (80, 80, 80), -1)
        cv2.rectangle(frame, (30, 120), (30 + bar_width, 135), conf_color, -1)

    # state
    if state:
        cv2.putText(
            frame,
            state,
            (w - 150, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

    return frame
