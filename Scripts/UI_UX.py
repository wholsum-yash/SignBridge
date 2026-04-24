import cv2

# Maps confidence → color 
def get_conf_color(conf):
    if conf >= 0.8:
        return (0, 255, 0)     # green
    elif conf >= 0.6:
        return (0, 255, 255)   # yellow
    else:
        return (0, 0, 255)     # red


# GENERIC BANNER 
def draw_banner(frame, text, x, y, w, h, bg_color, text_color, scale, thickness):
    overlay = frame.copy()

    # background box
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)

    # transparency blend
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # text
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


# MAIN UI 
def draw_ui(frame, word, sentence, confidence, state):
    h, w, _ = frame.shape

    # get confidence color ONCE (used everywhere)
    conf_color = get_conf_color(confidence if confidence is not None else 0)

    # WORD BANNER (TOP - PRIMARY) 
    if word:

        # dynamic but CLAMPED width
        min_w = 90
        max_w = 220
        banner_width = min(max_w, max(min_w, 60 + len(word) * 12))

        frame = draw_banner(
            frame,
            word,
            x = 30,
            y = 30,
            w=banner_width,
            h = 45,
            bg_color=(30, 30, 30),
            text_color=conf_color,   # color reflects confidence
            scale = 0.95,
            thickness = 3
        )

    # SENTENCE BANNER (BOTTOM - CAPTION STYLE) 
    if sentence:
        frame = draw_banner(
            frame,
            sentence,
            x=30,
            y=h - 100,              # moved to bottom → caption feel
            w=w - 60,
            h=60,
            bg_color=(20, 20, 20),
            text_color=(255, 255, 255),
            scale=0.8,
            thickness=2
        )

    # CONFIDENCE BAR (TOP LEFT) 
    if confidence is not None:
        bar_max_width = 150
        bar_width = int(bar_max_width * confidence)

        # background (empty bar)
        cv2.rectangle(
            frame,
            (30, 120),
            (30 + bar_max_width, 135),
            (80, 80, 80),
            -1
        )

        # filled portion (uses SAME confidence color)
        cv2.rectangle(
            frame,
            (30, 120),
            (30 + bar_width, 135),
            conf_color,
            -1
        )

    # STATE DEBUG (TOP RIGHT) 
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
