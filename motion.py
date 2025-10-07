#motion.py
import cv2

def detect_motion(frame, avg_frame, min_area=500, accum_weight=0.5):
    """
    Detect motion via running average background subtraction.

    Args:
        frame (ndarray): current BGR frame
        avg_frame (ndarray or None): running average of previous frames (float32)
        min_area (int): minimum contour area to register motion
        accum_weight (float): weight for updating the running average

    Returns:
        avg_frame (ndarray): updated running average
        motion (bool): True if motion detected
        roi (tuple or None): (x, y, w, h) of motion bounding box
    """
    # convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # first frame: initialize background model
    if avg_frame is None:
        return gray.astype("float"), False, None

    # update background model, then compute diff
    cv2.accumulateWeighted(gray, avg_frame, accum_weight)
    frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours and see if any exceed min_area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            return avg_frame, True, (x, y, w, h)

    return avg_frame, False, None