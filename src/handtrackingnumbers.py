import cv2
import mediapipe as mp
import time

def fingers_up(hand_landmarks, handedness="Right"):
    fingers = []

    # thumb
    if handedness == "Right":
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    else:  # Left hand
        fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)

    # other 4 fingers
    for tip_id, pip_id in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(1 if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y else 0)

    return fingers  

def detect_numbers(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "0"
    elif fingers == [0, 1, 0, 0, 0]:
        return "1"
    elif fingers == [0, 1, 1, 0, 0]:
        return "2"
    elif fingers == [0, 0, 1, 1, 1]:
        return "3"
    elif fingers == [0, 1, 1, 1, 1]:
        return "4"
    elif fingers == [1, 1, 1, 1, 1]:
        return "5"
    else:
        return ""

def main():
    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        hand_numbers = []

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            for idx, handLms in enumerate(results.multi_hand_landmarks):
                handedness_label = results.multi_handedness[idx].classification[0].label
                fingers = fingers_up(handLms, handedness_label)
                number = detect_numbers(fingers)

                # store number as int (default 0 if empty)
                hand_numbers.append(int(number) if number != "" else 0)

                # draw landmarks
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                draw = True 

                if draw:
                    fingertip_ids = [4, 8, 12, 16, 20]
                    for tip_id in fingertip_ids:
                        cx , cy = int(handLms.landmark[tip_id].x * img.shape[1]) , int(handLms.landmark[tip_id].y * img.shape[0])
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        
                # show number near the hand
                h, w, _ = img.shape
                cx, cy = int(handLms.landmark[0].x * w), int(handLms.landmark[0].y * h)
                if number != "":
                    cv2.putText(img, f'{number}', (cx - 90, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            # if 2 hands, show sum at top
            if num_hands == 2:
                total = sum(hand_numbers)
                cv2.putText(img, f'Sum: {total}', (img.shape[1]//2 - 50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # show FPS
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.imshow("Hand Tracking Numbers", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

