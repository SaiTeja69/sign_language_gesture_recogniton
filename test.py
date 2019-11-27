from keras.models import load_model
import cv2
import numpy as np

CLASS_MAP = { #any new gesture has to be added here
    0:"hello",
    1:"none"
}

NUM_CLASSES = len(CLASS_MAP)


def mapper(val):
    return CLASS_MAP[val]

model = load_model("gesturecheck.h5")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    edges = cv2.Canny(frame,100,200)
    roi = edges[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    gesture_name = mapper(move_code)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, gesture_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Gesture Detection", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
