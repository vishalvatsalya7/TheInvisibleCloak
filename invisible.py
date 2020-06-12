import cv2
import  numpy as np
import time

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('../video/output1.avi', fourcc, 20.0, (640, 480))

time.sleep(2)

background = 0

for i in range(30):

    ret, background = cap.read()

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lr = np.array([0,120,70])
    ur = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lr, ur)

    lr = np.array([170, 120, 70])
    ur = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lr, ur)

    mask1 = mask1+mask2

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

    mask2 = cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)
    final_op = cv2.addWeighted(res1, 1, res2, 1, 0)
    out.write(final_op)
    cv2.imshow('output', final_op)
    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
