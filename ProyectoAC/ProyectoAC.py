
from turtle import color
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
up = False
down = False
count = 0
mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_pose = mp.solutions.pose  # type: ignore
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)
# establecer dimensiones de la imagen a tomar
cap.set(cv2.CAP_PROP_FRAME_WIDTH,2560) # ancho
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1440) # alto
# Tomamos una foto con la cámara web
ret, frame = cap.read()
# Guardamos la imagen en un archivo y la ruta seleccionada
cv2.imwrite('C:/Users/Admin/Downloads/AC/ejemplo.jpg',frame)
with mp_pose.Pose(
    static_image_mode=True) as pose:
    image = cv2.imread("ejemplo.jpg")
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    print("Pose landmarks:", results.pose_landmarks)
    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(image, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
        cv2.imshow("Image", image)
    cv2.waitKey(0)

with mp_pose.Pose(
    static_image_mode=False) as pose:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks is not None:
            
               x1 = int(results.pose_landmarks.landmark[11].x * width)
               y1 = int(results.pose_landmarks.landmark[11].y * height)
               x2 = int(results.pose_landmarks.landmark[23].x * width)
               y2 = int(results.pose_landmarks.landmark[23].y * height)
               x3 = int(results.pose_landmarks.landmark[25].x * width)
               y3 = int(results.pose_landmarks.landmark[25].y * height)
               x4 = int(results.pose_landmarks.landmark[27].x * width)
               y4 = int(results.pose_landmarks.landmark[27].y * height)

               p1 = np.array([x1, y1])
               p2 = np.array([x2, y2])
               p3 = np.array([x3, y3])
               p4 = np.array([x4, y4])
               l1 = np.linalg.norm(p2 - p3)  # type: ignore
               l2 = np.linalg.norm(p1 - p3)  # type: ignore
               l3 = np.linalg.norm(p1 - p2)  # type: ignore
               
               # Calcular el ángulo
               angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
               # Visualización
               aux_image = np.zeros(frame.shape, np.uint8)
               cv2.line(aux_image, (x1, y1), (x2, y2), (255, 255, 0), 20)
               cv2.line(aux_image, (x2, y2), (x3, y3), (255, 255, 0), 20)
               cv2.line(aux_image, (x3, y3), (x4, y4), (255, 255, 0), 20)
               cv2.line(aux_image, (x1, y1), (x3, y3), (255, 255, 0), 5)
               contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
              #color del triángulo
               cv2.fillPoly(aux_image, pts=[contours], color=(128, 0, 250))
               output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)
               cv2.circle(output, (x1, y1), 6, (128, 0, 250), 4)
               cv2.circle(output, (x2, y2), 6, (0, 255, 255), 4)
               cv2.circle(output, (x3, y3), 6, (128, 0, 250), 4)
               cv2.circle(output, (x4, y4), 6, (128, 0, 250), 4)
               cv2.putText(output, str(int(angle)), (x2 + 30, y2), 1, 1.5, (128, 0, 250), 2)
               cv2.imshow("output", output)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
        
cap.release()
cv2.destroyAllWindows()