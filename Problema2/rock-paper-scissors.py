import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import tensorflow as tf

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

    # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN


    return annotated_image

# Cargar modelo entrenado
modelo = tf.keras.models.load_model("rps_model.h5")

# Clases
clases = ["Piedra", "Papel", "Tijera"]

# Iniciar MediaPipe Task
base_options = python.BaseOptions(model_asset_path='./model.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

counter = 1

# Iniciar webcam
cap = cv2.VideoCapture(0)

print("--- Instrucciones ---")
print("Presiona 's' para guardar una foto")
print("Presiona 'q' para salir.")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar frame.")
        break

    frame = cv2.flip(frame, 1)  
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Crear imagen de MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    prediccion = None

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0] 

        entrada = np.array([coord for lm in landmarks for coord in (lm.x, lm.y)])
        entrada = np.expand_dims(entrada, axis=0) 

        salida = modelo.predict(entrada, verbose=0)
        clase_idx = np.argmax(salida)
        prediccion = clases[clase_idx]

        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result)

    # Mostrar resultado
    if prediccion:
        cv2.putText(annotated_image, f"Eleccion: {prediccion}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    try:
        cv2.imshow("Piedra, Papel o Tijera", annotated_image)
    except:
        cv2.imshow("Piedra, Papel o Tijera", frame)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and prediccion: 
        dir = f"./predicciones/{prediccion}"
        filename = f"{dir}/{prediccion}_{counter}.jpg"
        os.makedirs(dir, exist_ok=True)
        cv2.imwrite(filename, annotated_image)
        print(f'Capturando la imagenes de la clase {prediccion}')
        print(f"Foto guardada como: {filename}")
        counter += 1

        

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
