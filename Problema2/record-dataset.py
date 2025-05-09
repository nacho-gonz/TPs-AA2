import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
import urllib.request
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

if 'fotos' in os.listdir('./'):
    shutil.rmtree('./fotos')

if 'augmented_images' in os.listdir('./'):
    shutil.rmtree('./augmented_images')

webcam = cv2.VideoCapture(0)
counter = 1
clases = ['papel', 'piedra', 'tijera']
ppt = ['clase_papel','clase_piedra','clase_tijera']
n = 0
print("--- Instrucciones ---")
print("Las imagenes serán tomadas por clase en el siguiente orden: piedra, papel y tijera")
print("Presiona 's' para guardar una foto.")
print("Presiona 'e' para cambiar de clase")
print("Presiona 'q' para salir.")


while True:
    try:
        check, frame = webcam.read()
        if not check:
            print("No se pudo capturar el frame.")
            break
        
        cv2.imshow("Capturando (Presiona 's' para guardar, 'e' para cambiar de clase y 'q' para salir)", frame)
        clase = clases[n]
        dirs = ppt[n]
        key = cv2.waitKey(1)

        if key == ord('s'): 
            dir = f"./fotos/{dirs}"
            filename = f"{dir}/{clase}_{counter}.jpg"
            os.makedirs(dir, exist_ok=True)
            cv2.imwrite(filename, frame)
            print(f'Capturando la imagenes de la clase {clase}')
            print(f"Foto guardada como: {filename}")
            counter += 1 
        elif key == ord('e'):
            if n >= 2:
                print('No hay más clases, se procede a terminar el programa')
                webcam.release()
                cv2.destroyAllWindows()
                break
            n += 1
            counter = 1
            print(f'Cambiando a la clase {clases[n]}')
        elif key == ord('q'):
            print("Apagando cámara...")
            webcam.release()
            cv2.destroyAllWindows()
            print("Programa terminado.")
            break

    except KeyboardInterrupt:
        print("\nApagando cámara...")
        webcam.release()
        cv2.destroyAllWindows()
        print("Programa terminado.")
        break


if 'model.task' not in os.listdir('./'):

    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url, "model.task")

base_options = python.BaseOptions(model_asset_path='./model.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=1)

detector = vision.HandLandmarker.create_from_options(options)


train_data_dir = './fotos'  # Carpeta de origen de las imágenes
output_dir = './augmented_images'  # Carpeta para guardar imágenes aumentadas
os.makedirs(output_dir, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=30,         
    width_shift_range=0.1,    
    height_shift_range=0.1,   
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True,     
    vertical_flip=True,     
    brightness_range=[0.8, 1.2],  
    fill_mode='nearest'       
)

batch_size = 32
img_width, img_height = 640, 480

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  
    save_to_dir=output_dir,    
    save_prefix='aug',         
    save_format='jpg'         
)

augmentation_factor = 3

def augment_and_save(class_folder):
    class_path = os.path.join(train_data_dir, class_folder)
    output_class_path = os.path.join(output_dir, class_folder)
    os.makedirs(output_class_path, exist_ok=True)
    
    print(f'Procesando clase: {class_folder}')
    
    for img_name in os.listdir(class_path):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(class_path, img_name)
        img = load_img(img_path)  
        x = img_to_array(img)     
        x = x.reshape((1,) + x.shape)  
        
        # Guardar imagen original
        original_filename = f'original_{img_name}'
        img.save(os.path.join(output_class_path, original_filename))
        
        # Generar y guardar imágenes aumentadas
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir=output_class_path,
                                save_prefix='aug',
                                save_format='jpeg'):
            i += 1
            if i >= augmentation_factor:
                break  

if 'fotos' in os.listdir('./') and 'augmented_images' in os.listdir('./'):

    for class_folder in os.listdir(train_data_dir):
        if os.path.isdir(os.path.join(train_data_dir, class_folder)):
            augment_and_save(class_folder)



    lista_puntos = []
    lista_labels = []


    for gesto in ppt:
        for i, foto in enumerate(os.listdir(f'./augmented_images/{gesto}')):
            mp_image = mp.Image.create_from_file(f'./augmented_images/{gesto}/{foto}')
            hand_landmarker_result = detector.detect(mp_image)
            try:
                lista_puntos.append([coord for lm in hand_landmarker_result.hand_landmarks[0] for coord in (lm.x, lm.y)])
            except:
                continue
            
            if gesto == 'clase_papel':
                lista_labels.append(1)
            elif gesto == 'clase_piedra':
                lista_labels.append(0)
            else:
                lista_labels.append(2)



    np.save('./rps_dataset.npy', lista_puntos)
    np.save('./rps_labels.npy', lista_labels)