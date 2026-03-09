import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np

import easyocr

reader = easyocr.Reader(['en', 'es'], gpu=torch.cuda.is_available())

def load_model(weights_path):
    """
    Carga el modelo YOLO entrenado
    """
    model = YOLO(weights_path)
    return model

def predict_image(model, image_path, conf_threshold=0.25):
    """
    Realiza predicciones en una imagen y dibuja las cajas delimitadoras
    """
    # Realizar predicción
    results = model(image_path)[0]
    
    # Cargar imagen para dibujar
    image = cv2.imread(str(image_path))
  #  import pdb; pdb.set_trace()
    labels = []
    legends = []
    metadata = {}
    # Dibujar cada predicción
    for result in results.boxes.data:
        x1, y1, x2, y2, conf, cls = result
        
        #if conf > conf_threshold:
        # Convertir coordenadas a enteros
        box = np.array([x1, y1, x2, y2]).astype(int)
        # crop image to get metadata
        crop = image[box[1]:box[3], box[0]:box[2]]
        cv2.imshow("image",crop)
        cv2.waitKey()
        
        # Use easyocr to read text from the cropped image
        ocr_result = reader.readtext(crop)
        for (bbox, text, score) in ocr_result:
            print(f"Text: {text}, Score: {score}")
        # Dibujar caja
        cv2.rectangle(image, 
                    (box[0], box[1]), 
                    (box[2], box[3]), 
                    (0, 255, 0), 2)
        # Agregar etiqueta y confianza
        label = f"{results.names[int(cls)]} {conf:.2f}"
        labels.append(label)
        cv2.putText(image, label, 
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
    
    return image, labels

def main():
    # parser = argparse.ArgumentParser(description='Realizar predicciones con modelo YOLO entrenado')
    # parser.add_argument('--weights', type=str, required=True,
    #                   help='Ruta al archivo de pesos del modelo') #/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/models/pytorch/YOLO/yolo_trained_unal/last.pt
    # parser.add_argument('--input', type=str, required=True,
    #                   help='Ruta a la imagen o directorio de imágenes') #/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/data_UN/imagenes/COL000343749.png
    # parser.add_argument('--output', type=str, default='predictions',
    #                   help='Directorio de salida para las predicciones') #/Users/juanpablovargasacosta/herbario/scripts
    # parser.add_argument('--conf', type=float, default=0.25,
    #                   help='Umbral de confianza para las predicciones')
    
    # args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    output_dir = Path("/Users/juanpablovargasacosta/herbario/scripts")
    #output_dir.mkdir(exist_ok=True, parents=True)
    
    # Cargar modelo
    model = load_model("/Users/juanpablovargasacosta/herbario/runs/detect/train94/weights/best.pt")#"/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/federated_learning/runs/train26/weights/best.pt")
    
    # Procesar entrada
    input_path = Path("/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario/data_MELU/train/images/MELUD102629a_sp69013617863657648142_medium.png")
   # import pdb; pdb.set_trace()
    if input_path.is_file():
        # Procesar una sola imagen
        image,labels = predict_image(model, input_path, 0.25)
        output_path = output_dir / f"pred_{input_path.name}"
        cv2.imwrite(output_path, image)
        print(f"Predicción guardada en: {output_path}")
        print(f"Labels: {labels}")
    
    else:
        # Procesar directorio de imágenes
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        for img_path in image_files:
            image,labels = predict_image(model, img_path, 0.25)
            output_path = output_dir / f"pred_{img_path.name}"
            cv2.imwrite(output_path, image)
            print(f"Predicción guardada en: {output_path}")
            print(f"Labels: {labels}")

if __name__ == "__main__":
    main()