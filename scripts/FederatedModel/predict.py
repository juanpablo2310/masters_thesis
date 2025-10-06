import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from scripts.utils.paths import get_project_configs

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
    
    # Dibujar cada predicción
    for result in results.boxes.data:
        x1, y1, x2, y2, conf, cls = result
        
        if conf > conf_threshold:
            # Convertir coordenadas a enteros
            box = np.array([x1, y1, x2, y2]).astype(int)
            
            # Dibujar caja
            cv2.rectangle(image, 
                        (box[0], box[1]), 
                        (box[2], box[3]), 
                        (0, 255, 0), 2)
            
            # Agregar etiqueta y confianza
            label = f"{results.names[int(cls)]} {conf:.2f}"
            cv2.putText(image, label, 
                       (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
    
    return image

def main():
    parser = argparse.ArgumentParser(description='Realizar predicciones con modelo YOLO entrenado')
    parser.add_argument('--weights', type=str, required=True,
                      help='Ruta al archivo de pesos del modelo')
    parser.add_argument('--input', type=str, required=True,
                      help='Ruta a la imagen o directorio de imágenes')
    parser.add_argument('--output', type=str, default='predictions',
                      help='Directorio de salida para las predicciones')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Umbral de confianza para las predicciones')
    
    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Cargar modelo
    model = load_model(args.weights)
    
    # Procesar entrada
    input_path = Path(args.input)
    if input_path.is_file():
        # Procesar una sola imagen
        image = predict_image(model, input_path, args.conf)
        output_path = output_dir / f"pred_{input_path.name}"
        cv2.imwrite(str(output_path), image)
        print(f"Predicción guardada en: {output_path}")
    
    else:
        # Procesar directorio de imágenes
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        for img_path in image_files:
            image = predict_image(model, img_path, args.conf)
            output_path = output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(output_path), image)
            print(f"Predicción guardada en: {output_path}")

if __name__ == "__main__":
    main()