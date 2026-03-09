from database import insert_etiqueta_async
import csv 
import asyncio
from dotenv import load_dotenv
import os

load_dotenv("./scripts/.env")

LABEL_LIST_PATH = os.getenv("ETIQUETAS_LIST_PATH")

def load_etiquetas_from_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        etiquetas = []
        for row in reader:
            etiquetas.append({
                "nombre": row["nombre"],
                "descripcion": row["descripcion"],
                "herbario_id": int(row["herbario_id"])
            })
    return etiquetas

async def insert_etiquetas_from_csv(file_path):
    etiquetas = load_etiquetas_from_csv(file_path)
    for etiqueta in etiquetas:
        await insert_etiqueta_async(etiqueta["nombre"], etiqueta["descripcion"], etiqueta["herbario_id"])
        
        
def main():
    file_path = LABEL_LIST_PATH
    asyncio.run(insert_etiquetas_from_csv(file_path))
    
    
if __name__ == "__main__":
    main()    