from database import insert_muestra_async
import csv 
import asyncio
from dotenv import load_dotenv
import os

load_dotenv("./scripts/.env")

SAMPLE_DATA_PATH = os.getenv("SAMPLE_DATA_PATH")

def load_sample_data_from_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        etiquetas = []
        for row in reader:
            etiquetas.append({
                "metadata": row["metadata"],
                "herbario_id": row["herbario_id"],
                "etiquetas": int(row["etiquetas"])
            })
    return etiquetas

async def insert_sample_data_from_csv(file_path):
    etiquetas = load_sample_data_from_csv(file_path)
    for etiqueta in etiquetas:
        await insert_muestra_async(etiqueta["metadata"], etiqueta["herbario_id"], etiqueta["etiquetas"])
        
        
def main():
    file_path = SAMPLE_DATA_PATH
    asyncio.run(insert_sample_data_from_csv(file_path))
    
    
if __name__ == "__main__":
    main()    