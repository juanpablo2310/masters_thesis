from database import insert_herbario_async
import csv 
import asyncio

def load_herbarios_from_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        herbarios = []
        for row in reader:
            herbarios.append({
                "nombre": row["nombre"],
                "afiliacion": row["afiliacion"],
                "localizacion": row["localizacion"]
            })
    return herbarios

async def insert_herbarios_from_csv(file_path):
    herbarios = load_herbarios_from_csv(file_path)
    for herbario in herbarios:
        await insert_herbario_async(herbario["nombre"], herbario["afiliacion"], herbario["localizacion"])
        
        
def main():
    file_path = "scripts/database/herbarios.csv"
    asyncio.run(insert_herbarios_from_csv(file_path))
    
    
if __name__ == "__main__":
    main()    