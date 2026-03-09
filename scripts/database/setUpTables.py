import asyncio
from HerbariumTable import HERBARIUM_LIST_PATH, insert_herbarios_from_csv
from LabelTable import LABEL_LIST_PATH, insert_etiquetas_from_csv
from SampleTable import SAMPLE_DATA_PATH, insert_sample_data_from_csv

async def main():
    await insert_herbarios_from_csv(HERBARIUM_LIST_PATH)
    await insert_etiquetas_from_csv(LABEL_LIST_PATH)
    await insert_sample_data_from_csv(SAMPLE_DATA_PATH)
    
if __name__ == "__main__":  
    asyncio.run(main())
    
    
    