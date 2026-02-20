import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


async def get_engine_async():
    return create_engine(DATABASE_URL, future=True)

async def connect_db_async():
    engine = get_engine_async()
    return engine.connect()


async def insert_herbario_async(nombre: str, afiliacion: str, localizacion: str):
    #engine = get_engine_async()
    async with connect_db_async() as connection:
        await connection.execute(
            "INSERT INTO herbarios (nombre, afiliacion, localizacion) VALUES (%s, %s, %s)",
            (nombre, afiliacion, localizacion)
        )
        
async def insert_etiqueta_async(nombre: str, descripcion: str, herbario_id: int):
    #engine = get_engine_async()
    async with connect_db_async() as connection:
        await connection.execute(
            "INSERT INTO etiquetas (nombre, descripcion, herbario_id) VALUES (%s, %s, %s)",
            (nombre, descripcion, herbario_id)
        )
        
async def insert_muestra_async(herbario_id: int, etiquetas: list, descripcion: str):
    #engine = get_engine_async()
    async with connect_db_async() as connection:
        await connection.execute(
            "INSERT INTO muestras (herbario_id, etiquetas, descripcion) VALUES (%s, %s, %s)",
            (herbario_id, etiquetas, descripcion)
        )
        
async def delete_async(table, id):
    #engine = get_engine_async()
    async with connect_db_async() as connection:
        await connection.execute(
            "DELETE FROM %s WHERE id = %s",
            (table, id)
        )

