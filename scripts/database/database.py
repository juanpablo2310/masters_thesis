from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import text
import asyncio
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_session_maker():
    engine = create_async_engine(DATABASE_URL)
    return async_sessionmaker(engine, expire_on_commit=False)

async def insert_herbario_async(nombre: str, afiliacion: str, localizacion: str):
    async_session = get_session_maker()
    async with async_session() as connection:
        await connection.execute(
            text("INSERT INTO herbarios (nombre, afiliacion, localizacion) VALUES (:nombre, :afiliacion, :localizacion)"),
            {"nombre": nombre, "afiliacion": afiliacion, "localizacion": localizacion}
        )
        await connection.commit()
        logging.info(f"Inserted herbario: {nombre}")
        
async def insert_etiqueta_async(nombre: str, descripcion: str, herbario_id: int):
    async_session = get_session_maker()
    async with async_session() as connection:
        await connection.execute(
            text("INSERT INTO etiquetas (nombre, descripcion, herbario_id) VALUES (:nombre, :descripcion, :herbario_id)"),
            {"nombre": nombre, "descripcion": descripcion, "herbario_id": herbario_id}
        )
        await connection.commit()
        logging.info(f"Inserted etiqueta: {nombre}")
        
async def insert_muestra_async(herbario_id: int, etiquetas: list, descripcion: str):
    async_session = get_session_maker()
    async with async_session() as connection:
        await connection.execute(
            text("INSERT INTO muestras (herbario_id, etiquetas, descripcion) VALUES (:herbario_id, :etiquetas, :descripcion)"),
            {"herbario_id": herbario_id, "etiquetas": etiquetas, "descripcion": descripcion}
        )
        await connection.commit()
        logging.info(f"Inserted muestra for herbario_id: {herbario_id} with etiquetas: {etiquetas}")
        
async def delete_async(table, id):
    async_session = get_session_maker()
    async with async_session() as connection:
        await connection.execute(
            text(f"DELETE FROM {table} WHERE id = :id"),
            {"id": id}
        )
        await connection.commit()
