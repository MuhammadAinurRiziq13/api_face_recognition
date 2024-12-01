from sqlalchemy import Column, Integer, String
from app.database import Base

class Pegawai(Base):
    __tablename__ = 'pegawai'

    id = Column(Integer, primary_key=True, index=True)
    id_pegawai = Column(String, unique=True, index=True)
    foto_1 = Column(String, nullable=True)
    foto_2 = Column(String, nullable=True)
    foto_3 = Column(String, nullable=True)
    foto_4 = Column(String, nullable=True)
    foto_5 = Column(String, nullable=True)
