from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# URL koneksi ke PostgreSQL
DATABASE_URL = "postgresql://postgres:123@localhost/face_recognition_db"   

# Membuat engine dan session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Fungsi untuk membuat tabel di database jika belum ada
def create_tables():
    Base.metadata.create_all(bind=engine)

# Fungsi untuk mendapatkan sesi database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Panggil create_tables saat aplikasi dimulai
create_tables()
