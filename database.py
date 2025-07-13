import sqlite3
from contextlib import contextmanager
from config import Config

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_db_cursor():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        finally:
            cursor.close()

def init_db():
    with get_db_cursor() as cursor:
        # Таблица пользователей
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT CHECK(role IN ('admin', 'employee', 'client')) DEFAULT 'client',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Таблица заказов
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            employee_id INTEGER,
            device_type TEXT NOT NULL,
            device_model TEXT NOT NULL,
            problem_description TEXT NOT NULL,
            price DECIMAL(10, 2),
            status TEXT CHECK(status IN (
                'Ожидание сдачи',
                'Принят в работу',
                'Готов к выдаче',
                'Выдано',
                'Отменено'
            )) DEFAULT 'Ожидание сдачи',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (employee_id) REFERENCES users (id)
        )
        """)
        
        # Создаем первого администратора, если его нет
        cursor.execute("SELECT id FROM users WHERE role = 'admin'")
        admin_exists = cursor.fetchone()
        
        if not admin_exists:
            cursor.execute(
                "INSERT INTO users (phone, password, name, role) VALUES (?, ?, ?, 'admin')",
                ("+71112223344", "admin123", "Главный Администратор")
            )