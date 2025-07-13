from werkzeug.security import generate_password_hash
from database import get_db_cursor

def create_first_admin():
    with get_db_cursor() as cursor:
        cursor.execute("SELECT id FROM users WHERE role = 'admin'")
        admin_exists = cursor.fetchone()
        
        if not admin_exists:
            cursor.execute(
                "INSERT INTO users (phone, password, name, role) VALUES (?, ?, ?, 'admin')",
                ("+7 (111) 222-33-44", generate_password_hash("admin123"), "Главный Администратор")
            )
            print("Создан первый администратор: телефон +7 (111) 222-33-44, пароль admin123")
if __name__ == "__main__":
    create_first_admin()