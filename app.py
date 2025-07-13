from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db_cursor, init_db
import sqlite3
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
init_db()

# Регистрация
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        phone = request.form['phone']
        name = request.form['name']
        password = request.form['password']
        
        hashed_password = generate_password_hash(password)
        
        try:
            with get_db_cursor() as cursor:
                cursor.execute(
                    "INSERT INTO users (phone, password, name, role) VALUES (?, ?, ?, 'client')",
                    (phone, hashed_password, name)
                )
            flash('Регистрация успешна! Теперь вы можете войти.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Этот номер телефона уже зарегистрирован.', 'danger')
    
    return render_template('register.html')

# Вход
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        phone = request.form['phone']
        password = request.form['password']
        
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE phone = ?", (phone,))
            user = cursor.fetchone()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_phone'] = user['phone']
            session['role'] = user['role']
            return redirect(url_for('dashboard'))
        else:
            flash('Неверный номер телефона или пароль', 'danger')
    
    return render_template('login.html')

# Выход
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Главная панель (редирект)
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif session.get('role') == 'employee':
        return redirect(url_for('employee_dashboard'))
    else:
        return redirect(url_for('client_dashboard'))

# Панель администратора
@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('login'))
    
    with get_db_cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE id != ? ORDER BY created_at DESC", (session['user_id'],))
        users = cursor.fetchall()
    
    return render_template('admin_dashboard.html', users=users)

# Панель сотрудника
@app.route('/employee/dashboard')
def employee_dashboard():
    if 'user_id' not in session or session.get('role') != 'employee':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('login'))
    
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT o.*, u.name as client_name, u.phone as client_phone 
            FROM orders o
            JOIN users u ON o.user_id = u.id
            WHERE o.status != 'Отменено'
            ORDER BY 
                CASE o.status
                    WHEN 'Ожидание сдачи' THEN 1
                    WHEN 'Принят в работу' THEN 2
                    WHEN 'В процессе ремонта' THEN 3
                    WHEN 'Готов к выдаче' THEN 4
                    WHEN 'Выдан' THEN 5
                    WHEN 'Отмена' THEN 6
                END,
                o.created_at DESC
        """)
        orders = cursor.fetchall()
    
    return render_template('employee_dashboard.html', orders=orders)

# Панель клиента
@app.route('/client/dashboard')
def client_dashboard():
    if 'user_id' not in session or session.get('role') != 'client':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('login'))
    
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT * FROM orders 
            WHERE user_id = ?
            ORDER BY 
                CASE status
                    WHEN 'Ожидание сдачи' THEN 1
                    WHEN 'Принят в работу' THEN 2
                    WHEN 'В процессе ремонта' THEN 3
                    WHEN 'Готов к выдаче' THEN 4
                    WHEN 'Выдан' THEN 5
                    WHEN 'Отмена' THEN 6
                END,
                created_at DESC
        """, (session['user_id'],))
        orders = cursor.fetchall()
    
    return render_template('client_dashboard.html', orders=orders)

# Создание нового заказа
@app.route('/new_order', methods=['GET', 'POST'])
def new_order():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        device_type = request.form['device_type']
        device_model = request.form['device_model']
        problem_description = request.form['problem_description']
        
        with get_db_cursor() as cursor:
            cursor.execute("""
                INSERT INTO orders (user_id, device_type, device_model, problem_description, status)
                VALUES (?, ?, ?, ?, 'Ожидание сдачи')
            """, (session['user_id'], device_type, device_model, problem_description))
        
        flash('Заказ успешно создан!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('new_order.html')

# Просмотр статуса заказа
@app.route('/order/<int:order_id>')
def order_status(order_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    with get_db_cursor() as cursor:
        if session.get('role') in ('admin', 'employee'):
            cursor.execute("""
                SELECT o.*, u.name as client_name, u.phone as client_phone 
                FROM orders o
                JOIN users u ON o.user_id = u.id
                WHERE o.id = ?
            """, (order_id,))
        else:
            cursor.execute("""
                SELECT * FROM orders 
                WHERE id = ? AND user_id = ?
            """, (order_id, session['user_id']))
        
        order = cursor.fetchone()
    
    if not order:
        flash('Заказ не найден', 'danger')
        return redirect(url_for('dashboard'))
    
    return render_template('order_status.html', order=order)

# Обновление статуса заказа
@app.route('/update_order/<int:order_id>', methods=['POST'])
def update_order(order_id):
    if 'user_id' not in session or session.get('role') not in ('admin', 'employee'):
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('login'))
    
    new_status = request.form['status']
    price = request.form.get('price')
    
    with get_db_cursor() as cursor:
        cursor.execute("""
            UPDATE orders 
            SET status = ?, 
                price = ?,
                employee_id = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (new_status, price, session['user_id'], order_id))
    
    flash('Статус заказа обновлен', 'success')
    return redirect(url_for('employee_dashboard'))

# Создание пользователя (админом)
@app.route('/admin/users/create', methods=['GET', 'POST'])
def create_user():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        phone = request.form['phone']
        name = request.form['name']
        password = request.form['password']
        role = request.form['role']
        
        hashed_password = generate_password_hash(password)
        
        try:
            with get_db_cursor() as cursor:
                cursor.execute(
                    "INSERT INTO users (phone, password, name, role) VALUES (?, ?, ?, ?)",
                    (phone, hashed_password, name, role)
                )
            flash('Пользователь успешно создан', 'success')
            return redirect(url_for('admin_dashboard'))
        except sqlite3.IntegrityError:
            flash('Этот номер телефона уже зарегистрирован', 'danger')
    
    return render_template('create_user.html')

# Изменение роли пользователя
@app.route('/admin/users/<int:user_id>/update_role', methods=['POST'])
def update_user_role(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('login'))
    
    new_role = request.form['role']
    
    with get_db_cursor() as cursor:
        cursor.execute("""
            UPDATE users 
            SET role = ?
            WHERE id = ?
        """, (new_role, user_id))
    
    flash('Роль пользователя обновлена', 'success')
    return redirect(url_for('admin_dashboard'))

# Удаление пользователя
@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
def delete_user(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('login'))
    
    with get_db_cursor() as cursor:
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    
    flash('Пользователь удален', 'success')
    return redirect(url_for('admin_dashboard'))

# Главная страница
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)