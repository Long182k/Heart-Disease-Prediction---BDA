import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import bcrypt
import uuid
from datetime import datetime
import json

# Database connection string
DATABASE_URL = "postgres://avnadmin:AVNS_GfjCznKQEW3lQZFhGGl@heart-disease-prediction-tuanzz1k23-70f4.h.aivencloud.com:20046/defaultdb?sslmode=require"

# Create a connection pool
connection_pool = pool.ThreadedConnectionPool(1, 10, DATABASE_URL)

def get_connection():
    """Get a connection from the pool."""
    return connection_pool.getconn()

def release_connection(conn):
    """Release a connection back to the pool."""
    connection_pool.putconn(conn)

def init_db():
    """Initialize the database with required tables."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Create users table with UUID as primary key
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(20) DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id UUID PRIMARY KEY,
            user_id UUID REFERENCES users(id) NOT NULL,
            prediction_result JSONB NOT NULL,
            patient_data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            release_connection(conn)

def register_user(username, email, password):
    """Register a new user."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if username or email already exists
        cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
        if cursor.fetchone():
            return {"success": False, "message": "Username or email already exists"}
        
        # Generate UUID for user
        user_id = str(uuid.uuid4())
        
        # Hash the password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        cursor.execute(
            "INSERT INTO users (id, username, email, password_hash, role) VALUES (%s, %s, %s, %s, %s) RETURNING id, username, email, role, created_at",
            (user_id, username, email, password_hash, 'user')
        )
        new_user = cursor.fetchone()
        conn.commit()
        
        return {"success": True, "user": new_user}
    except Exception as e:
        print(f"Error registering user: {e}")
        if conn:
            conn.rollback()
        return {"success": False, "message": str(e)}
    finally:
        if conn:
            release_connection(conn)

def login_user(username, password):
    """Login a user with username and password."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get user by username
        cursor.execute("SELECT id, username, email, password_hash, role FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if not user:
            return {"success": False, "message": "Invalid username or password"}
        
        # Check password
        if bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            # Remove password_hash from the returned user object
            del user['password_hash']
            return {"success": True, "user": user}
        else:
            return {"success": False, "message": "Invalid username or password"}
    except Exception as e:
        print(f"Error logging in user: {e}")
        return {"success": False, "message": str(e)}
    finally:
        if conn:
            release_connection(conn)

# Add a new function to update user role
def update_user_role(user_id, new_role):
    """Update a user's role."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            "UPDATE users SET role = %s WHERE id = %s RETURNING id, username, email, role",
            (new_role, user_id)
        )
        updated_user = cursor.fetchone()
        conn.commit()
        
        if not updated_user:
            return {"success": False, "message": "User not found"}
        
        return {"success": True, "user": updated_user}
    except Exception as e:
        print(f"Error updating user role: {e}")
        if conn:
            conn.rollback()
        return {"success": False, "message": str(e)}
    finally:
        if conn:
            release_connection(conn)

def save_prediction(user_id, prediction_result, patient_data):
    """Save a prediction result for a user."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Generate a unique ID for the prediction
        prediction_id = str(uuid.uuid4())
        
        # Debug the user_id type
        print(f"Saving prediction with user_id: {user_id} (type: {type(user_id).__name__})")
        
        # Insert the prediction
        cursor.execute(
            "INSERT INTO predictions (id, user_id, prediction_result, patient_data) VALUES (%s, %s, %s, %s) RETURNING id, created_at",
            (prediction_id, user_id, psycopg2.extras.Json(prediction_result), psycopg2.extras.Json(patient_data))
        )
        prediction = cursor.fetchone()
        conn.commit()
        
        return {"success": True, "prediction": prediction}
    except Exception as e:
        print(f"Error saving prediction: {e}")
        if conn:
            conn.rollback()
        return {"success": False, "message": str(e)}
    finally:
        if conn:
            release_connection(conn)

def get_user_predictions(user_id):
    """Get all predictions for a user."""
    # No changes needed here as we'll just use the id field
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            "SELECT * FROM predictions WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        predictions = cursor.fetchall()
        
        return {"success": True, "predictions": predictions}
    except Exception as e:
        print(f"Error getting user predictions: {e}")
        return {"success": False, "message": str(e)}
    finally:
        if conn:
            release_connection(conn)

def get_prediction_by_id(prediction_id):
    """Get a specific prediction by ID."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            "SELECT * FROM predictions WHERE id = %s",
            (prediction_id,)
        )
        prediction = cursor.fetchone()
        
        if not prediction:
            return {"success": False, "message": "Prediction not found"}
        
        return {"success": True, "prediction": prediction}
    except Exception as e:
        print(f"Error getting prediction: {e}")
        return {"success": False, "message": str(e)}
    finally:
        if conn:
            release_connection(conn)

def get_all_predictions():
    """Get all predictions from the database."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Query to get all predictions with user information
        # Updated to match the actual database schema
        cursor.execute('''
            SELECT p.id, p.user_id, p.prediction_result, p.patient_data, p.created_at, u.username
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC
        ''')
        
        rows = cursor.fetchall()
        
        predictions = []
        
        for row in rows:
            # The data might already be a dict if using RealDictCursor
            # Check the type before trying to parse
            if isinstance(row[2], str):
                prediction_result = json.loads(row[2]) if row[2] else {}
            else:
                prediction_result = row[2] if row[2] else {}
                
            if isinstance(row[3], str):
                patient_data = json.loads(row[3]) if row[3] else {}
            else:
                patient_data = row[3] if row[3] else {}
            
            predictions.append({
                'id': row[0],
                'user_id': row[1],
                'prediction_data': prediction_result,  # Changed to match API expectations
                'input_data': patient_data,  # Changed to match API expectations
                'created_at': row[4],
                'username': row[5]
            })
        
        release_connection(conn)  # Use release_connection instead of close
        return {'success': True, 'predictions': predictions}
    except Exception as e:
        print(f"Database error in get_all_predictions: {e}")
        return {'success': False, 'message': f'Database error: {str(e)}'}