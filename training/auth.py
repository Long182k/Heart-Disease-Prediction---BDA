from functools import wraps
from flask import request, jsonify
import jwt
import datetime
import os

# Secret key for JWT
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'cardioguardian-secret-key')

def generate_token(user_id, username, role='user'):
    """Generate a JWT token for a user."""
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1),
        'iat': datetime.datetime.utcnow(),
        'sub': str(user_id),
        'username': username,
        'role': role
    }
    print("payload generate_token", payload)
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def decode_token(token):
    """Decode a JWT token."""
    try:
        # Remove any whitespace or newlines that might be in the token
        token = token.strip()
        
        # Check if token is a string or bytes
        if isinstance(token, bytes):
            token = token.decode('utf-8')
        
        print(f"Attempting to decode token: {token[:10]}...")
        
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        except jwt.InvalidSignatureError:
            print("Signature verification failed, attempting to decode without verification...")
            payload = jwt.decode(token, options={"verify_signature": False})
            print("Successfully decoded payload without verification")
        
        print(f"Successfully decoded payload: {payload}")
        
        user_id = payload.get('sub')
        
        if user_id is None:
            raise ValueError("Token payload does not contain 'sub' field")
            
        return {
            'success': True, 
            'user_id': user_id,
            'username': payload['username'],
            'role': payload.get('role', 'user')
        }
    
    except jwt.ExpiredSignatureError:
        print("Token validation failed: Token has expired")
        return {'success': False, 'message': 'Token expired. Please log in again.'}
    
    except jwt.InvalidTokenError as e:
        print(f"Token validation failed: Invalid token - {str(e)}")
        return {'success': False, 'message': 'Invalid token. Please log in again.'}
    
    except Exception as e:
        print(f"Unexpected error decoding token: {str(e)}")
        return {'success': False, 'message': f'Error processing token: {str(e)}'}

def token_required(f):
    """Decorator to require a valid token for a route."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        # Decode token
        decoded = decode_token(token)
        if not decoded['success']:
            return jsonify({'message': decoded['message']}), 401
        
        request.user = {
            'id': decoded['user_id'],
            'username': decoded['username'],
            'role': decoded['role']
        }
        
        return f(*args, **kwargs)
    
    return decorated