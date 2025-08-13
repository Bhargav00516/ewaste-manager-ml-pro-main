from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import os

app = Flask(__name__)

# MySQL Database Configuration with corrected connection string
app.config['SQLALCHEMY_DATABASE_URI'] = 'sql connection'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Function to check database connection
def check_database_connection(username, password):
    try:
        # Temporarily set the connection URI with the provided credentials
        test_uri = f'mysql+pymysql://{username}:{password}@localhost/ewaste_manager'
        app.config['SQLALCHEMY_DATABASE_URI'] = test_uri
        db.engine.connect()  # Test the connection
        return "Database connection successful with username '{}' and password '{}'!".format(username, password)
    except Exception as e:
        return f"Database connection failed: {str(e)}"
    finally:
        # Restore the original URI to avoid affecting the app
        app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Bhargav%40200516@localhost/ewaste_manager'

# Test route to verify database connection using the function
@app.route('/')
def home():
    result = check_database_connection('username','password')
    return result

if __name__ == '__main__':
    app.run(debug=True)
