import logging
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import base64
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'image_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'


logging.basicConfig(filename='app.log', level=logging.DEBUG)

mysql = MySQL(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            if 'image' in request.files and 'image_type' in request.form:
                image = request.files['image']
                image_type = request.form['image_type']
                if image.filename != '':
                    # Read image data
                    image_data = image.read()

                    # Convert image data to base64
                    encoded_image = base64.b64encode(image_data).decode('utf-8')

                    # Save image and image type to MySQL database
                    cur = mysql.connection.cursor()
                    cur.execute("INSERT INTO images (image_type, image_data) VALUES (%s, %s)", (image_type,encoded_image))
                    mysql.connection.commit()
                    session['last_uploaded_image_id'] = cur.lastrowid
                    cur.close()

        # Fetch the last uploaded image from database
        last_uploaded_image_id = session.get('last_uploaded_image_id')
        if last_uploaded_image_id:
            cur = mysql.connection.cursor()
            cur.execute("SELECT id, image_data, image_type FROM images WHERE id = %s", (last_uploaded_image_id,))
            image = cur.fetchone()
            cur.close()
        else:
            image = None

        return render_template('index.html', image=image)

    except Exception as e:
        logging.exception("An error occurred: %s", str(e))
        return "An error occurred while processing your request. Please try again later."

if __name__ == '__main__':
    app.run(debug=True)
