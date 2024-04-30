from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import os
import tempfile

app = Flask(__name__)
app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "img_db"

mysql = MySQL(app)

@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'content' not in request.files or 'style' not in request.files:
        return "No Images Uploaded!"

    content = request.files['content']
    style = request.files['style']

    if content.filename == '' or style.filename == '':
        return "No Image Selected!"

    temp_dir = tempfile.gettempdir()
    file_path1 = os.path.join(temp_dir, content.filename)
    file_path2 = os.path.join(temp_dir, style.filename)
    content.save(file_path1)
    style.save(file_path2)
    print(file_path1)
    print(file_path2)
    name1 = content.filename
    name2 = style.filename
    with open(file_path1, 'rb') as f:
        data1 = f.read()
    with open(file_path2, 'rb') as f:
        data2 = f.read()

    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO images (content, style) VALUES (%s, %s)", (data1, data2))
    mysql.connection.commit()
    cur.close()
    os.remove(file_path1)
    os.remove(file_path2)

    return render_template('generate.html')


if __name__ == '__main__':
    app.run(debug=True)