from flask import Flask
from flask import request
from flask import send_file
from flask import redirect
from flask import url_for

import os

from run_deep import run

app = Flask(__name__)

app.config["PATH"] = "./static/video/"
app.config["DEEP_PATH"] = "/SSD/hackathon/data/data_validation/"

flag = [1]

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['files']
        if f:
            tmpName = "validation_video.mp4"
            fName = f.filename
            f.save(app.config["DEEP_PATH"] + tmpName)
        print("File Upload Success : " + app.config["DEEP_PATH"] + "validation_video.mp4")

        flag[0] = run()
        print("flag value = ", flag[0])
    return "ok"


@app.route('/download', methods=['GET', 'POST'])
def download_File():
    print("Flag Value = ", flag[0])
    if request.method == 'POST' and flag[0] == 0:
        filenames = request.form['filenames'].split("#")
        local_files = os.listdir(app.config["PATH"])
        for f in filenames:
            if f in local_files:
                file_name = app.config["PATH"] + f
                return send_file(file_name, mimetype="video/mp4", attachment_filename=f, as_attachment=True)
        flag[0] = 1
        print("Flag Value = ", flag[0])
    return "no"

@app.route('/mp4')
def play_mp4():
    print(url_for('static', filename='video/test.mp4'))
    return redirect(url_for('static', filename='video/test.mp4'))

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(debug=True, host="192.168.0.94", port=18080)
