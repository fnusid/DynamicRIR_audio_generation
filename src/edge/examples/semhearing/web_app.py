from flask import Flask
from flask import request
from flask import render_template

import json

app = Flask(__name__, static_url_path='', static_folder='res')

@app.route('/', methods = ['GET', 'POST'])
def user():
    if request.method == 'POST':
        data = json.loads(request.data.decode()) # a multidict containing POST data
        class_id = data['class_id'][:-4]

        print("Received selection:", class_id)
    
    return render_template('index.html')

def run_webserver():
    app.run(host='0.0.0.0', port=8080, debug=True)

run_webserver()