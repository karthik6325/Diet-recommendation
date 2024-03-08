# from flask import Flask
# from api import api_blueprint
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app, origins=['http://localhost:3000'])
# app.register_blueprint(api_blueprint)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
#     app.run(debug=True)



# app.py
from flask import Flask
from api import api_blueprint

app = Flask(__name__)
app.register_blueprint(api_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
