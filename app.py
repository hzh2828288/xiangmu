from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 模拟用户数据库
users = {}

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"code": 1, "message": "用户名和密码不能为空"}), 400

    if username in users:
        return jsonify({"code": 1, "message": "用户名已存在"}), 400

    users[username] = password
    return jsonify({"code": 0, "message": "注册成功"}), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"code": 1, "message": "用户名和密码不能为空"}), 400

    if username not in users or users[username] != password:
        return jsonify({"code": 1, "message": "账号或密码错误"}), 400

    return jsonify({"code": 0, "message": "登录成功"}), 200

if __name__ == '__main__':
    app.run(debug=True)