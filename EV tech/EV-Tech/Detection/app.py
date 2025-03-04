from flask import Flask, render_template, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Load the HTML file

@app.route('/start-python', methods=['GET'])
def start_python():
    try:
        subprocess.Popen(["python", "detection.py"])  # Runs detection script in the background
        return jsonify({"status": "success", "message": "Python script started."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
