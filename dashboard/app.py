# dashboard/app.py
import os, json, redis, psutil, time
from flask import Flask, render_template, jsonify

app = Flask(__name__)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
r = redis.from_url(REDIS_URL)

def local_metrics():
    cpu = psutil.cpu_percent(interval=None); mem = psutil.virtual_memory().percent
    load1 = os.getloadavg()[0]; disk = psutil.disk_io_counters().read_bytes/(1024*1024)
    net = psutil.net_io_counters().bytes_sent/(1024*1024)
    return {"cpu":cpu,"mem":mem,"load1":load1,"disk":disk,"net":net}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/metrics")
def metrics():
    raw = r.get("aether:metrics")
    if raw:
        return jsonify(json.loads(raw))
    return jsonify(local_metrics())

@app.route("/decision")
def decision():
    dec = r.get("aether:decision") or "UNKNOWN"
    raw_ex = r.get("aether:explain")
    expl = json.loads(raw_ex) if raw_ex else {"top":[],"text":""}
    raw_m = r.get("aether:model")
    model = json.loads(raw_m) if raw_m else {"action": dec, "probs": {"edge": 0.5, "cloud": 0.5}}
    return jsonify({"decision": dec, "explain": expl, "model": model})

@app.route("/timeline")
def timeline():
    items = r.lrange("aether:timeline", 0, 100)
    out = [json.loads(i) for i in items]
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)