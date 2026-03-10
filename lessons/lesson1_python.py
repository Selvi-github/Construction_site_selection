# Lesson 1: Python basics (code-first)

print("Lesson 1: Python basics")

# 1) Variables
name = "Asha"
age = 16
print("Name:", name)
print("Age:", age)

# 2) Lists
scores = [72, 85, 90]
print("Scores:", scores)

# 3) For loop + sum
total = 0
for s in scores:
    total += s
avg = total / len(scores)
print("Average score:", avg)

# 4) Function

def greet(person):
    return f"Hello, {person}!"

print(greet(name))

# 5) Simple input + output
# Uncomment to use interactive input
# user_name = input("Enter your name: ")
# print(greet(user_name))


# Lesson 2: Tiny Flask API (code-first)
# Run: python lessons/lesson1_python.py
# Then open: http://127.0.0.1:5000/api/health

from flask import Flask, request, jsonify

api = Flask(__name__)


@api.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


@api.route("/api/echo", methods=["POST"])
def api_echo():
    data = request.get_json() or {}
    name = data.get("name", "friend")
    return jsonify({"message": f"Hello, {name}!"})


if __name__ == "__main__":
    api.run(debug=True)


# Lesson 3: Flask input validation + in-memory storage
# Add simple data, validate inputs, return errors

items = []


@api.route("/api/add", methods=["POST"])
def api_add():
    data = request.get_json() or {}
    title = (data.get("title") or "").strip()
    qty = data.get("qty")
    if not title:
        return jsonify({"error": "title is required"}), 400
    if not isinstance(qty, int) or qty <= 0:
        return jsonify({"error": "qty must be a positive integer"}), 400
    item = {"id": len(items) + 1, "title": title, "qty": qty}
    items.append(item)
    return jsonify({"success": True, "item": item})


@api.route("/api/list", methods=["GET"])
def api_list():
    return jsonify({"count": len(items), "items": items})


# Lesson 4: Project-style analyze endpoint (simplified)
# This mimics the real /api/analyze flow with basic checks and a mock score.
#
# Command line examples:
# Start server:
#   python lessons/lesson1_python.py
# Health check:
#   curl http://127.0.0.1:5000/api/health
# Analyze example:
#   curl -X POST http://127.0.0.1:5000/api/analyze-simple \
#     -H "Content-Type: application/json" \
#     -d "{\"lat\":12.97,\"lon\":77.59,\"building_type\":\"House\",\"floors\":2}"


@api.route("/api/analyze-simple", methods=["POST"])
def analyze_simple():
    data = request.get_json() or {}
    try:
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon must be numbers"}), 400

    building_type = (data.get("building_type") or "House").strip()
    floors = data.get("floors", 2)
    try:
        floors = int(floors)
    except (TypeError, ValueError):
        return jsonify({"error": "floors must be an integer"}), 400

    if not (6.5 <= lat <= 37.5 and 67.0 <= lon <= 97.5):
        return jsonify({"error": "Location must be within India"}), 400

    # Simple mock score for learning
    base = 70
    if building_type.lower() in {"factory", "warehouse"}:
        base -= 10
    if floors >= 5:
        base -= 8
    score = max(20, min(95, base))
    risk = "Low Risk" if score >= 70 else "Medium Risk" if score >= 45 else "High Risk"

    return jsonify({
        "feasibility_score": score,
        "risk_level": risk,
        "building_type": building_type,
        "floors": floors
    })
