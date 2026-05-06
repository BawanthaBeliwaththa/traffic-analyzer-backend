import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, origins=['*'])

# ============================================
# SRI LANKA CITIES DATA
# ============================================
SRI_LANKA_CITIES = [
    {"id": "W001", "name": "Colombo", "lat": 6.9271, "lng": 79.8608, "district": "Colombo", "province": "Western Province", "type": "capital"},
    {"id": "W002", "name": "Dehiwala", "lat": 6.8400, "lng": 79.8600, "district": "Colombo", "province": "Western Province", "type": "suburb"},
    {"id": "W003", "name": "Negombo", "lat": 7.2167, "lng": 79.8833, "district": "Gampaha", "province": "Western Province", "type": "city"},
    {"id": "W004", "name": "Kalutara", "lat": 6.7800, "lng": 79.9500, "district": "Kalutara", "province": "Western Province", "type": "city"},
    {"id": "W005", "name": "Moratuwa", "lat": 6.8200, "lng": 79.8800, "district": "Colombo", "province": "Western Province", "type": "city"},
    {"id": "W006", "name": "Gampaha", "lat": 7.0900, "lng": 79.9900, "district": "Gampaha", "province": "Western Province", "type": "city"},
    {"id": "C001", "name": "Kandy", "lat": 7.2930, "lng": 80.6333, "district": "Kandy", "province": "Central Province", "type": "major_city"},
    {"id": "C002", "name": "Matale", "lat": 7.4700, "lng": 80.6200, "district": "Matale", "province": "Central Province", "type": "city"},
    {"id": "C003", "name": "Nuwara Eliya", "lat": 6.9700, "lng": 80.7800, "district": "Nuwara Eliya", "province": "Central Province", "type": "tourist"},
    {"id": "C004", "name": "Dambulla", "lat": 7.8600, "lng": 80.6500, "district": "Matale", "province": "Central Province", "type": "tourist"},
    {"id": "S001", "name": "Galle", "lat": 6.0328, "lng": 80.2167, "district": "Galle", "province": "Southern Province", "type": "major_city"},
    {"id": "S002", "name": "Matara", "lat": 5.9500, "lng": 80.5500, "district": "Matara", "province": "Southern Province", "type": "city"},
    {"id": "S003", "name": "Hikkaduwa", "lat": 6.0800, "lng": 80.1800, "district": "Galle", "province": "Southern Province", "type": "beach"},
    {"id": "S004", "name": "Hambantota", "lat": 6.1200, "lng": 81.1200, "district": "Hambantota", "province": "Southern Province", "type": "city"},
    {"id": "N001", "name": "Jaffna", "lat": 9.6615, "lng": 80.0255, "district": "Jaffna", "province": "Northern Province", "type": "major_city"},
    {"id": "N002", "name": "Vavuniya", "lat": 8.7500, "lng": 80.5000, "district": "Vavuniya", "province": "Northern Province", "type": "city"},
    {"id": "N003", "name": "Kilinochchi", "lat": 9.4000, "lng": 80.4000, "district": "Kilinochchi", "province": "Northern Province", "type": "town"},
    {"id": "E001", "name": "Batticaloa", "lat": 7.7167, "lng": 81.8333, "district": "Batticaloa", "province": "Eastern Province", "type": "major_city"},
    {"id": "E002", "name": "Trincomalee", "lat": 8.5714, "lng": 81.2339, "district": "Trincomalee", "province": "Eastern Province", "type": "major_city"},
    {"id": "E003", "name": "Arugam Bay", "lat": 7.4500, "lng": 81.8500, "district": "Ampara", "province": "Eastern Province", "type": "beach"},
    {"id": "NC001", "name": "Anuradhapura", "lat": 8.3114, "lng": 80.4037, "district": "Anuradhapura", "province": "North Central Province", "type": "major_city"},
    {"id": "NC002", "name": "Polonnaruwa", "lat": 7.9333, "lng": 81.0000, "district": "Polonnaruwa", "province": "North Central Province", "type": "major_city"},
    {"id": "NW001", "name": "Kurunegala", "lat": 7.4869, "lng": 80.3643, "district": "Kurunegala", "province": "North Western Province", "type": "major_city"},
    {"id": "NW002", "name": "Puttalam", "lat": 8.0300, "lng": 79.8300, "district": "Puttalam", "province": "North Western Province", "type": "city"},
    {"id": "U001", "name": "Badulla", "lat": 6.9900, "lng": 81.0550, "district": "Badulla", "province": "Uva Province", "type": "major_city"},
    {"id": "U002", "name": "Bandarawela", "lat": 6.8300, "lng": 80.9900, "district": "Badulla", "province": "Uva Province", "type": "town"},
    {"id": "SG001", "name": "Ratnapura", "lat": 6.6833, "lng": 80.4000, "district": "Ratnapura", "province": "Sabaragamuwa Province", "type": "major_city"},
    {"id": "SG002", "name": "Kegalle", "lat": 7.2500, "lng": 80.3000, "district": "Kegalle", "province": "Sabaragamuwa Province", "type": "city"}
]

# ============================================
# LOAD DATA AND TRAIN MODEL
# ============================================
print("🚀 Starting Traffic Prediction API...")
print("📂 Loading data...")

# Try to load the CSV from multiple possible locations
possible_paths = [
    'traffic_data_1million.csv',
    'data/traffic_data_1million.csv',
    '/app/traffic_data_1million.csv',
    '/app/data/traffic_data_1million.csv'
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✅ Loaded data from: {path}")
        break

if df is None:
    print("❌ Could not find traffic_data_1million.csv")
    print("📁 Current directory contents:")
    for root, dirs, files in os.walk('.'):
        for file in files:
            print(f"   {os.path.join(root, file)}")
    raise FileNotFoundError("traffic_data_1million.csv not found")

print(f"📊 Dataset shape: {df.shape}")

# Preprocess data
print("🔄 Preprocessing data...")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_peak'] = df['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

# Create target variable
congestion_mapping = {'flowing': 0.3, 'moderate': 0.6, 'congested': 0.85}
df['congestion'] = df['congestion_level'].map(congestion_mapping).fillna(0.5)

# Train model
print("🤖 Training model...")
feature_cols = ['hour', 'day_of_week', 'is_weekend', 'month', 'is_peak']
X = df[feature_cols].values
y = df['congestion'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled, y)

# Save model artifacts
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/traffic_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Model trained and saved!")

# Calculate model accuracy
from sklearn.metrics import mean_absolute_error, r2_score
y_pred = model.predict(X_scaled)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"📈 Model Accuracy: {(1 - mae) * 100:.2f}%")
print(f"📈 R² Score: {r2:.4f}")

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_city(city, current_time=None):
    """Predict traffic congestion for a specific city"""
    if current_time is None:
        current_time = datetime.now()
    
    hour = current_time.hour
    day = current_time.weekday()
    is_weekend = 1 if day >= 5 else 0
    month = current_time.month
    is_peak = 1 if hour in [7, 8, 9, 16, 17, 18, 19] else 0
    
    # Make prediction
    features = np.array([[hour, day, is_weekend, month, is_peak]])
    features_scaled = scaler.transform(features)
    congestion = float(model.predict(features_scaled)[0])
    
    # Adjust based on city type
    if city["type"] == "capital":
        congestion += 0.1
    elif city["type"] == "tourist" and is_weekend:
        congestion += 0.15
    
    congestion = max(0.05, min(0.95, congestion))
    
    # Determine congestion level
    if congestion > 0.75:
        level = "congested"
    elif congestion > 0.5:
        level = "moderate"
    else:
        level = "flowing"
    
    # Calculate estimated speed
    speed = round(60 - (congestion * 45), 1)
    
    # Calculate delay in minutes
    delay = round(congestion * 30, 0)
    
    return {
        "id": city["id"],
        "name": city["name"],
        "district": city["district"],
        "province": city["province"],
        "type": city["type"],
        "lat": city["lat"],
        "lng": city["lng"],
        "congestion_percent": round(congestion * 100, 1),
        "level": level,
        "speed_kmh": speed,
        "delay_minutes": delay,
        "prediction_time": current_time.isoformat(),
        "is_peak_hour": bool(is_peak),
        "is_weekend": bool(is_weekend)
    }

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation"""
    return jsonify({
        "name": "Sri Lanka Traffic Prediction API",
        "version": "2.0.0",
        "description": "Real-time traffic congestion predictions for Sri Lankan cities",
        "endpoints": {
            "health": "/health",
            "all_cities": "/api/all-cities",
            "specific_city": "/api/city/<city_name>",
            "summary": "/api/summary",
            "provinces": "/api/provinces",
            "search": "/api/search?q=<query>"
        },
        "example_cities": ["Colombo", "Kandy", "Galle", "Jaffna"],
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "traffic-prediction-api",
        "model_accuracy": f"{(1 - mae) * 100:.2f}%",
        "cities_covered": len(SRI_LANKA_CITIES),
        "uptime": "running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/all-cities', methods=['GET'])
def get_all_cities():
    """Get traffic predictions for all cities"""
    current_time = datetime.now()
    
    # Optional query parameters
    province_filter = request.args.get('province')
    type_filter = request.args.get('type')
    
    predictions = [predict_city(city, current_time) for city in SRI_LANKA_CITIES]
    
    # Apply filters if provided
    if province_filter:
        predictions = [p for p in predictions if p['province'].lower() == province_filter.lower()]
    if type_filter:
        predictions = [p for p in predictions if p['type'].lower() == type_filter.lower()]
    
    # Sort by congestion (highest first)
    predictions.sort(key=lambda x: x["congestion_percent"], reverse=True)
    
    return jsonify({
        "success": True,
        "total_cities": len(predictions),
        "timestamp": current_time.isoformat(),
        "data": predictions
    })

@app.route('/api/city/<name>', methods=['GET'])
def get_city(name):
    """Get traffic prediction for a specific city"""
    current_time = datetime.now()
    
    # Search for city (case-insensitive)
    city = next((c for c in SRI_LANKA_CITIES if c["name"].lower() == name.lower()), None)
    
    if not city:
        # Try partial match
        matching_cities = [c for c in SRI_LANKA_CITIES if name.lower() in c["name"].lower()]
        if matching_cities:
            return jsonify({
                "success": False,
                "error": f"City '{name}' not found. Did you mean one of these?",
                "suggestions": [c["name"] for c in matching_cities]
            }), 404
        else:
            return jsonify({
                "success": False,
                "error": f"City '{name}' not found",
                "available_cities": [c["name"] for c in SRI_LANKA_CITIES]
            }), 404
    
    prediction = predict_city(city, current_time)
    return jsonify({
        "success": True,
        "data": prediction
    })

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get traffic summary statistics"""
    current_time = datetime.now()
    predictions = [predict_city(city, current_time) for city in SRI_LANKA_CITIES]
    
    congested = [p for p in predictions if p["level"] == "congested"]
    moderate = [p for p in predictions if p["level"] == "moderate"]
    flowing = [p for p in predictions if p["level"] == "flowing"]
    
    speeds = [p["speed_kmh"] for p in predictions]
    
    return jsonify({
        "success": True,
        "timestamp": current_time.isoformat(),
        "summary": {
            "total_cities": len(predictions),
            "congested_count": len(congested),
            "moderate_count": len(moderate),
            "flowing_count": len(flowing),
            "avg_speed_kmh": round(sum(speeds) / len(speeds), 1),
            "max_speed_kmh": max(speeds),
            "min_speed_kmh": min(speeds),
            "most_congested": congested[:3] if congested else [],
            "least_congested": flowing[-3:] if flowing else []
        }
    })

@app.route('/api/provinces', methods=['GET'])
def get_provinces():
    """Get traffic summary by province"""
    current_time = datetime.now()
    predictions = [predict_city(city, current_time) for city in SRI_LANKA_CITIES]
    
    provinces = {}
    for p in predictions:
        province = p["province"]
        if province not in provinces:
            provinces[province] = {
                "name": province,
                "cities": [],
                "avg_congestion": 0,
                "avg_speed": 0
            }
        provinces[province]["cities"].append({
            "name": p["name"],
            "congestion": p["congestion_percent"],
            "level": p["level"],
            "speed": p["speed_kmh"]
        })
    
    # Calculate averages
    for province in provinces.values():
        province["avg_congestion"] = round(
            sum(c["congestion"] for c in province["cities"]) / len(province["cities"]), 1
        )
        province["avg_speed"] = round(
            sum(c["speed"] for c in province["cities"]) / len(province["cities"]), 1
        )
        province["city_count"] = len(province["cities"])
        province["status"] = "congested" if province["avg_congestion"] > 75 else "moderate" if province["avg_congestion"] > 50 else "flowing"
    
    return jsonify({
        "success": True,
        "timestamp": current_time.isoformat(),
        "data": list(provinces.values())
    })

@app.route('/api/search', methods=['GET'])
def search():
    """Search for cities"""
    query = request.args.get('q', '').lower()
    
    if not query:
        return jsonify({"success": False, "error": "Please provide a search query 'q'"}), 400
    
    matching_cities = [
        city for city in SRI_LANKA_CITIES 
        if query in city["name"].lower() or query in city["district"].lower() or query in city["province"].lower()
    ]
    
    if not matching_cities:
        return jsonify({
            "success": True,
            "query": query,
            "total": 0,
            "message": "No cities found matching your query",
            "data": []
        })
    
    current_time = datetime.now()
    predictions = [predict_city(city, current_time) for city in matching_cities]
    
    return jsonify({
        "success": True,
        "query": query,
        "total": len(predictions),
        "data": predictions
    })

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "message": "Check / for available endpoints"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "message": str(error)
    }), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting server on port {port}...")
    print(f"📡 API will be available at http://0.0.0.0:{port}")
    print(f"📖 Check http://0.0.0.0:{port}/ for documentation\n")
    app.run(host='0.0.0.0', port=port, debug=False)