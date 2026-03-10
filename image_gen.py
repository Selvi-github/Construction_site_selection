import requests
from PIL import Image
import io
import os
import logging

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
# Using SDXL base as suggested for better output
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

def build_scenario_prompt(result):
    building_type = result.get('building_type', 'building')
    base = f"Ultra realistic 3D architectural visualization of a {building_type} building in India."
    
    risk_details = []
    
    # We map the raw inputs or predicted risks
    # Assuming the predictor returns 'risk_level' and various domain scores/risks in raw_data 
    raw = result.get('raw_data', {})
    env = raw.get('env', {})
    climate = raw.get('climate', {})
    soil = raw.get('soil', {})
    animal = raw.get('animal', {})
    
    flood_risk = str(env.get('flood_risk') or climate.get('flood_risk') or 'Low').upper()
    seismic_risk = str(env.get('earthquake_risk') or 'Low').upper()
    bearing = Number(soil.get('bearing_capacity_kNm2', 150)) if isinstance(soil.get('bearing_capacity_kNm2'), (int, float, str)) and str(soil.get('bearing_capacity_kNm2')).replace('.','',1).isdigit() else 150
    soil_strength = "WEAK" if float(bearing) < 100 else "STRONG"
    animal_conflict = str(animal.get('protected_area_risk') or 'Low').upper()
    
    # Store these in result so visualization.html can use them easily
    result['flood_risk'] = flood_risk
    result['seismic_risk'] = "HIGH" if seismic_risk in ["HIGH", "SEVERE", "ZONE V", "ZONE IV"] else "LOW"
    result['soil_strength'] = soil_strength
    result['animal_conflict'] = animal_conflict
    
    if flood_risk == "HIGH":
        risk_details.append("Heavy monsoon rain, water accumulation around foundation, partial flooding")
        
    if result['seismic_risk'] == "HIGH":
        risk_details.append("Visible structural cracks, minor wall damage from earthquake tremors")
        
    if animal_conflict == "HIGH":
        risk_details.append("Wildlife movement near site boundary, animal intrusion signs")
        
    if soil_strength == "WEAK":
        risk_details.append("Foundation settlement, uneven ground, soil erosion cracks")
        
    if len(risk_details) > 0:
        base += " Scene shows " + ", ".join(risk_details) + "."
    else:
        base += " Safe environment, stable soil, clear weather, structurally sound building."
        
    base += " Cinematic lighting, engineering simulation style, highly detailed, realistic rendering, professional architecture visualization."
    
    return base

def generate_image(prompt, output_dir="static"):
    if not HF_TOKEN:
        logging.warning("HUGGINGFACE_TOKEN is not set. Image generation will fail or return default.")
        
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }
    
    payload = {
        "inputs": prompt
    }
    
    output_path = os.path.join(output_dir, "scenario.png")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            logging.error(f"HF API Error: {response.text}")
            # return a fallback or raise
            raise Exception(f"HuggingFace API Error: {response.text}")
            
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))
        image.save(output_path)
        return "scenario.png"
    except Exception as e:
        logging.error(f"Image generation failed: {e}")
        # Could create a blank dummy image or just return none
        return None
