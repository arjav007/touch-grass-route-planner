from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
import json

app = FastAPI(title="Touch Grass AI Route Planner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteRequest(BaseModel):
    location: str
    distance_km: int = 20
    activity: str = "cycling"
    time_of_day: str = "Morning"

@app.get("/")
def read_root():
    return {"status": "Touch Grass Backend Online", "model": "gemma:2b"}

@app.post("/api/generate-route")
def generate_route(req: RouteRequest):
    # 1. Python handles the realistic duration
    if req.activity == "walking" or req.activity == "hiking":
        realistic_mins = req.distance_km * 12  
    elif req.activity == "running":
        realistic_mins = req.distance_km * 6   
    else: 
        realistic_mins = req.distance_km * 3   
        
    # 2. Python handles the math to prevent skipped kilometers
    # Caps at 12 waypoints so the UI stays clean, evenly distributing the km markers
    num_waypoints = max(5, min(12, int(req.distance_km / 3.5)))
    step_size = req.distance_km / (num_waypoints - 1)
    km_markers = [round(i * step_size, 1) for i in range(num_waypoints - 1)] + [float(req.distance_km)]
    
    # 3. Build a strict fill-in-the-blank array template for the AI
    waypoints_template = ",\n        ".join([f'"[Action/Direction] near [Physically Adjacent Landmark] ({km}km)"' for km in km_markers])
        
    prompt = f"""
    You are an expert endurance sports and navigation assistant mapping a continuous {req.distance_km} km {req.activity} route in {req.location}.

    CRITICAL GEOGRAPHY RULES:
    1. DO NOT TELEPORT. You must pick landmarks that are physically connected and logically sequential. Do not jump between distant neighborhoods.
    2. If {req.distance_km} is very long (like 42km), make it an out-and-back route or a multi-loop route staying within the same general region.
    3. Use the EXACT JSON structure below. I have pre-calculated the kilometer markers. You must replace the bracketed text with real street-level geography.

    Return ONLY a valid JSON object matching this exact template:
    {{
      "title": "Short catchy route name",
      "summary": "1-2 sentence description",
      "estimated_duration_mins": {realistic_mins},
      "waypoints": [
        {waypoints_template}
      ],
      "gear_checklist": ["Item 1", "Item 2"],
      "safety_tips": ["Tip 1"]
    }}
    """

    try:
        response = ollama.chat(
            model='gemma:2b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        content = response['message']['content'].strip()
        
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        return json.loads(content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model execution error: {str(e)}")