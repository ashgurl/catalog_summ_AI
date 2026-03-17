import os
import certifi
from pymongo import MongoClient
import json
import warnings
import instructor
import httpx
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from groq import Groq
from typing import List
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

# ==========================================
# 1. CLOUD SECRETS & CONFIGURATION
# ==========================================
# We now pull these securely from Render's Environment Variables!
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

# Setup MongoDB Connection
print("Connecting to MongoDB Atlas...")
try:
    mongo_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    mongo_db = mongo_client["SE_Products"]
    products_collection = mongo_db["Power Supplies"]
except Exception as e:
    print(f"MongoDB Connection Error: {e}")

# Setup Groq Client
custom_http_client = httpx.Client(verify=False, trust_env=False)
if GROQ_API_KEY:
    client = instructor.from_groq(
        Groq(api_key=GROQ_API_KEY, http_client=custom_http_client),
        mode=instructor.Mode.JSON,
    )

# Setup FastAPI
app = FastAPI(title="Product Catalog and Brochure Extractor and Summarizer API")

# VERY IMPORTANT FOR YOUR SENIOR: This allows her local frontend to talk to your cloud backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

# ==========================================
# 2. EPHEMERAL CLOUD CACHE (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect("se_product_cache.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS summaries (product_id TEXT PRIMARY KEY, json_data TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 3. DATA MODELS
# ==========================================
class PillarData(BaseModel):
    application: str = Field(..., description="Target Sector + Specific Use Case. Max 20 words.")
    specification: str = Field(..., description="Operational Benefit + Hard Spec. Max 20 words.")
    safety: str = Field(..., description="Protection Promise + Rating. Max 20 words.")
    compatibility: str = Field(..., description="Ease of Use + Protocol/Size. Max 20 words.")
    longevity: str = Field(..., description="ROI Statement + Cycle Count/Temp. Max 20 words.")
    sustainability: str = Field(..., description="Eco Promise + Label. Max 20 words.")

class AIProductSummary(BaseModel):
    heading: str = Field(..., description="The main product headline (Killer Feature). Max 6 words.")
    summary: str = Field(..., description="A short, 'You-focused' paragraph summarizing the product and solving a pain point. Max 30 words.")
    pillars: PillarData

class BulkProductRequest(BaseModel):
    product_ids: List[str]

# ==========================================
# 4. PYTHON METADATA EXTRACTOR
# ==========================================
def extract_business_logic(data):
    meta = {
        "ref": "Unknown", "is_upcoming": False, "is_discontinued": False,
        "badges": [], "alternative": None, "compatible": None, "refurbished_variant": None 
    }
    
    if isinstance(data, dict):
        base_data = data.get("base") or {}
        meta["ref"] = base_data.get("productCR") or base_data.get("productId") or (data.get("metaTags") or {}).get("productId") or data.get("commercialReference") or "Unknown"

        def find_refurb(obj):
            if isinstance(obj, dict):
                c_type = str(obj.get("circularType", "")).lower()
                if "refurb" in c_type or "repack" in c_type: return obj.get("productId")
                for v in obj.values():
                    res = find_refurb(v)
                    if res: return res
            elif isinstance(obj, list):
                for item in obj:
                    res = find_refurb(item)
                    if res: return res
            return None
        meta["refurbished_variant"] = find_refurb(data)

        status_data = base_data.get("productStatus") or data.get("productStatus") or {}
        com_msg = str(status_data.get("commercialMessage", "")).lower()
        if status_data.get("preCommercial") or "coming soon" in com_msg: meta["is_upcoming"] = True
            
        discontinued_terms = ["discontinued", "obsolete", "end of life", "arrêt", "fin de commercialisation"]
        if any(term in com_msg for term in discontinued_terms): meta["is_discontinued"] = True
            
        links_data = base_data.get("links") or {}
        for k, v in links_data.items():
            if isinstance(v, dict) and 'Marketplace' in str(v.get('link', '')): meta["badges"].append("NEW MARKETPLACE")

        alts_data = base_data.get("alternatives") or {}
        alts = alts_data.get("products") or []
        if alts:
            meta["alternative"] = f"{alts[0].get('commRef')} ({alts[0].get('description', 'Equivalent')})"

        asset_bar = data.get("assetBarRelatedProducts") or {}
        prod_relations = asset_bar.get("productRelations") or {}
        relations = prod_relations.get("info") or []
        
        comps = []
        for group in relations:
            group_id = group.get("groupId", "")
            if "ACCESSORIES" in group_id or "Spare part" in group_id:
                for p in group.get("products", [])[:2]: 
                    product_info = p.get("product") or {}
                    sku = product_info.get("skuId")
                    desc = product_info.get("description")
                    if sku and desc: comps.append(f"{sku} ({desc})")
        if comps: meta["compatible"] = " | ".join(comps)

    return meta

# ==========================================
# 5. TECHNICAL SPEC REFINERY
# ==========================================
def clean_payload(data):
    dense_bits = []   
    def harvest(obj, depth=0):
        if len(" | ".join(dense_bits)) > 3000: return 
        if depth > 10: return
        if isinstance(obj, dict):
            if 'charName' in obj and 'charValue' in obj:
                name = obj['charName'].get('labelText', '')
                val = obj['charValue'].get('labelText', '')
                dense_bits.append(f"{name}: {val}")
                return
            for k, v in obj.items():
                if any(x in k.lower() for x in ['url', 'img', 'id', 'token', 'date']): continue
                triggers = ['desc', 'val', 'spec', 'life', 'green', 'safe', 'volt', 'amp', 'cycle', 'temp', 'ip', 'nema', 'epd', 'eco', 'carbon', 'co2', 'rohs']
                if any(x in k.lower() for x in triggers) or (isinstance(v, str) and any(x in v.lower() for x in triggers)):
                    if isinstance(v, (str, int, float, bool)): dense_bits.append(f"{k}: {v}")
                harvest(v, depth + 1)
        elif isinstance(obj, list):
            for i in obj: harvest(i, depth + 1)
    harvest(data)
    return " | ".join(dict.fromkeys(dense_bits))[:3500] 

# ==========================================
# 6. ENDPOINTS
# ==========================================

# Health Check Endpoint (To verify your server is alive)
@app.get("/")
def health_check():
    return {"status": "Catalog Summarizer API is LIVE and ready for requests!"}

# The Main Frontend Endpoint
@app.post("/api/summaries")
def get_multiple_summaries(request: BulkProductRequest):
    response_data = {}
    conn = sqlite3.connect("se_product_cache.db")
    c = conn.cursor()
    
    for pid in request.product_ids:
        c.execute("SELECT json_data FROM summaries WHERE product_id=?", (pid,))
        db_result = c.fetchone()
        
        if db_result:
            print(f"⚡ CACHE HIT for {pid}")
            response_data[pid] = json.loads(db_result[0])
            continue
            
        print(f"☁️ CACHE MISS for {pid} - Fetching from MongoDB Atlas...")
        raw_data = products_collection.find_one({"product_id": pid}, {"_id": 0})
        
        if not raw_data:
            print(f"❌ {pid} not found in MongoDB.")
            response_data[pid] = {"error": f"Data for {pid} not found in cloud database."}
            continue

        meta = extract_business_logic(raw_data)
        dense_data = clean_payload(raw_data)

        prompt = f"""
        ROLE: Technical Sales Lead.
        TASK: Write a persuasive product summary that PROVES its value with data.
        DATA: {dense_data}

        1. THE HOOK (Heading & Summary):
           - Speak directly to the user ("You," "Your").
           - Identify their pain point and solve it.

        2. THE 6 PILLARS:
           - You MUST format each answer as: "Persuasive Benefit Statement (Evidence Data)"
           - The Evidence MUST be inside parentheses ().
           
        EXAMPLES:
           - Good: "Guaranteed operation in wet washdown zones (IP66 Rated)."
           - Good: "Install it once and forget it for years (1M Mechanical Cycles)."
           - Good: "Seamlessly integrates into your standard panels (24V AC/DC)."
        """

        try:
            ai_output = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                response_model=AIProductSummary,
                temperature=0.3, 
            )
            
            final_product_data = {
                "reference_id": meta["ref"],
                "business_alerts": {
                    "upcoming_launch": meta["is_upcoming"],
                    "discontinued": meta["is_discontinued"],
                    "badges": meta["badges"],
                    "alternative_available": meta["alternative"],
                    "refurbished_variant": meta["refurbished_variant"],
                    "compatible_products": meta["compatible"]
                },
                "content": ai_output.model_dump() 
            }
            
            c.execute("INSERT OR REPLACE INTO summaries (product_id, json_data) VALUES (?, ?)", 
                      (pid, json.dumps(final_product_data)))
            conn.commit()
            
            response_data[pid] = final_product_data
            
        except Exception as e:
            response_data[pid] = {"error": str(e)}

    conn.close()
    return response_data
