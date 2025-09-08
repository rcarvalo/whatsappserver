#!/usr/bin/env python3
"""
🤖 Serveur Python WhatsApp RAG 
Serveur complet pour webhook WhatsApp + RAG + Supabase
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Imports pour l'extraction de montres
try:
    from watch_info_extractor import WatchInfoExtractor
    watch_extractor = WatchInfoExtractor()
    logger.info("🕰️ Extracteur de montres initialisé avec succès")
except ImportError as e:
    logger.warning(f"⚠️ Extracteur de montres non disponible: {e}")
    watch_extractor = None

# Variables d'environnement
VERIFY_TOKEN = os.getenv('WHATSAPP_VERIFY_TOKEN', 'hellotesttoken')
ACCESS_TOKEN = os.getenv('WHATSAPP_ACCESS_TOKEN')
PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialisation de l'application FastAPI
app = FastAPI(
    title="🤖 WhatsApp RAG Server",
    description="Serveur Python pour webhook WhatsApp avec RAG et Supabase",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour les processeurs
whatsapp_api = None
embedding_processor = None
rag_searcher = None

# Statistiques
stats = {
    "messages_received": 0,
    "messages_processed": 0,
    "errors": 0,
    "startup_time": datetime.now().isoformat(),
    # 🕰️ Statistiques spécifiques montres
    "watch_messages_detected": 0,
    "sales_detected": 0,
    "wanted_detected": 0,
    "questions_detected": 0,
    "total_value_detected": 0.0  # Valeur totale des montres détectées
}

def log_env_status():
    """Log le statut des variables d'environnement"""
    logger.info("🔍 Configuration des variables d'environnement:")
    env_vars = {
        "VERIFY_TOKEN": VERIFY_TOKEN[:5] + "..." if VERIFY_TOKEN else "❌ Non défini",
        "ACCESS_TOKEN": "✅ Défini" if ACCESS_TOKEN else "❌ Non défini", 
        "PHONE_NUMBER_ID": "✅ Défini" if PHONE_NUMBER_ID else "❌ Non défini",
        "SUPABASE_URL": "✅ Défini" if SUPABASE_URL else "❌ Non défini",
        "SUPABASE_KEY": "✅ Défini" if SUPABASE_KEY else "❌ Non défini",
        "OPENAI_API_KEY": "✅ Défini" if OPENAI_API_KEY else "❌ Non défini"
    }
    
    for var, status in env_vars.items():
        logger.info(f"  {var}: {status}")

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global whatsapp_api, embedding_processor, rag_searcher
    
    logger.info("🚀 Démarrage du serveur WhatsApp RAG...")
    log_env_status()
    
    # Essayer d'initialiser les modules avancés
    try:
        from src.whatsapp_realtime_api import WhatsAppRealtimeAPI
        
        if ACCESS_TOKEN and PHONE_NUMBER_ID and SUPABASE_URL and OPENAI_API_KEY:
            whatsapp_api = WhatsAppRealtimeAPI(
                access_token=ACCESS_TOKEN,
                phone_number_id=PHONE_NUMBER_ID,
                verify_token=VERIFY_TOKEN,
                webhook_secret=os.getenv('WHATSAPP_WEBHOOK_SECRET'),
                supabase_url=SUPABASE_URL,
                supabase_key=SUPABASE_KEY,
                openai_api_key=OPENAI_API_KEY
            )
            logger.info("✅ WhatsAppRealtimeAPI initialisé avec succès")
            
            # Récupérer les processeurs
            if hasattr(whatsapp_api, 'embedding_processor'):
                embedding_processor = whatsapp_api.embedding_processor
                logger.info("✅ EmbeddingProcessor disponible")
                
            if hasattr(whatsapp_api, 'rag_searcher'):
                rag_searcher = whatsapp_api.rag_searcher
                logger.info("✅ RAGSearcher disponible")
        else:
            logger.warning("⚠️ Variables d'environnement manquantes pour le mode avancé")
            
    except Exception as e:
        logger.warning(f"⚠️ Mode avancé indisponible: {e}")
        
    logger.info("✅ Serveur démarré en mode webhook basique")

def extract_whatsapp_messages(webhook_data: Dict[str, Any]) -> list:
    """Extrait les messages WhatsApp du payload webhook"""
    messages = []
    
    if webhook_data.get('object') != 'whatsapp_business_account':
        return messages
        
    for entry in webhook_data.get('entry', []):
        for change in entry.get('changes', []):
            if change.get('field') == 'messages':
                value = change.get('value', {})
                for message in value.get('messages', []):
                    if message.get('type') == 'text' and message.get('text'):
                        extracted = {
                            'id': message.get('id'),
                            'from': message.get('from'),
                            'timestamp': message.get('timestamp'),
                            'text': message.get('text', {}).get('body'),
                            'phone_number_id': value.get('metadata', {}).get('phone_number_id'),
                            'received_at': datetime.now().isoformat()
                        }
                        messages.append(extracted)
                        
    return messages

async def process_message_with_rag(message: Dict[str, Any]) -> Dict[str, Any]:
    """Traite un message avec le système RAG et extraction de montres"""
    try:
        content = message.get('text', '')
        phone_number = message.get('from', '')
        
        # 🕰️ EXTRACTION D'INFORMATIONS MONTRES
        watch_info = None
        if watch_extractor:
            try:
                watch_info = watch_extractor.extract_watch_info(content)
                logger.info(f"🔍 Extraction montres: {watch_info.brand} {watch_info.model} - {watch_info.price}€ ({watch_info.message_type})")
            except Exception as e:
                logger.error(f"❌ Erreur extraction montres: {e}")
        
        if embedding_processor:
            # Créer l'embedding
            embedding = embedding_processor.generate_embedding(content)
            
            if embedding:
                # Créer l'objet MessageEmbedding avec infos montres
                from src.embedding_processor import MessageEmbedding
                
                # Log des informations de montres extraites
                if watch_info and watch_info.confidence_score > 0.2:
                    logger.info(f"💎 Message enrichi avec données montres (confiance: {watch_info.confidence_score:.2f})")
                    logger.info(f"   🏷️ Marque: {watch_info.brand}, Modèle: {watch_info.model}")
                    logger.info(f"   💰 Prix: {watch_info.price} {watch_info.currency}, Condition: {watch_info.condition}")
                    logger.info(f"   📊 Type: {watch_info.message_type}")
                
                message_emb = MessageEmbedding(
                    id=None,
                    phone_number=phone_number,
                    message_content=content,
                    timestamp=datetime.now().isoformat(),
                    sender=phone_number,
                    is_outgoing=False,
                    embedding=embedding
                )
                
                # Stocker en base
                result = embedding_processor.store_message_embedding(message_emb)
                
                # 📊 Mettre à jour les statistiques montres
                if watch_info and watch_info.confidence_score > 0.2:
                    stats["watch_messages_detected"] += 1
                    if watch_info.message_type == "sale":
                        stats["sales_detected"] += 1
                    elif watch_info.message_type == "wanted":
                        stats["wanted_detected"] += 1
                    elif watch_info.message_type == "question":
                        stats["questions_detected"] += 1
                    
                    if watch_info.price:
                        stats["total_value_detected"] += watch_info.price
                
                logger.info(f"✅ Message traité avec RAG enrichi: {result}")
                return {
                    "success": True, 
                    "embedding_id": result,
                    "watch_info": watch_info.__dict__ if watch_info else None
                }
            else:
                logger.error("❌ Impossible de générer l'embedding")
                return {"success": False, "reason": "Embedding failed"}
        else:
            # Mode basique sans RAG mais avec extraction montres
            basic_result = {
                "success": True, 
                "mode": "basic",
                "watch_info": watch_info.__dict__ if watch_info else None
            }
            
            if watch_info and watch_info.confidence_score > 0.3:
                logger.info(f"📝 Message stocké en mode basique avec infos montres: {watch_info.brand} {watch_info.model}")
            else:
                logger.info("📝 Message stocké en mode basique (RAG indisponible)")
                
            return basic_result
            
    except Exception as e:
        logger.error(f"❌ Erreur traitement RAG: {e}")
        return {"success": False, "error": str(e)}

## 📍 ENDPOINTS DE L'API

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "🤖 WhatsApp RAG Server",
        "status": "running", 
        "version": "2.0.0",
        "server": "Python FastAPI",
        "features": [
            "✅ Webhook WhatsApp",
            "✅ Traitement RAG" if rag_searcher else "⚠️ RAG indisponible",
            "✅ Stockage Supabase" if embedding_processor else "⚠️ Supabase indisponible",
            "✅ Embeddings OpenAI" if OPENAI_API_KEY else "⚠️ OpenAI indisponible"
        ],
        "endpoints": {
            "webhook": "/webhook (GET/POST)",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de santé détaillée"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": datetime.now().isoformat(),
        "components": {
            "webhook": True,
            "whatsapp_api": whatsapp_api is not None,
            "embedding_processor": embedding_processor is not None,
            "rag_searcher": rag_searcher is not None,
            "supabase": bool(SUPABASE_URL and SUPABASE_KEY),
            "openai": bool(OPENAI_API_KEY)
        },
        "stats": stats
    }

@app.get("/stats")
async def get_stats():
    """Statistiques du serveur"""
    return {
        "server_info": {
            "type": "python",
            "framework": "FastAPI",
            "version": "2.0.0"
        },
        "configuration": {
            "verify_token": VERIFY_TOKEN[:5] + "..." if VERIFY_TOKEN else "Non défini",
            "access_token_configured": bool(ACCESS_TOKEN),
            "phone_number_configured": bool(PHONE_NUMBER_ID),
            "supabase_configured": bool(SUPABASE_URL),
            "openai_configured": bool(OPENAI_API_KEY),
            "watch_extractor": bool(watch_extractor)
        },
        "runtime_stats": stats,
        "watch_stats": {
            "messages_analyzed": stats["watch_messages_detected"],
            "sales_found": stats["sales_detected"],
            "wanted_found": stats["wanted_detected"],
            "questions_found": stats["questions_detected"],
            "total_value_eur": f"{stats['total_value_detected']:.2f}€"
        },
        "timestamp": datetime.now().isoformat()
    }

# ===========================
# 🚨 ENDPOINTS TEMPORAIRES POUR DEBUG
# ===========================

@app.get("/")
async def verify_root_webhook(request: Request):
    """
    🔐 Vérification du webhook WhatsApp sur la racine (GET) - TEMPORAIRE
    Certaines configurations Meta pointent sur la racine
    """
    mode = request.query_params.get('hub.mode')
    token = request.query_params.get('hub.verify_token') 
    challenge = request.query_params.get('hub.challenge')
    
    logger.info(f"📞 Vérification webhook RACINE - Mode: {mode}, Token: {token[:10] if token else 'None'}..., Challenge: {challenge}")
    
    if mode == 'subscribe' and token == VERIFY_TOKEN:
        logger.info("✅ Webhook vérifié avec succès (racine)")
        stats["messages_received"] += 1
        return PlainTextResponse(challenge, status_code=200)
    else:
        logger.warning(f"❌ Échec vérification webhook racine - Token attendu: {VERIFY_TOKEN[:5]}...")
        stats["errors"] += 1
        return PlainTextResponse("Forbidden", status_code=403)

@app.post("/")
async def webhook_root_event(request: Request):
    """
    📨 Reception des messages WhatsApp sur la racine (POST) - TEMPORAIRE
    Redirection vers le traitement normal
    """
    logger.info("📨 ⚠️  Message reçu sur la RACINE - Redirection vers traitement webhook")
    return await webhook_event(request)

# ===========================
# 🔗 ENDPOINTS WEBHOOK OFFICIELS
# ===========================

@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    🔐 Vérification du webhook WhatsApp (GET)
    Facebook utilise cette méthode pour vérifier l'URL du webhook
    """
    mode = request.query_params.get('hub.mode')
    token = request.query_params.get('hub.verify_token') 
    challenge = request.query_params.get('hub.challenge')
    
    logger.info(f"📞 Vérification webhook - Mode: {mode}, Token: {token[:10] if token else 'None'}..., Challenge: {challenge}")
    
    if mode == 'subscribe' and token == VERIFY_TOKEN:
        logger.info("✅ Webhook vérifié avec succès")
        stats["messages_received"] += 1
        return PlainTextResponse(challenge, status_code=200)
    else:
        logger.warning(f"❌ Échec vérification webhook - Token attendu: {VERIFY_TOKEN[:5]}...")
        stats["errors"] += 1
        return PlainTextResponse("Forbidden", status_code=403)

@app.post("/webhook")
async def webhook_event(request: Request):
    """
    📨 Reception des messages WhatsApp (POST)  
    Facebook envoie les messages WhatsApp ici
    """
    try:
        body = await request.json()
        logger.info("📨 Webhook POST reçu de Facebook")
        
        # Extraire les messages
        messages = extract_whatsapp_messages(body)
        stats["messages_received"] += len(messages)
        
        if messages:
            logger.info(f"🔍 {len(messages)} message(s) WhatsApp détecté(s)")
            
            # Traiter chaque message
            for message in messages:
                logger.info(f"📱 Message de {message['from']}: {message['text'][:50]}...")
                
                # Traitement avec RAG si disponible
                result = await process_message_with_rag(message)
                
                if result["success"]:
                    stats["messages_processed"] += 1
                    logger.info(f"✅ Message traité avec succès")
                else:
                    stats["errors"] += 1
                    logger.error(f"❌ Erreur traitement: {result.get('error')}")
        else:
            logger.info("📭 Aucun nouveau message texte détecté")
            logger.debug(f"Raw payload: {body}")
        
        return JSONResponse({"status": "received"}, status_code=200)
        
    except Exception as e:
        logger.error(f"❌ Erreur traitement webhook: {e}")
        stats["errors"] += 1
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/search")
async def search_messages(request: Request):
    """
    🔍 Endpoint de recherche RAG
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        limit = body.get("limit", 5)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query parameter required")
            
        if rag_searcher:
            results = await rag_searcher.search(query, limit=limit)
            return {"success": True, "query": query, "results": results}
        else:
            return {"success": False, "error": "RAG searcher not available"}
            
    except Exception as e:
        logger.error(f"❌ Erreur recherche: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

## 🚀 POINT D'ENTRÉE POUR RENDER

if __name__ == "__main__":
    import uvicorn
    
    # Configuration pour Render (port automatique)
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    logger.info("=" * 60)
    logger.info("🤖 WHATSAPP RAG SERVER - PYTHON")
    logger.info("=" * 60)
    logger.info(f"🚀 Démarrage sur {host}:{port}")
    logger.info(f"📚 Documentation: http://{host}:{port}/docs")
    logger.info(f"🔍 Health check: http://{host}:{port}/health")
    logger.info("=" * 60)
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
