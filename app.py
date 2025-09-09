#!/usr/bin/env python3
"""
🤖 Serveur Python WhatsApp RAG 
Serveur complet pour webhook WhatsApp + RAG + Supabase
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
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


# Variables d'environnement
VERIFY_TOKEN = os.getenv('WHATSAPP_VERIFY_TOKEN', 'hellotesttoken')
ACCESS_TOKEN = os.getenv('WHATSAPP_ACCESS_TOKEN')
PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


try:
    # Priorité à l'extracteur LLM
    if OPENAI_API_KEY:
        from src.llm_watch_extractor import LLMWatchExtractor
        watch_extractor = LLMWatchExtractor(openai_api_key=OPENAI_API_KEY)
        logger.info("🤖 Extracteur LLM de montres initialisé avec succès")
    else:
        # Fallback vers l'extracteur regex si pas d'API key
        from watch_info_extractor import WatchInfoExtractor
        watch_extractor = WatchInfoExtractor()
        logger.info("🕰️ Extracteur regex de montres initialisé (fallback)")
except ImportError as e:
    logger.warning(f"⚠️ Extracteur de montres non disponible: {e}")
    watch_extractor = None
    
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

def _normalize_watch_info(watch_info) -> Dict:
    """Normalise les informations de montre depuis différents extracteurs"""
    if watch_info is None:
        return {}
    
    # Convertir l'objet watch_info en dictionnaire uniforme
    if hasattr(watch_info, '__dict__'):
        normalized = watch_info.__dict__.copy()
    else:
        normalized = {}
    
    # Normaliser les champs clés pour la compatibilité
    field_mapping = {
        # LLM extracteur vers format uniforme
        'collection': 'model_collection',
        'condition_details': 'condition_description',
        'dial_color': 'dial_color',
        'material': 'case_material',
        'has_box': 'box_mentioned',
        'has_papers': 'papers_mentioned',
        'has_warranty': 'warranty_mentioned',
        'shipping_available': 'shipping_mentioned',
        'seller_motivation': 'urgency_context',
        'llm_reasoning': 'extraction_reasoning',
        
        # Champs communs (pas de mapping nécessaire)
        'brand': 'brand',
        'model': 'model',
        'reference': 'reference',
        'price': 'price',
        'currency': 'currency',
        'condition': 'condition',
        'year': 'year',
        'size': 'size',
        'movement_type': 'movement_type',
        'message_type': 'message_type',
        'confidence_score': 'confidence_score'
    }
    
    # Appliquer le mapping
    result = {}
    for source_key, target_key in field_mapping.items():
        if source_key in normalized:
            result[target_key] = normalized[source_key]
    
    # Ajouter les champs non mappés
    for key, value in normalized.items():
        if key not in field_mapping and key not in result:
            result[key] = value
    
    return result

def create_message_embedding_from_watch_info(
    phone_number: str, 
    content: str, 
    embedding: list, 
    watch_info=None,
    whatsapp_metadata=None
) -> 'MessageEmbedding':
    """Crée un MessageEmbedding enrichi avec les données d'extraction de montres"""
    from src.embedding_processor import MessageEmbedding
    
    # 🎯 ENRICHISSEMENT AVEC MÉTADONNÉES WHATSAPP ET WATCH_INFO NORMALISÉ
    whatsapp_meta = whatsapp_metadata or {}
    semantic_meta = whatsapp_meta.get('semantic_metadata', {})
    
    # Normaliser les informations de montre pour la compatibilité
    normalized_watch = _normalize_watch_info(watch_info)
    
    # Extraction du nom de groupe depuis les métadonnées WhatsApp
    group_name = None
    if whatsapp_meta.get('is_group_message'):
        group_name = whatsapp_meta.get('sender_profile_name')
        if not group_name and whatsapp_meta.get('group_context_indicators'):
            group_name = f"Groupe détecté ({', '.join(whatsapp_meta.get('group_context_indicators', [])[:2])})"
    
    # Déterminer le type de message enrichi (priorité à l'extracteur LLM)
    message_type = 'general'
    if normalized_watch.get('message_type'):
        message_type = normalized_watch.get('message_type')
    elif semantic_meta.get('intent_signals', {}).get('is_selling'):
        message_type = 'sale'
    elif semantic_meta.get('intent_signals', {}).get('is_seeking'):
        message_type = 'wanted'
    elif semantic_meta.get('intent_signals', {}).get('is_question'):
        message_type = 'question'
    
    # Score d'intention enrichi (priorité à l'extracteur LLM)
    intent_score = 0.0
    if normalized_watch.get('confidence_score'):
        intent_score = normalized_watch.get('confidence_score')
    else:
        # Calculer un score basé sur les signaux sémantiques
        intent_signals = semantic_meta.get('intent_signals', {})
        signal_count = sum(1 for signal in intent_signals.values() if signal)
        intent_score = min(signal_count * 0.2, 1.0)  # Score maximum de 1.0
    
    return MessageEmbedding(
        id=None,
        phone_number=phone_number,
        message_content=content,
        message_timestamp=datetime.now().isoformat(),
        sender=whatsapp_meta.get('sender_formatted_name') or whatsapp_meta.get('sender_profile_name') or phone_number,
        is_outgoing=False,
        embedding=embedding,
        
        # 📱 MÉTADONNÉES WHATSAPP ENRICHIES
        group_name=group_name,
        
        # 🕰️ DONNÉES MONTRES EXTRAITES
        message_type=message_type,
        intent_score=intent_score,
        
        # 🕰️ INFORMATIONS MONTRES (sources: LLM extracteur + regex extracteur)
        watch_brand=normalized_watch.get('brand'),
        watch_model=normalized_watch.get('model'),
        watch_reference=normalized_watch.get('reference'),
        
        # 💰 INFORMATIONS PRIX
        price_mentioned=normalized_watch.get('price'),
        currency=normalized_watch.get('currency', 'EUR'),
        price_type=normalized_watch.get('price_type'),
        
        # 📊 ÉTAT ET CARACTÉRISTIQUES
        condition_mentioned=normalized_watch.get('condition'),
        year_mentioned=normalized_watch.get('year'),
        size_mentioned=normalized_watch.get('size'),
        movement_type=normalized_watch.get('movement_type'),
        
        # 🏪 INFORMATIONS VENTE
        seller_type=normalized_watch.get('seller_type'),
        location_mentioned=normalized_watch.get('location'),
        shipping_info=normalized_watch.get('shipping_info') or normalized_watch.get('shipping_details'),
        authenticity_mentioned=normalized_watch.get('authenticity_mentioned', False),
        
        # 🎯 MÉTADONNÉES ENRICHIES POUR RECHERCHE SÉMANTIQUE
        extracted_keywords=_extract_enhanced_keywords(content, normalized_watch, whatsapp_meta),
        sentiment_score=_calculate_sentiment_score(content, semantic_meta) or normalized_watch.get('sentiment_score'),
        urgency_level=_calculate_urgency_level(content, semantic_meta) or normalized_watch.get('urgency_level', 0),
        
        # 📊 JSON POUR INFOS COMPLEXES ENRICHIES
        detailed_extraction=_create_detailed_extraction(normalized_watch, whatsapp_meta),
        search_metadata=_create_enhanced_search_metadata(normalized_watch, whatsapp_meta, semantic_meta)
    )

def extract_whatsapp_messages(webhook_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extrait les messages WhatsApp avec métadonnées enrichies du payload webhook"""
    messages = []
    
    if webhook_data.get('object') != 'whatsapp_business_account':
        return messages
        
    for entry in webhook_data.get('entry', []):
        for change in entry.get('changes', []):
            if change.get('field') == 'messages':
                value = change.get('value', {})
                metadata = value.get('metadata', {})
                
                # 🔍 EXTRACTION DES CONTACTS ET PROFILS
                contacts = {contact.get('wa_id'): contact for contact in value.get('contacts', [])}
                
                for message in value.get('messages', []):
                    if message.get('type') == 'text' and message.get('text'):
                        sender_wa_id = message.get('from', '')
                        contact_info = contacts.get(sender_wa_id, {})
                        
                        # 🏷️ INFORMATIONS ENRICHIES DU CONTEXTE
                        context_info = message.get('context', {}) or {}
                        
                        extracted = {
                            # 📱 DONNÉES BASIQUES
                            'id': message.get('id'),
                            'from': sender_wa_id,
                            'timestamp': message.get('timestamp'),
                            'text': message.get('text', {}).get('body'),
                            'phone_number_id': metadata.get('phone_number_id'),
                            'received_at': datetime.now().isoformat(),
                            
                            # 👤 INFORMATIONS EXPÉDITEUR ENRICHIES
                            'sender_profile_name': contact_info.get('profile', {}).get('name'),
                            'sender_formatted_name': contact_info.get('profile', {}).get('formatted_name'),
                            'sender_wa_id': sender_wa_id,
                            
                            # 📱 MÉTADONNÉES WHATSAPP BUSINESS
                            'business_phone_number_id': metadata.get('phone_number_id'),
                            'display_phone_number': metadata.get('display_phone_number'),
                            
                            # 🏢 INFORMATIONS DE CONTEXTE (groupes, replies, etc.)
                            'context_from': context_info.get('from'),  # ID du message auquel on répond
                            'context_id': context_info.get('id'),      # ID du message de contexte
                            'context_quoted': context_info.get('quoted'),  # Message cité
                            
                            # 🔍 DÉTECTION DE GROUPE (via patterns heuristiques)
                            'is_group_message': _detect_group_message(sender_wa_id, contact_info, context_info),
                            'group_context_indicators': _extract_group_indicators(message.get('text', {}).get('body', '')),
                            
                            # 📊 MÉTADONNÉES TECHNIQUES
                            'message_type': message.get('type'),
                            'webhook_entry_id': entry.get('id'),
                            'webhook_timestamp': entry.get('time'),
                            
                            # 🎯 MÉTADONNÉES POUR RECHERCHE SÉMANTIQUE
                            'semantic_metadata': _create_semantic_metadata(
                                sender_wa_id, 
                                contact_info, 
                                message.get('text', {}).get('body', ''),
                                context_info
                            )
                        }
                        messages.append(extracted)
                        
    return messages

def _detect_group_message(sender_wa_id: str, contact_info: Dict, context_info: Dict) -> bool:
    """Détecte si un message provient probablement d'un groupe"""
    # Heuristiques pour détecter les messages de groupe
    
    # 1. Format du numéro (les groupes ont souvent des formats spécifiques)
    if '@g.us' in sender_wa_id:
        return True
    
    # 2. Présence de contexte de réponse (plus fréquent dans les groupes)
    if context_info and (context_info.get('from') or context_info.get('id')):
        return True
        
    # 3. Nom de profil avec indicateurs de groupe
    profile_name = contact_info.get('profile', {}).get('name', '').lower()
    group_indicators = ['groupe', 'group', 'team', 'équipe', 'vente', 'montres', 'rolex', 'watch']
    
    return any(indicator in profile_name for indicator in group_indicators)

def _extract_group_indicators(message_text: str) -> List[str]:
    """Extrait les indicateurs de contexte de groupe depuis le texte"""
    indicators = []
    text_lower = message_text.lower()
    
    # Patterns typiques des groupes de vente de montres
    group_patterns = [
        'dans le groupe', 'groupe', '@', 'cc:', 'tous:', 'hello tous',
        'salut les amis', 'bonjour à tous', 'qui est intéressé',
        'quelqu\'un pour', 'membres du groupe'
    ]
    
    for pattern in group_patterns:
        if pattern in text_lower:
            indicators.append(pattern)
            
    return indicators

def _create_semantic_metadata(sender_wa_id: str, contact_info: Dict, message_text: str, context_info: Dict) -> Dict:
    """Crée des métadonnées enrichies pour améliorer la recherche sémantique"""
    
    metadata = {
        # 👤 PROFIL EXPÉDITEUR
        'sender': {
            'wa_id': sender_wa_id,
            'profile_name': contact_info.get('profile', {}).get('name'),
            'formatted_name': contact_info.get('profile', {}).get('formatted_name'),
            'is_verified': contact_info.get('profile', {}).get('verified', False)
        },
        
        # 📱 CONTEXTE CONVERSATIONNEL
        'conversation': {
            'has_context': bool(context_info),
            'is_reply': bool(context_info.get('id')),
            'reply_to': context_info.get('from') if context_info else None,
            'thread_context': bool(context_info.get('quoted'))
        },
        
        # 📝 ANALYSE TEXTUELLE
        'text_analysis': {
            'length': len(message_text),
            'word_count': len(message_text.split()),
            'has_urls': 'http' in message_text or 'www.' in message_text,
            'has_phone': any(char.isdigit() for char in message_text) and len([c for c in message_text if c.isdigit()]) >= 8,
            'has_email': '@' in message_text and '.' in message_text,
            'has_price': any(symbol in message_text for symbol in ['€', '$', '£', 'EUR', 'USD']),
            'language_hints': _detect_language_hints(message_text)
        },
        
        # ⏰ TEMPORALITÉ
        'timing': {
            'processed_at': datetime.now().isoformat(),
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'is_business_hours': 9 <= datetime.now().hour <= 18
        },
        
        # 🎯 INTENTIONS DÉTECTÉES
        'intent_signals': {
            'is_selling': any(word in message_text.lower() for word in ['vends', 'vente', 'à vendre', 'prix', 'sell', 'for sale']),
            'is_seeking': any(word in message_text.lower() for word in ['cherche', 'recherche', 'want', 'looking for', 'iso']),
            'is_question': '?' in message_text or message_text.lower().startswith(('comment', 'pourquoi', 'quand', 'qui', 'how', 'why', 'when')),
            'is_greeting': any(word in message_text.lower() for word in ['bonjour', 'salut', 'hello', 'hi', 'bonsoir']),
            'has_urgency': any(word in message_text.lower() for word in ['urgent', 'rapidement', 'vite', 'asap', 'quickly'])
        }
    }
    
    return metadata

def _detect_language_hints(text: str) -> List[str]:
    """Détecte les indices de langue dans le texte"""
    hints = []
    text_lower = text.lower()
    
    # Mots français caractéristiques
    french_words = ['bonjour', 'merci', 'très', 'avec', 'pour', 'dans', 'sur', 'est', 'une', 'des']
    # Mots anglais caractéristiques  
    english_words = ['hello', 'thank', 'very', 'with', 'for', 'in', 'on', 'is', 'the', 'and']
    
    french_count = sum(1 for word in french_words if word in text_lower)
    english_count = sum(1 for word in english_words if word in text_lower)
    
    if french_count > english_count:
        hints.append('french')
    elif english_count > french_count:
        hints.append('english')
    
    return hints

def _extract_enhanced_keywords(content: str, normalized_watch=None, whatsapp_meta=None) -> List[str]:
    """Extrait des mots-clés enrichis pour améliorer la recherche"""
    keywords = []
    
    # 🕰️ Mots-clés de montres (compatible LLM et regex extracteurs)
    if normalized_watch:
        if normalized_watch.get('brand'):
            keywords.append(f"brand:{normalized_watch.get('brand')}")
        if normalized_watch.get('model'):
            keywords.append(f"model:{normalized_watch.get('model')}")
        if normalized_watch.get('price'):
            keywords.append(f"price_range:{_get_price_range(normalized_watch.get('price'))}")
        if normalized_watch.get('condition'):
            keywords.append(f"condition:{normalized_watch.get('condition')}")
        
        # 🎯 Mots-clés spécifiques LLM
        if normalized_watch.get('collection'):
            keywords.append(f"collection:{normalized_watch.get('collection')}")
        if normalized_watch.get('material') or normalized_watch.get('case_material'):
            material = normalized_watch.get('material') or normalized_watch.get('case_material')
            keywords.append(f"material:{material}")
        if normalized_watch.get('dial_color'):
            keywords.append(f"dial:{normalized_watch.get('dial_color')}")
        if normalized_watch.get('size'):
            keywords.append(f"size:{normalized_watch.get('size')}")
        
        # 📦 Accessoires détectés par LLM
        if normalized_watch.get('box_mentioned') or normalized_watch.get('has_box'):
            keywords.append("accessory:box")
        if normalized_watch.get('papers_mentioned') or normalized_watch.get('has_papers'):
            keywords.append("accessory:papers")
        if normalized_watch.get('warranty_mentioned') or normalized_watch.get('has_warranty'):
            keywords.append("accessory:warranty")
    
    # 📱 Mots-clés de contexte WhatsApp
    if whatsapp_meta:
        if whatsapp_meta.get('is_group_message'):
            keywords.append("source:group")
        else:
            keywords.append("source:private")
            
        semantic_meta = whatsapp_meta.get('semantic_metadata', {})
        intent_signals = semantic_meta.get('intent_signals', {})
        
        for intent, detected in intent_signals.items():
            if detected:
                keywords.append(f"intent:{intent}")
                
        # Ajouter la langue détectée
        language_hints = semantic_meta.get('text_analysis', {}).get('language_hints', [])
        for lang in language_hints:
            keywords.append(f"language:{lang}")
    
    # 📝 Mots-clés du contenu textuel
    content_lower = content.lower()
    
    # Marques populaires
    brands = ['rolex', 'omega', 'seiko', 'casio', 'tudor', 'breitling', 'tag heuer', 'cartier']
    for brand in brands:
        if brand in content_lower:
            keywords.append(f"brand_mentioned:{brand}")
    
    # Types de messages
    if any(word in content_lower for word in ['urgent', 'vite', 'rapidement']):
        keywords.append("urgency:high")
    if any(word in content_lower for word in ['négociable', 'negotiable', 'prix à débattre']):
        keywords.append("price:negotiable")
    
    return list(set(keywords))  # Supprime les doublons

def _get_price_range(price: float) -> str:
    """Détermine la gamme de prix pour les mots-clés"""
    if price < 500:
        return "entry_level"
    elif price < 2000:
        return "mid_range"
    elif price < 10000:
        return "luxury"
    else:
        return "high_end"

def _calculate_sentiment_score(content: str, semantic_meta: Dict) -> float:
    """Calcule un score de sentiment simple basé sur les mots-clés"""
    content_lower = content.lower()
    
    positive_words = ['excellent', 'parfait', 'superbe', 'magnifique', 'neuf', 'impeccable', 'great', 'perfect']
    negative_words = ['problème', 'défaut', 'cassé', 'rayé', 'usé', 'abîmé', 'broken', 'damaged']
    neutral_words = ['correct', 'standard', 'normal', 'moyen', 'acceptable']
    
    positive_count = sum(1 for word in positive_words if word in content_lower)
    negative_count = sum(1 for word in negative_words if word in content_lower)
    neutral_count = sum(1 for word in neutral_words if word in content_lower)
    
    total_words = positive_count + negative_count + neutral_count
    if total_words == 0:
        return 0.0
    
    # Score entre -1 et 1
    score = (positive_count - negative_count) / total_words
    return max(-1.0, min(1.0, score))

def _calculate_urgency_level(content: str, semantic_meta: Dict) -> int:
    """Calcule le niveau d'urgence (0-5)"""
    urgency_level = 0
    content_lower = content.lower()
    
    # Mots d'urgence forte
    high_urgency = ['urgent', 'rapidement', 'vite', 'asap', 'immédiatement']
    if any(word in content_lower for word in high_urgency):
        urgency_level += 3
    
    # Ponctuations d'urgence
    if content.count('!') >= 2:
        urgency_level += 2
    elif '!' in content:
        urgency_level += 1
    
    # Messages courts (souvent urgents)
    if len(content.split()) < 10:
        urgency_level += 1
    
    # Limiter à 5
    return min(5, urgency_level)

def _create_detailed_extraction(watch_info=None, whatsapp_meta=None) -> Dict:
    """Crée un objet d'extraction détaillé enrichi"""
    extraction = {}
    
    # Données de montres
    if watch_info:
        extraction['watch_extraction'] = watch_info.__dict__
    
    # Métadonnées WhatsApp
    if whatsapp_meta:
        extraction['whatsapp_context'] = {
            'sender_profile': {
                'name': whatsapp_meta.get('sender_profile_name'),
                'formatted_name': whatsapp_meta.get('sender_formatted_name'),
                'wa_id': whatsapp_meta.get('sender_wa_id')
            },
            'group_context': {
                'is_group': whatsapp_meta.get('is_group_message', False),
                'indicators': whatsapp_meta.get('group_context_indicators', [])
            },
            'conversation_context': {
                'has_reply': bool(whatsapp_meta.get('context_id')),
                'reply_to': whatsapp_meta.get('context_from'),
                'is_quoted': bool(whatsapp_meta.get('context_quoted'))
            }
        }
    
    extraction['extraction_timestamp'] = datetime.now().isoformat()
    return extraction

def _create_enhanced_search_metadata(watch_info=None, whatsapp_meta=None, semantic_meta=None) -> Dict:
    """Crée des métadonnées de recherche enrichies style Azure Search"""
    metadata = {
        'processed_at': datetime.now().isoformat(),
        'extraction_confidence': watch_info.confidence_score if watch_info else 0.0,
        'has_watch_info': bool(watch_info and watch_info.confidence_score > 0.2),
        
        # 🎯 ENRICHISSEMENTS SÉMANTIQUES
        'semantic_enrichment': {
            'intent_classification': {
                'primary_intent': _get_primary_intent(semantic_meta),
                'confidence_scores': _get_intent_confidence_scores(semantic_meta),
                'detected_intents': _get_detected_intents(semantic_meta)
            },
            
            'content_analysis': {
                'readability_score': _calculate_readability(whatsapp_meta),
                'information_density': _calculate_information_density(semantic_meta),
                'commercial_indicators': _get_commercial_indicators(semantic_meta)
            },
            
            'contextual_signals': {
                'source_reliability': _assess_source_reliability(whatsapp_meta),
                'temporal_relevance': _calculate_temporal_relevance(),
                'social_context': _extract_social_context(whatsapp_meta)
            }
        },
        
        # 🔍 MÉTADONNÉES POUR RECHERCHE AVANCÉE
        'search_optimization': {
            'boost_factors': _calculate_boost_factors(watch_info, semantic_meta),
            'filter_categories': _extract_filter_categories(watch_info, whatsapp_meta),
            'ranking_signals': _calculate_ranking_signals(watch_info, semantic_meta, whatsapp_meta)
        }
    }
    
    return metadata

def _get_primary_intent(semantic_meta: Dict) -> str:
    """Détermine l'intention primaire du message"""
    if not semantic_meta:
        return 'unknown'
    
    intent_signals = semantic_meta.get('intent_signals', {})
    if intent_signals.get('is_selling'):
        return 'selling'
    elif intent_signals.get('is_seeking'):
        return 'seeking'
    elif intent_signals.get('is_question'):
        return 'question'
    elif intent_signals.get('is_greeting'):
        return 'greeting'
    else:
        return 'general'

def _get_intent_confidence_scores(semantic_meta: Dict) -> Dict[str, float]:
    """Calcule les scores de confiance pour chaque intention"""
    if not semantic_meta:
        return {}
    
    intent_signals = semantic_meta.get('intent_signals', {})
    # Convertir les booléens en scores de confiance
    return {intent: 0.8 if detected else 0.1 for intent, detected in intent_signals.items()}

def _get_detected_intents(semantic_meta: Dict) -> List[str]:
    """Retourne la liste des intentions détectées"""
    if not semantic_meta:
        return []
    
    intent_signals = semantic_meta.get('intent_signals', {})
    return [intent for intent, detected in intent_signals.items() if detected]

def _calculate_readability(whatsapp_meta: Dict) -> float:
    """Calcule un score de lisibilité simple"""
    if not whatsapp_meta:
        return 0.5
    
    semantic_meta = whatsapp_meta.get('semantic_metadata', {})
    text_analysis = semantic_meta.get('text_analysis', {})
    
    word_count = text_analysis.get('word_count', 0)
    length = text_analysis.get('length', 0)
    
    if word_count == 0:
        return 0.0
    
    # Score basé sur la longueur moyenne des mots et la structure
    avg_word_length = length / word_count if word_count > 0 else 0
    
    # Score idéal autour de 5-6 caractères par mot
    readability = max(0.0, min(1.0, 1.0 - abs(avg_word_length - 5.5) / 10))
    return readability

def _calculate_information_density(semantic_meta: Dict) -> float:
    """Calcule la densité d'information du message"""
    if not semantic_meta:
        return 0.0
    
    text_analysis = semantic_meta.get('text_analysis', {})
    
    # Facteurs qui augmentent la densité d'information
    density_factors = 0
    if text_analysis.get('has_price'):
        density_factors += 0.3
    if text_analysis.get('has_phone'):
        density_factors += 0.2
    if text_analysis.get('has_email'):
        density_factors += 0.2
    if text_analysis.get('has_urls'):
        density_factors += 0.2
    
    return min(1.0, density_factors)

def _get_commercial_indicators(semantic_meta: Dict) -> List[str]:
    """Extrait les indicateurs commerciaux"""
    indicators = []
    
    if not semantic_meta:
        return indicators
    
    intent_signals = semantic_meta.get('intent_signals', {})
    text_analysis = semantic_meta.get('text_analysis', {})
    
    if intent_signals.get('is_selling'):
        indicators.append('selling_intent')
    if intent_signals.get('is_seeking'):
        indicators.append('buying_intent')
    if text_analysis.get('has_price'):
        indicators.append('price_mentioned')
    if intent_signals.get('has_urgency'):
        indicators.append('urgency')
    
    return indicators

def _assess_source_reliability(whatsapp_meta: Dict) -> float:
    """Évalue la fiabilité de la source"""
    if not whatsapp_meta:
        return 0.5
    
    reliability = 0.5  # Score de base
    
    # Profil complet augmente la fiabilité
    if whatsapp_meta.get('sender_profile_name'):
        reliability += 0.2
    if whatsapp_meta.get('sender_formatted_name'):
        reliability += 0.1
    
    # Messages de groupe peuvent être moins fiables
    if whatsapp_meta.get('is_group_message'):
        reliability -= 0.1
    
    return max(0.0, min(1.0, reliability))

def _calculate_temporal_relevance() -> float:
    """Calcule la pertinence temporelle (messages récents plus pertinents)"""
    return 1.0  # Pour l'instant, tous les messages sont récents

def _extract_social_context(whatsapp_meta: Dict) -> Dict[str, Any]:
    """Extrait le contexte social"""
    context = {
        'is_group_conversation': whatsapp_meta.get('is_group_message', False) if whatsapp_meta else False,
        'has_social_indicators': bool(whatsapp_meta.get('group_context_indicators')) if whatsapp_meta else False,
        'conversation_thread': bool(whatsapp_meta.get('context_id')) if whatsapp_meta else False
    }
    return context

def _calculate_boost_factors(watch_info=None, semantic_meta=None) -> Dict[str, float]:
    """Calcule les facteurs de boost pour le ranking"""
    boosts = {
        'relevance': 1.0,
        'completeness': 1.0,
        'freshness': 1.0,
        'authority': 1.0
    }
    
    # Boost pour les informations de montres complètes
    if watch_info and watch_info.confidence_score > 0.7:
        boosts['relevance'] = 1.5
        if watch_info.brand and watch_info.model and watch_info.price:
            boosts['completeness'] = 1.3
    
    # Boost pour les intentions claires
    if semantic_meta:
        intent_signals = semantic_meta.get('intent_signals', {})
        if any(intent_signals.values()):
            boosts['authority'] = 1.2
    
    return boosts

def _extract_filter_categories(watch_info=None, whatsapp_meta=None) -> List[str]:
    """Extrait les catégories pour le filtrage"""
    categories = []
    
    if watch_info:
        if watch_info.brand:
            categories.append(f"brand_{watch_info.brand.lower()}")
        if watch_info.message_type:
            categories.append(f"type_{watch_info.message_type}")
    
    if whatsapp_meta:
        if whatsapp_meta.get('is_group_message'):
            categories.append("source_group")
        else:
            categories.append("source_private")
    
    return categories

def _calculate_ranking_signals(watch_info=None, semantic_meta=None, whatsapp_meta=None) -> Dict[str, float]:
    """Calcule les signaux de ranking"""
    signals = {
        'content_quality': 0.5,
        'user_engagement': 0.5,
        'information_completeness': 0.5,
        'source_trust': 0.5
    }
    
    # Qualité du contenu
    if watch_info and watch_info.confidence_score:
        signals['content_quality'] = watch_info.confidence_score
    
    # Complétude des informations
    info_fields = 0
    total_fields = 5
    if watch_info:
        if watch_info.brand: info_fields += 1
        if watch_info.model: info_fields += 1
        if watch_info.price: info_fields += 1
        if watch_info.condition: info_fields += 1
        if watch_info.year: info_fields += 1
    
    signals['information_completeness'] = info_fields / total_fields
    
    # Confiance dans la source
    if whatsapp_meta:
        signals['source_trust'] = _assess_source_reliability(whatsapp_meta)
    
    return signals

async def process_message_with_rag(message: Dict[str, Any]) -> Dict[str, Any]:
    """Traite un message avec le système RAG et extraction de montres"""
    try:
        content = message.get('text', '')
        phone_number = message.get('from', '')
        
        # 🤖 EXTRACTION D'INFORMATIONS MONTRES AVEC LLM
        watch_info = None
        if watch_extractor:
            try:
                # Passer les métadonnées WhatsApp à l'extracteur LLM pour plus de contexte
                if hasattr(watch_extractor, 'extract_watch_info') and len(watch_extractor.extract_watch_info.__code__.co_varnames) > 2:
                    # Extracteur LLM - supporte les métadonnées
                    watch_info = watch_extractor.extract_watch_info(content, message)
                    logger.info(f"🤖 Extraction LLM: {watch_info.brand} {watch_info.model} - {watch_info.price}€ ({watch_info.message_type}) [Confiance: {watch_info.confidence_score:.2f}]")
                else:
                    # Extracteur regex - mode legacy
                    watch_info = watch_extractor.extract_watch_info(content)
                    logger.info(f"🔍 Extraction regex: {watch_info.brand} {watch_info.model} - {watch_info.price}€ ({watch_info.message_type})")
            except Exception as e:
                logger.error(f"❌ Erreur extraction montres: {e}")
        
        if embedding_processor:
            # 🎯 CRÉER L'EMBEDDING ENRICHI AVEC MÉTADONNÉES
            embedding = embedding_processor.generate_enhanced_embedding(content, message)
            
            if embedding:
                # Créer l'objet MessageEmbedding avec infos montres
                from src.embedding_processor import MessageEmbedding
                
                # Log des informations de montres extraites
                if watch_info and watch_info.confidence_score > 0.2:
                    logger.info(f"💎 Message enrichi avec données montres (confiance: {watch_info.confidence_score:.2f})")
                    logger.info(f"   🏷️ Marque: {watch_info.brand}, Modèle: {watch_info.model}")
                    logger.info(f"   💰 Prix: {watch_info.price} {watch_info.currency}, Condition: {watch_info.condition}")
                    logger.info(f"   📊 Type: {watch_info.message_type}")
                
                # 🕰️ CRÉATION DU MESSAGE ENRICHI AVEC DONNÉES MONTRES ET MÉTADONNÉES
                message_emb = create_message_embedding_from_watch_info(
                    phone_number=phone_number,
                    content=content,
                    embedding=embedding,
                    watch_info=watch_info,
                    whatsapp_metadata=message  # Passer toutes les métadonnées WhatsApp extraites
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
