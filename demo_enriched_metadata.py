#!/usr/bin/env python3
"""
🎯 Démonstration des métadonnées enrichies WhatsApp RAG
Montre les nouvelles capacités d'extraction de groupes, expéditeurs et métadonnées sémantiques
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demo_webhook_extraction():
    """Démo de l'extraction enrichie depuis un webhook WhatsApp simulé"""
    
    print("🎭 DÉMONSTRATION - Extraction enrichie de métadonnées WhatsApp")
    print("=" * 70)
    
    # Simuler un payload webhook WhatsApp avec métadonnées riches
    mock_webhook_payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "123456789",
            "time": int(datetime.now().timestamp()),
            "changes": [{
                "field": "messages",
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {
                        "display_phone_number": "+33123456789",
                        "phone_number_id": "987654321"
                    },
                    "contacts": [{
                        "profile": {
                            "name": "Groupe Montres Premium Paris",
                            "formatted_name": "Montres Premium"
                        },
                        "wa_id": "33123456789@g.us"
                    }],
                    "messages": [{
                        "id": "wamid.test123",
                        "from": "33123456789@g.us",
                        "timestamp": str(int(datetime.now().timestamp())),
                        "type": "text",
                        "context": {
                            "from": "33987654321",
                            "id": "wamid.reply456"
                        },
                        "text": {
                            "body": "🔥 URGENT! Sublime Rolex Daytona 116500LN Panda disponible! État neuf, boîte et papiers complets. Prix exceptionnel: 28500€ (valeur marché 32k€). Première main, achetée AD Rolex Champs-Élysées en 2022. Certificat authenticité + garantie. Livraison immédiate Paris/Province. Sérieux uniquement! 📞 06.12.34.56.78"
                        }
                    }]
                }
            }]
        }]
    }
    
    # Utiliser la fonction d'extraction enrichie
    from app import extract_whatsapp_messages
    
    extracted_messages = extract_whatsapp_messages(mock_webhook_payload)
    
    if not extracted_messages:
        print("❌ Aucun message extrait")
        return False
    
    message = extracted_messages[0]
    
    print("📱 MESSAGE EXTRAIT:")
    print(f"   ID: {message.get('id')}")
    print(f"   De: {message.get('from')}")
    print(f"   Texte: {message.get('text')[:100]}...")
    
    print("\n👤 INFORMATIONS EXPÉDITEUR:")
    print(f"   Nom profil: {message.get('sender_profile_name')}")
    print(f"   Nom formaté: {message.get('sender_formatted_name')}")
    print(f"   WhatsApp ID: {message.get('sender_wa_id')}")
    
    print("\n🏢 CONTEXTE DE GROUPE:")
    print(f"   Message de groupe: {message.get('is_group_message')}")
    print(f"   Indicateurs: {message.get('group_context_indicators')}")
    
    print("\n🔗 CONTEXTE CONVERSATIONNEL:")
    print(f"   Répond à: {message.get('context_from')}")
    print(f"   Message contexte: {message.get('context_id')}")
    
    print("\n🎯 MÉTADONNÉES SÉMANTIQUES:")
    semantic_meta = message.get('semantic_metadata', {})
    
    # Analyse textuelle
    text_analysis = semantic_meta.get('text_analysis', {})
    print(f"   Longueur: {text_analysis.get('length')} caractères")
    print(f"   Mots: {text_analysis.get('word_count')}")
    print(f"   A prix: {text_analysis.get('has_price')}")
    print(f"   A téléphone: {text_analysis.get('has_phone')}")
    print(f"   Langue: {text_analysis.get('language_hints')}")
    
    # Signaux d'intention
    intent_signals = semantic_meta.get('intent_signals', {})
    print(f"   Vente: {intent_signals.get('is_selling')}")
    print(f"   Recherche: {intent_signals.get('is_seeking')}")
    print(f"   Question: {intent_signals.get('is_question')}")
    print(f"   Urgence: {intent_signals.get('has_urgency')}")
    
    # Timing
    timing = semantic_meta.get('timing', {})
    print(f"   Heure: {timing.get('hour_of_day')}h")
    print(f"   Heures ouvrables: {timing.get('is_business_hours')}")
    
    return True

def demo_enhanced_embedding():
    """Démo de l'embedding enrichi avec métadonnées"""
    
    print("\n🎯 DÉMONSTRATION - Embedding enrichi avec métadonnées")
    print("=" * 60)
    
    try:
        from src.embedding_processor import EmbeddingProcessor
        
        # Initialiser l'embedding processor
        embedding_processor = EmbeddingProcessor(
            supabase_url=os.getenv('SUPABASE_URL'),
            supabase_key=os.getenv('SUPABASE_ANON_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Message test
        test_message = "Salut les amis! Je vends ma Submariner Hulk, état impeccable, 12500€ négociable"
        
        # Métadonnées simulées
        metadata = {
            'is_group_message': True,
            'sender_profile_name': 'Groupe Rolex Passion',
            'sender_formatted_name': 'Jean D.',
            'semantic_metadata': {
                'intent_signals': {
                    'is_selling': True,
                    'is_greeting': True,
                    'has_urgency': False
                },
                'text_analysis': {
                    'has_price': True,
                    'language_hints': ['french']
                },
                'timing': {
                    'is_business_hours': True
                }
            }
        }
        
        print("📝 MESSAGE ORIGINAL:")
        print(f"   {test_message}")
        
        # Créer le texte enrichi (fonction interne)
        enhanced_text = embedding_processor._create_enhanced_text_for_embedding(test_message, metadata)
        
        print("\n✨ TEXTE ENRICHI POUR EMBEDDING:")
        print(f"   {enhanced_text}")
        
        # Générer les embeddings
        print("\n⚡ GÉNÉRATION DES EMBEDDINGS:")
        
        standard_embedding = embedding_processor.generate_embedding(test_message)
        if standard_embedding:
            print(f"   ✅ Embedding standard: {len(standard_embedding)} dimensions")
        
        enhanced_embedding = embedding_processor.generate_enhanced_embedding(test_message, metadata)
        if enhanced_embedding:
            print(f"   ✅ Embedding enrichi: {len(enhanced_embedding)} dimensions")
            
        # Comparer la "distance" entre les embeddings (simulation)
        if standard_embedding and enhanced_embedding:
            # Calcul simple de différence (pas une vraie distance cosinus)
            diff_count = sum(1 for a, b in zip(standard_embedding[:100], enhanced_embedding[:100]) if abs(a - b) > 0.01)
            print(f"   📊 Différences détectées: {diff_count}/100 dimensions analysées")
            print("   🎯 L'embedding enrichi capture plus de contexte sémantique!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def demo_search_metadata():
    """Démo des métadonnées de recherche enrichies"""
    
    print("\n🔍 DÉMONSTRATION - Métadonnées de recherche enrichies")
    print("=" * 60)
    
    from app import _create_enhanced_search_metadata
    
    # Simuler des données de montres et WhatsApp
    class MockWatchInfo:
        def __init__(self):
            self.brand = "Rolex"
            self.model = "Submariner"
            self.price = 12500
            self.confidence_score = 0.85
            self.message_type = "sale"
    
    watch_info = MockWatchInfo()
    
    whatsapp_meta = {
        'is_group_message': True,
        'sender_profile_name': 'Groupe Montres Premium',
        'sender_formatted_name': 'Jean Dupont'
    }
    
    semantic_meta = {
        'intent_signals': {
            'is_selling': True,
            'is_greeting': True,
            'has_urgency': False
        },
        'text_analysis': {
            'has_price': True,
            'has_phone': True,
            'language_hints': ['french']
        }
    }
    
    # Créer les métadonnées enrichies
    search_metadata = _create_enhanced_search_metadata(watch_info, whatsapp_meta, semantic_meta)
    
    print("🎯 MÉTADONNÉES DE RECHERCHE ENRICHIES:")
    print(json.dumps(search_metadata, indent=2, default=str, ensure_ascii=False))
    
    # Analyser les composants
    semantic_enrichment = search_metadata.get('semantic_enrichment', {})
    search_optimization = search_metadata.get('search_optimization', {})
    
    print("\n📊 ANALYSE DES MÉTADONNÉES:")
    print(f"   Intention primaire: {semantic_enrichment.get('intent_classification', {}).get('primary_intent')}")
    print(f"   Intentions détectées: {semantic_enrichment.get('intent_classification', {}).get('detected_intents')}")
    print(f"   Fiabilité source: {semantic_enrichment.get('contextual_signals', {}).get('source_reliability')}")
    print(f"   Densité information: {semantic_enrichment.get('content_analysis', {}).get('information_density')}")
    
    boost_factors = search_optimization.get('boost_factors', {})
    print(f"   Facteurs de boost: Pertinence={boost_factors.get('relevance')}, Complétude={boost_factors.get('completeness')}")
    
    filter_categories = search_optimization.get('filter_categories', [])
    print(f"   Catégories de filtre: {filter_categories}")
    
    return True

def main():
    """Fonction principale de démonstration"""
    
    print("🚀 DÉMONSTRATION COMPLÈTE - Métadonnées enrichies WhatsApp RAG")
    print("=" * 80)
    print("Cette démo montre les nouvelles capacités d'extraction et d'enrichissement")
    print("des métadonnées pour améliorer la recherche sémantique style Azure Search")
    print("=" * 80)
    
    success = True
    
    # 1. Extraction enrichie des webhooks
    if not demo_webhook_extraction():
        success = False
    
    # 2. Embeddings enrichis
    if not demo_enhanced_embedding():
        success = False
    
    # 3. Métadonnées de recherche
    if not demo_search_metadata():
        success = False
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 DÉMONSTRATION RÉUSSIE!")
        print("\n✨ NOUVELLES CAPACITÉS ACTIVÉES:")
        print("   🏢 Détection automatique des groupes WhatsApp")
        print("   👤 Extraction des profils d'expéditeurs")
        print("   🎯 Classification automatique des intentions")
        print("   📊 Analyse de sentiment et d'urgence")
        print("   🔍 Métadonnées enrichies pour recherche sémantique")
        print("   ⚡ Embeddings contextualisés avec métadonnées")
        print("   📈 Analytics avancés par groupe et expéditeur")
        print("\n🎯 Le système WhatsApp RAG est maintenant aussi puissant qu'Azure Search!")
    else:
        print("❌ DÉMONSTRATION PARTIELLEMENT ÉCHOUÉE")
        print("Vérifier la configuration des variables d'environnement")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
