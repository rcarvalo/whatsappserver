#!/usr/bin/env python3
"""
🧪 Test d'intégration pour le nouveau schéma watch_conversations
Valide que l'extraction de montres fonctionne avec la nouvelle base de données
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_watch_extraction_and_storage():
    """Test complet d'extraction et stockage de montres"""
    
    print("🧪 Test d'intégration - Nouveau schéma watch_conversations")
    print("=" * 60)
    
    try:
        # 1. Initialiser les modules
        from src.llm_watch_extractor import LLMWatchExtractor
        from src.embedding_processor import EmbeddingProcessor
        from app import create_message_embedding_from_watch_info
        
        # Variables d'environnement requises
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY') 
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not all([supabase_url, supabase_key, openai_api_key]):
            print("❌ Variables d'environnement manquantes")
            print(f"   SUPABASE_URL: {'✅' if supabase_url else '❌'}")
            print(f"   SUPABASE_KEY: {'✅' if supabase_key else '❌'}")
            print(f"   OPENAI_API_KEY: {'✅' if openai_api_key else '❌'}")
            return False
            
        print("✅ Variables d'environnement OK")
        
        # 2. Initialiser les extracteurs (LLM prioritaire)
        watch_extractor = LLMWatchExtractor(openai_api_key=openai_api_key)
        embedding_processor = EmbeddingProcessor(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            openai_api_key=openai_api_key
        )
        
        print("✅ Modules initialisés")
        
        # 3. Message de test avec informations de montre et métadonnées WhatsApp simulées
        test_message = """
        Salut ! Je vends ma Rolex Submariner 116610LV Hulk.
        État impeccable, boîte et papiers, achetée en 2019.
        Prix: 12500€, négociable.
        Livraison possible en France.
        Certificat d'authenticité inclus.
        """
        
        # Simuler les métadonnées WhatsApp enrichies
        mock_whatsapp_metadata = {
            'id': 'test_msg_001',
            'from': '+33123456789',
            'sender_profile_name': 'Jean Dupont - Montres Premium',
            'sender_formatted_name': 'Jean D.',
            'sender_wa_id': '+33123456789',
            'is_group_message': True,
            'group_context_indicators': ['groupe', 'vente'],
            'context_id': None,
            'context_from': None,
            'semantic_metadata': {
                'sender': {
                    'wa_id': '+33123456789',
                    'profile_name': 'Jean Dupont - Montres Premium',
                    'formatted_name': 'Jean D.',
                    'is_verified': False
                },
                'conversation': {
                    'has_context': False,
                    'is_reply': False,
                    'reply_to': None,
                    'thread_context': False
                },
                'text_analysis': {
                    'length': len(test_message),
                    'word_count': len(test_message.split()),
                    'has_urls': False,
                    'has_phone': False,
                    'has_email': False,
                    'has_price': True,
                    'language_hints': ['french']
                },
                'timing': {
                    'processed_at': datetime.now().isoformat(),
                    'hour_of_day': datetime.now().hour,
                    'day_of_week': datetime.now().weekday(),
                    'is_business_hours': 9 <= datetime.now().hour <= 18
                },
                'intent_signals': {
                    'is_selling': True,
                    'is_seeking': False,
                    'is_question': False,
                    'is_greeting': True,
                    'has_urgency': False
                }
            }
        }
        
        print("📱 Message de test:")
        print(f"   {test_message.strip()}")
        
        # 4. Extraction des informations de montre avec LLM + métadonnées
        watch_info = watch_extractor.extract_watch_info(test_message, mock_whatsapp_metadata)
        
        print("\n🤖 Extraction LLM des informations de montre:")
        if watch_info:
            print(f"   🏷️  Marque: {watch_info.brand}")
            print(f"   ⌚  Modèle: {watch_info.model}")
            print(f"   📦  Collection: {watch_info.collection}")
            print(f"   🔢  Référence: {watch_info.reference}")
            print(f"   💰  Prix: {watch_info.price} {watch_info.currency}")
            print(f"   📊  Type: {watch_info.message_type}")
            print(f"   🎯  Confiance: {watch_info.confidence_score:.2f}")
            print(f"   🧠  Raisonnement: {watch_info.llm_reasoning[:100]}...")
            
            # Afficher les informations enrichies spécifiques au LLM
            if watch_info.has_box:
                print(f"   📦  Boîte: Oui")
            if watch_info.has_papers:
                print(f"   📄  Papiers: Oui")
            if watch_info.condition_details:
                print(f"   🔍  Détails état: {watch_info.condition_details}")
            if watch_info.material:
                print(f"   🏗️  Matériau: {watch_info.material}")
        else:
            print("   ❌ Aucune information extraite")
            return False
            
        # 5. Test de l'embedding enrichi avec métadonnées
        print("\n🎯 Test de l'embedding enrichi:")
        
        # Embedding standard
        embedding_standard = embedding_processor.generate_embedding(test_message)
        if not embedding_standard:
            print("❌ Erreur génération embedding standard")
            return False
        print("   ✅ Embedding standard généré")
        
        # Embedding enrichi avec métadonnées
        embedding_enhanced = embedding_processor.generate_enhanced_embedding(test_message, mock_whatsapp_metadata)
        if not embedding_enhanced:
            print("❌ Erreur génération embedding enrichi")
            return False
        print("   ✅ Embedding enrichi généré")
        
        # Utiliser l'embedding enrichi pour la suite
        embedding = embedding_enhanced
        
        # 6. Création du MessageEmbedding enrichi avec métadonnées WhatsApp
        message_emb = create_message_embedding_from_watch_info(
            phone_number="+33123456789",
            content=test_message,
            embedding=embedding,
            watch_info=watch_info,
            whatsapp_metadata=mock_whatsapp_metadata
        )
        
        print("✅ MessageEmbedding créé avec données montres")
        
        # 7. Stockage en base de données
        result = embedding_processor.store_message_embedding(message_emb)
        
        if result:
            print(f"✅ Message stocké avec ID: {result.get('id')}")
            print("\n🎉 INTÉGRATION ENRICHIE RÉUSSIE!")
            print("\n📊 Données stockées dans watch_conversations:")
            print(f"   - Marque: {result.get('watch_brand')}")
            print(f"   - Modèle: {result.get('watch_model')}")
            print(f"   - Prix: {result.get('price_mentioned')} {result.get('currency')}")
            print(f"   - Type de message: {result.get('message_type')}")
            print(f"   - Score d'intention: {result.get('intent_score')}")
            print(f"   - Groupe: {result.get('group_name')}")
            print(f"   - Expéditeur: {result.get('sender')}")
            print(f"   - Mots-clés enrichis: {len(result.get('extracted_keywords', []))} mots-clés")
            print(f"   - Score sentiment: {result.get('sentiment_score')}")
            print(f"   - Niveau urgence: {result.get('urgency_level')}")
            
            # Afficher quelques métadonnées enrichies
            search_metadata = result.get('search_metadata', {})
            if search_metadata:
                semantic_enrichment = search_metadata.get('semantic_enrichment', {})
                intent_classification = semantic_enrichment.get('intent_classification', {})
                print(f"   - Intention primaire: {intent_classification.get('primary_intent')}")
                print(f"   - Intentions détectées: {intent_classification.get('detected_intents', [])}")
            
            return True
        else:
            print("❌ Erreur lors du stockage")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_functions():
    """Test des nouvelles fonctions de recherche"""
    
    print("\n🔍 Test des fonctions de recherche")
    print("=" * 40)
    
    try:
        from src.embedding_processor import EmbeddingProcessor
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        embedding_processor = EmbeddingProcessor(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Test de la fonction search_watches_by_criteria
        result = embedding_processor.supabase.rpc('search_watches_by_criteria', {
            'brand_query': 'Rolex',
            'days_back': 30
        }).execute()
        
        print(f"✅ Fonction search_watches_by_criteria testée")
        print(f"   Résultats trouvés: {len(result.data) if result.data else 0}")
        
        # Test de la vue active_watch_sales
        result = embedding_processor.supabase.table('active_watch_sales').select('*').limit(5).execute()
        
        print(f"✅ Vue active_watch_sales testée")
        print(f"   Annonces actives: {len(result.data) if result.data else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test recherche: {e}")
        return False

if __name__ == "__main__":
    success = test_watch_extraction_and_storage()
    
    if success:
        test_search_functions()
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("Le serveur est maintenant compatible avec le nouveau schéma watch_conversations")
    else:
        print("\n❌ Tests échoués - vérifier la configuration")
        sys.exit(1)
