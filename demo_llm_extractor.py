#!/usr/bin/env python3
"""
🤖 Démonstration de l'Extracteur LLM de Montres
Exemples d'utilisation et cas d'usage avancés
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

def demo_basic_extraction():
    """Démonstration de l'extraction basique"""
    
    print("🤖 DÉMONSTRATION - Extracteur LLM Basique")
    print("=" * 50)
    
    try:
        from src.llm_watch_extractor import LLMWatchExtractor
        
        # Initialiser l'extracteur
        extractor = LLMWatchExtractor(openai_api_key=os.getenv('OPENAI_API_KEY'))
        
        # Messages d'exemple
        messages = [
            "Je vends ma Rolex Submariner Date 116610LN, état neuf avec boîte et papiers. Prix: 9500€",
            "Cherche Omega Speedmaster Professional Moonwatch, budget max 4000€",
            "Quelqu'un connaît la cote d'une Patek Philippe Nautilus ?",
            "URGENT! Daytona Panda 116500LN disponible, 28k€ négociable",
            "Ma collection se sépare: PP 5711, AP 15400, Rolex Hulk - Prix sur demande"
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"\n📱 MESSAGE {i}: {message}")
            
            # Extraction
            result = extractor.extract_watch_info(message)
            
            print(f"🤖 EXTRACTION:")
            print(f"   Marque: {result.brand}")
            print(f"   Modèle: {result.model}")
            print(f"   Prix: {result.price} {result.currency}")
            print(f"   Type: {result.message_type}")
            print(f"   Confiance: {result.confidence_score:.2f}")
            
            if result.llm_reasoning:
                print(f"   Raisonnement: {result.llm_reasoning[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def demo_advanced_extraction():
    """Démonstration avec métadonnées WhatsApp"""
    
    print("\n🎯 DÉMONSTRATION - Extraction avec Contexte WhatsApp")
    print("=" * 55)
    
    try:
        from src.llm_watch_extractor import LLMWatchExtractor
        
        extractor = LLMWatchExtractor(openai_api_key=os.getenv('OPENAI_API_KEY'))
        
        # Message avec contexte
        message = "Les gars, ma Hulk est dispo si ça intéresse quelqu'un. État impec, full set. 12.5k"
        
        # Métadonnées WhatsApp enrichies
        metadata = {
            'sender_profile_name': 'Groupe Rolex Passion France',
            'sender_formatted_name': 'Pierre M.',
            'is_group_message': True,
            'group_context_indicators': ['groupe', 'passion'],
            'semantic_metadata': {
                'intent_signals': {
                    'is_selling': True,
                    'is_greeting': True,
                    'has_urgency': False
                },
                'text_analysis': {
                    'language_hints': ['french'],
                    'has_price': True,
                    'word_count': len(message.split())
                },
                'timing': {
                    'is_business_hours': True
                }
            }
        }
        
        print(f"📱 MESSAGE: {message}")
        print(f"🏢 CONTEXTE: Groupe '{metadata['sender_profile_name']}'")
        
        # Extraction avec contexte
        result = extractor.extract_watch_info(message, metadata)
        
        print(f"\n🤖 EXTRACTION ENRICHIE:")
        print(f"   Marque: {result.brand}")
        print(f"   Modèle: {result.model}")
        print(f"   Collection: {result.collection}")
        print(f"   Prix: {result.price} {result.currency}")
        print(f"   Type de prix: {result.price_type}")
        print(f"   Condition: {result.condition}")
        print(f"   Boîte: {'Oui' if result.has_box else 'Non' if result.has_box is False else 'Non mentionné'}")
        print(f"   Papiers: {'Oui' if result.has_papers else 'Non' if result.has_papers is False else 'Non mentionné'}")
        print(f"   Type message: {result.message_type}")
        print(f"   Négociable: {'Oui' if result.negotiable else 'Non' if result.negotiable is False else 'Non précisé'}")
        print(f"   Confiance: {result.confidence_score:.2f}")
        
        print(f"\n🧠 RAISONNEMENT LLM:")
        print(f"   {result.llm_reasoning}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def demo_batch_extraction():
    """Démonstration d'extraction par lots"""
    
    print("\n⚡ DÉMONSTRATION - Extraction par Lots")
    print("=" * 45)
    
    try:
        from src.llm_watch_extractor import LLMWatchExtractor
        
        extractor = LLMWatchExtractor(openai_api_key=os.getenv('OPENAI_API_KEY'))
        
        # Messages multiples
        batch_messages = [
            {
                'content': 'Vends Rolex GMT Master II 126710BLRO, état neuf, 14000€',
                'metadata': {'sender_profile_name': 'Jean Dupont'}
            },
            {
                'content': 'Omega Seamaster Planet Ocean 600m, orange, 3500€',
                'metadata': {'sender_profile_name': 'Marc L.'}
            },
            {
                'content': 'ISO: Patek Philippe Aquanaut, budget flexible',
                'metadata': {'sender_profile_name': 'Collectionneur_Paris'}
            }
        ]
        
        print(f"📦 Traitement de {len(batch_messages)} messages...")
        
        # Extraction par lot
        results = extractor.extract_batch(batch_messages)
        
        for i, (message, result) in enumerate(zip(batch_messages, results), 1):
            print(f"\n📱 MESSAGE {i}: {message['content'][:50]}...")
            print(f"   Marque: {result.brand}")
            print(f"   Modèle: {result.model}")
            print(f"   Prix: {result.price}")
            print(f"   Type: {result.message_type}")
            print(f"   Confiance: {result.confidence_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def demo_statistics_and_cache():
    """Démonstration des statistiques et du cache"""
    
    print("\n📊 DÉMONSTRATION - Statistiques et Cache")
    print("=" * 45)
    
    try:
        from src.llm_watch_extractor import LLMWatchExtractor
        
        extractor = LLMWatchExtractor(openai_api_key=os.getenv('OPENAI_API_KEY'))
        
        # Plusieurs extractions pour générer des stats
        test_messages = [
            "Rolex Daytona acier, 25000€",
            "Omega Speedmaster, parfait état, 4000€",
            "Cherche Patek Philippe Nautilus",
            "Tudor Black Bay, quasi neuve, 2800€"
        ]
        
        print("⚡ Extraction de messages pour statistiques...")
        
        for message in test_messages:
            extractor.extract_watch_info(message)
        
        # Test du cache (même message)
        print(f"\n🔄 Test du cache...")
        start_time = datetime.now()
        extractor.extract_watch_info(test_messages[0])  # Déjà en cache
        cache_time = (datetime.now() - start_time).total_seconds()
        print(f"   Temps avec cache: {cache_time:.3f}s")
        
        # Statistiques
        stats = extractor.get_extraction_stats()
        print(f"\n📈 STATISTIQUES D'EXTRACTION:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        # Nettoyage du cache
        extractor.clear_cache()
        print(f"\n🧹 Cache vidé")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def demo_complex_scenarios():
    """Scénarios complexes pour tester les limites"""
    
    print("\n🎭 DÉMONSTRATION - Scénarios Complexes")
    print("=" * 45)
    
    try:
        from src.llm_watch_extractor import LLMWatchExtractor
        
        extractor = LLMWatchExtractor(openai_api_key=os.getenv('OPENAI_API_KEY'))
        
        complex_scenarios = [
            {
                'name': 'Message avec fautes',
                'content': 'salu jvend ma rolex submariner hulk etat impecable boite papier 12500e negociable'
            },
            {
                'name': 'Multiples montres',
                'content': 'Je vends ma collection: Rolex Sub (8k€), Omega Speedmaster (4k€), Tudor BB58 (3k€)'
            },
            {
                'name': 'Jargon technique',
                'content': 'Daytona 116500LN Panda, movement 4130, ceramic bezel, COSC certified, AD purchased'
            },
            {
                'name': 'Message émotionnel',
                'content': '😢 Je dois me séparer de ma Nautilus 5711/1A bleue... Prix de marché actuel 150k€ mais je cherche collectionneur sérieux'
            }
        ]
        
        for scenario in complex_scenarios:
            print(f"\n🎯 {scenario['name'].upper()}:")
            print(f"   Message: {scenario['content']}")
            
            result = extractor.extract_watch_info(scenario['content'])
            
            print(f"   🤖 Résultat:")
            print(f"      Marque: {result.brand}")
            print(f"      Modèle: {result.model}")
            print(f"      Prix: {result.price}")
            print(f"      Type: {result.message_type}")
            print(f"      Confiance: {result.confidence_score:.2f}")
            
            if result.confidence_score > 0.7:
                print(f"      ✅ Extraction de haute qualité")
            elif result.confidence_score > 0.4:
                print(f"      ⚠️ Extraction acceptable")
            else:
                print(f"      ❌ Extraction incertaine")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Fonction principale de démonstration"""
    
    print("🤖 SUITE COMPLÈTE DE DÉMONSTRATIONS - Extracteur LLM")
    print("=" * 65)
    print("Cette démo montre toutes les capacités de l'extracteur LLM")
    print("pour l'analyse précise des messages de vente de montres")
    print("=" * 65)
    
    success = True
    
    # Vérifier la clé API
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY non configurée")
        print("Configurer la variable d'environnement pour exécuter cette démo")
        return False
    
    # Démonstrations
    demos = [
        ("Extraction basique", demo_basic_extraction),
        ("Extraction avec contexte", demo_advanced_extraction),
        ("Extraction par lots", demo_batch_extraction),
        ("Statistiques et cache", demo_statistics_and_cache),
        ("Scénarios complexes", demo_complex_scenarios)
    ]
    
    for demo_name, demo_func in demos:
        print(f"\n🎬 Lancement: {demo_name}")
        if not demo_func():
            print(f"❌ Échec de la démo: {demo_name}")
            success = False
        else:
            print(f"✅ Succès de la démo: {demo_name}")
    
    print("\n" + "=" * 65)
    if success:
        print("🎉 TOUTES LES DÉMONSTRATIONS RÉUSSIES!")
        print("\n💡 AVANTAGES DE L'EXTRACTEUR LLM:")
        print("   🎯 Précision supérieure aux regex")
        print("   🧠 Compréhension du contexte et des nuances")
        print("   🌍 Gestion des fautes d'orthographe et du jargon")
        print("   📊 Classification automatique des intentions")
        print("   🔍 Extraction d'informations complexes")
        print("   ⚡ Cache intelligent pour optimiser les performances")
        print("   📈 Statistiques et monitoring intégrés")
        print("\n🚀 Votre système WhatsApp RAG est prêt pour la production!")
    else:
        print("❌ CERTAINES DÉMONSTRATIONS ONT ÉCHOUÉ")
        print("Vérifier la configuration des API keys")
    
    print("=" * 65)

if __name__ == "__main__":
    main()
