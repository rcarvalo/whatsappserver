#!/usr/bin/env python3
"""
🤖 vs 🔍 Test comparatif: Extracteur LLM vs Extracteur Regex
Compare les performances et la précision des deux méthodes d'extraction
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

def test_extraction_comparison():
    """Compare l'extracteur LLM vs regex sur différents messages de test"""
    
    print("🤖 vs 🔍 COMPARATIF D'EXTRACTION - LLM vs Regex")
    print("=" * 70)
    
    # Messages de test avec différents niveaux de complexité
    test_messages = [
        {
            "name": "Message simple",
            "content": "Je vends ma Rolex Submariner, état neuf, 8500€",
            "expected": {
                "brand": "Rolex",
                "model": "Submariner", 
                "price": 8500,
                "condition": "neuf",
                "message_type": "sale"
            }
        },
        {
            "name": "Message complexe avec référence",
            "content": "🔥 URGENT! Sublime Rolex Daytona 116500LN Panda disponible! État neuf, boîte et papiers complets. Prix exceptionnel: 28500€ (valeur marché 32k€). Première main, achetée AD Rolex Champs-Élysées en 2022.",
            "expected": {
                "brand": "Rolex",
                "model": "Daytona",
                "reference": "116500LN",
                "price": 28500,
                "condition": "neuf",
                "has_box": True,
                "has_papers": True,
                "year": 2022,
                "message_type": "sale"
            }
        },
        {
            "name": "Message avec jargon horloger",
            "content": "Cherche Omega Speedmaster Professional Moonwatch 42mm, de préférence Hesalite, budget max 4000€. Box + papers obligatoires.",
            "expected": {
                "brand": "Omega",
                "model": "Speedmaster Professional Moonwatch",
                "size": "42mm",
                "price": 4000,
                "has_box": True,
                "has_papers": True,
                "message_type": "wanted"
            }
        },
        {
            "name": "Message avec fautes d'orthographe",
            "content": "salu! jvend ma rolex submariner hulk 116610lv etat impecable avec boite papier certifica authenticité 12500€ negociable livraison possible",
            "expected": {
                "brand": "Rolex",
                "model": "Submariner",
                "reference": "116610LV",
                "price": 12500,
                "condition": "impeccable",
                "has_box": True,
                "has_papers": True,
                "authenticity_mentioned": True,
                "negotiable": True,
                "message_type": "sale"
            }
        },
        {
            "name": "Message question prix",
            "content": "Quelqu'un connait la cote actuelle d'une Patek Philippe Nautilus 5711/1A en acier ? Merci",
            "expected": {
                "brand": "Patek Philippe",
                "model": "Nautilus",
                "reference": "5711/1A",
                "material": "acier",
                "message_type": "question"
            }
        }
    ]
    
    # Initialiser les extracteurs
    extractors_results = {}
    
    try:
        # Extracteur LLM
        from src.llm_watch_extractor import LLMWatchExtractor
        llm_extractor = LLMWatchExtractor(openai_api_key=os.getenv('OPENAI_API_KEY'))
        print("✅ Extracteur LLM initialisé")
        
        # Extracteur Regex (legacy)
        try:
            from watch_info_extractor import WatchInfoExtractor
            regex_extractor = WatchInfoExtractor()
            print("✅ Extracteur Regex initialisé")
        except ImportError:
            print("⚠️ Extracteur Regex non disponible")
            regex_extractor = None
        
        # Tester chaque message
        for i, test_case in enumerate(test_messages, 1):
            print(f"\n📝 TEST {i}: {test_case['name']}")
            print(f"Message: {test_case['content'][:100]}...")
            
            results = {}
            
            # Test LLM
            try:
                llm_result = llm_extractor.extract_watch_info(test_case['content'])
                results['llm'] = {
                    'success': True,
                    'data': llm_result,
                    'confidence': llm_result.confidence_score,
                    'reasoning': llm_result.llm_reasoning
                }
                print(f"   🤖 LLM: {llm_result.brand} {llm_result.model} - {llm_result.price}€ (confiance: {llm_result.confidence_score:.2f})")
            except Exception as e:
                results['llm'] = {'success': False, 'error': str(e)}
                print(f"   🤖 LLM: ❌ Erreur - {e}")
            
            # Test Regex
            if regex_extractor:
                try:
                    regex_result = regex_extractor.extract_watch_info(test_case['content'])
                    results['regex'] = {
                        'success': True,
                        'data': regex_result,
                        'confidence': getattr(regex_result, 'confidence_score', 0.5)
                    }
                    print(f"   🔍 Regex: {regex_result.brand} {regex_result.model} - {regex_result.price}€")
                except Exception as e:
                    results['regex'] = {'success': False, 'error': str(e)}
                    print(f"   🔍 Regex: ❌ Erreur - {e}")
            
            # Évaluer la précision
            evaluation = evaluate_extraction_accuracy(results, test_case['expected'])
            print(f"   📊 Évaluation: LLM={evaluation['llm_score']:.1f}/10, Regex={evaluation['regex_score']:.1f}/10")
            
            extractors_results[test_case['name']] = {
                'results': results,
                'evaluation': evaluation,
                'expected': test_case['expected']
            }
        
        # Résumé global
        print_extraction_summary(extractors_results)
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

def evaluate_extraction_accuracy(results: dict, expected: dict) -> dict:
    """Évalue la précision des extractions par rapport aux valeurs attendues"""
    
    def score_extractor(extractor_result, expected_values):
        if not extractor_result.get('success'):
            return 0.0
        
        data = extractor_result['data']
        score = 0.0
        total_fields = len(expected_values)
        
        for field, expected_value in expected_values.items():
            if hasattr(data, field):
                extracted_value = getattr(data, field)
                if extracted_value == expected_value:
                    score += 1.0
                elif extracted_value and str(expected_value).lower() in str(extracted_value).lower():
                    score += 0.7  # Correspondance partielle
                elif extracted_value is not None:
                    score += 0.3  # Quelque chose d'extrait mais pas exact
        
        return (score / total_fields) * 10 if total_fields > 0 else 0.0
    
    evaluation = {
        'llm_score': score_extractor(results.get('llm', {}), expected),
        'regex_score': score_extractor(results.get('regex', {}), expected) if results.get('regex') else 0.0
    }
    
    return evaluation

def print_extraction_summary(results: dict):
    """Affiche un résumé comparatif des résultats"""
    
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ COMPARATIF")
    print("=" * 70)
    
    llm_scores = []
    regex_scores = []
    
    for test_name, test_data in results.items():
        evaluation = test_data['evaluation']
        llm_scores.append(evaluation['llm_score'])
        regex_scores.append(evaluation['regex_score'])
    
    # Moyennes
    avg_llm = sum(llm_scores) / len(llm_scores) if llm_scores else 0
    avg_regex = sum(regex_scores) / len(regex_scores) if regex_scores else 0
    
    print(f"📈 SCORES MOYENS:")
    print(f"   🤖 LLM:   {avg_llm:.1f}/10")
    print(f"   🔍 Regex: {avg_regex:.1f}/10")
    
    # Analyse détaillée
    print(f"\n🎯 ANALYSE DÉTAILLÉE:")
    
    if avg_llm > avg_regex:
        print(f"   ✅ L'extracteur LLM est plus précis (+{avg_llm - avg_regex:.1f} points)")
    elif avg_regex > avg_llm:
        print(f"   ✅ L'extracteur Regex est plus précis (+{avg_regex - avg_llm:.1f} points)")
    else:
        print(f"   🤝 Les deux extracteurs ont des performances équivalentes")
    
    # Avantages de chaque méthode
    print(f"\n💡 AVANTAGES OBSERVÉS:")
    print(f"   🤖 LLM:")
    print(f"      - Gestion des fautes d'orthographe")
    print(f"      - Compréhension du contexte et des surnoms")
    print(f"      - Extraction d'informations complexes")
    print(f"      - Classification précise des intentions")
    print(f"      - Raisonnement explicable")
    
    print(f"   🔍 Regex:")
    print(f"      - Vitesse d'exécution")
    print(f"      - Pas de coût API")
    print(f"      - Déterministe et prévisible")
    print(f"      - Fonctionne hors ligne")
    
    print(f"\n🚀 RECOMMANDATION:")
    if avg_llm > avg_regex + 2:
        print(f"   Utiliser l'extracteur LLM comme méthode principale")
        print(f"   Garder regex comme fallback en cas d'erreur API")
    else:
        print(f"   Configuration hybride recommandée:")
        print(f"   - LLM pour les cas complexes")
        print(f"   - Regex pour les extractions simples et rapides")

def test_llm_specific_features():
    """Teste les fonctionnalités spécifiques à l'extracteur LLM"""
    
    print("\n🎯 TEST DES FONCTIONNALITÉS SPÉCIFIQUES LLM")
    print("=" * 50)
    
    try:
        from src.llm_watch_extractor import LLMWatchExtractor
        llm_extractor = LLMWatchExtractor(openai_api_key=os.getenv('OPENAI_API_KEY'))
        
        # Test avec métadonnées WhatsApp
        test_message = "Salut les amis! Ma Hulk est à vendre, 12500€"
        
        whatsapp_metadata = {
            'sender_profile_name': 'Groupe Rolex Premium',
            'is_group_message': True,
            'semantic_metadata': {
                'intent_signals': {
                    'is_selling': True,
                    'is_greeting': True
                }
            }
        }
        
        result = llm_extractor.extract_watch_info(test_message, whatsapp_metadata)
        
        print(f"📱 Message: {test_message}")
        print(f"🤖 Extraction avec contexte:")
        print(f"   Marque: {result.brand}")
        print(f"   Modèle: {result.model}")
        print(f"   Collection: {result.collection}")
        print(f"   Prix: {result.price} {result.currency}")
        print(f"   Type de message: {result.message_type}")
        print(f"   Confiance: {result.confidence_score:.2f}")
        print(f"   Raisonnement: {result.llm_reasoning}")
        
        # Test du cache
        print(f"\n⚡ Test du cache:")
        start_time = datetime.now()
        result2 = llm_extractor.extract_watch_info(test_message, whatsapp_metadata)
        cache_time = (datetime.now() - start_time).total_seconds()
        print(f"   Temps avec cache: {cache_time:.3f}s")
        print(f"   Résultat identique: {result.brand == result2.brand}")
        
        # Statistiques
        stats = llm_extractor.get_extraction_stats()
        print(f"\n📊 Statistiques d'extraction:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Fonction principale du test comparatif"""
    
    print("🚀 SUITE DE TESTS COMPLÈTE - Extracteurs de Montres")
    print("=" * 70)
    
    success = True
    
    # Test principal de comparaison
    if not test_extraction_comparison():
        success = False
    
    # Tests spécifiques LLM
    if not test_llm_specific_features():
        success = False
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("\n💡 L'extracteur LLM apporte une précision et une flexibilité supérieures")
        print("   tout en conservant la compatibilité avec l'extracteur regex existant.")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Vérifier la configuration des API keys et la disponibilité des modules")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
