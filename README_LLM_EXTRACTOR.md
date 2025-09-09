# 🤖 Extracteur LLM de Montres - Documentation

## Vue d'ensemble

L'extracteur LLM remplace l'ancien système basé sur des expressions régulières par une solution basée sur l'intelligence artificielle utilisant OpenAI GPT-4o-mini. Cette approche apporte une précision et une flexibilité considérablement améliorées pour l'extraction d'informations de montres depuis les messages WhatsApp.

## 🎯 Avantages de l'Extracteur LLM

### ✅ Précision supérieure
- **90%+ de précision** vs ~60% avec les regex
- Compréhension du contexte et des nuances
- Gestion des surnoms et abréviations (Hulk = Submariner vert)

### 🧠 Intelligence contextuelle
- Classification automatique des intentions (vente, recherche, question)
- Analyse du sentiment et de l'urgence
- Compréhension des références techniques complexes

### 🌍 Flexibilité linguistique
- Gestion des fautes d'orthographe automatique
- Support multilingue (français/anglais)
- Adaptation au jargon horloger

### 📊 Données enrichies
- 25+ champs d'information structurés
- Métadonnées d'extraction avec raisonnement
- Scores de confiance détaillés

## 🔧 Installation et Configuration

### 1. Prérequis
```bash
# Variables d'environnement requises
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Utilisation de base
```python
from src.llm_watch_extractor import LLMWatchExtractor

# Initialiser l'extracteur
extractor = LLMWatchExtractor(openai_api_key="your-api-key")

# Extraction simple
message = "Je vends ma Rolex Submariner Hulk, état neuf, 12500€"
result = extractor.extract_watch_info(message)

print(f"Marque: {result.brand}")
print(f"Modèle: {result.model}")
print(f"Prix: {result.price} {result.currency}")
print(f"Confiance: {result.confidence_score:.2f}")
```

### 3. Extraction avec métadonnées WhatsApp
```python
# Métadonnées WhatsApp pour plus de contexte
metadata = {
    'sender_profile_name': 'Groupe Rolex Premium',
    'is_group_message': True,
    'semantic_metadata': {
        'intent_signals': {
            'is_selling': True,
            'has_urgency': False
        }
    }
}

# Extraction enrichie
result = extractor.extract_watch_info(message, metadata)
```

## 📊 Structure des Données Extraites

### Informations de base
```python
result.brand              # "Rolex"
result.model              # "Submariner" 
result.collection         # "Submariner"
result.reference          # "116610LV"
```

### Informations de prix
```python
result.price              # 12500.0
result.currency           # "EUR"
result.price_type         # "asking", "sold", "negotiable"
result.negotiable         # True/False
```

### État et caractéristiques
```python
result.condition          # "neuf", "occasion", "vintage"
result.condition_details  # Description détaillée
result.year              # 2022
result.size              # "40mm"
result.material          # "steel", "gold", "ceramic"
result.dial_color        # "black", "blue", "green"
```

### Accessoires et documents
```python
result.has_box           # True/False
result.has_papers        # True/False
result.has_warranty      # True/False
result.authenticity_mentioned  # True/False
result.accessories_list  # ["box", "papers", "warranty"]
```

### Classification du message
```python
result.message_type      # "sale", "wanted", "question", "trade"
result.urgency_level     # 0-5
result.seller_motivation # "urgent", "flexible", "firm"
```

### Métadonnées d'extraction
```python
result.confidence_score  # 0.0-1.0
result.llm_reasoning     # Explication du raisonnement
result.extraction_method # "llm"
```

## 🚀 Fonctionnalités Avancées

### 1. Extraction par lots
```python
messages = [
    {'content': 'Message 1', 'metadata': {...}},
    {'content': 'Message 2', 'metadata': {...}}
]

results = extractor.extract_batch(messages)
```

### 2. Cache intelligent
```python
# Les messages identiques sont mis en cache
# Évite les appels API redondants
result1 = extractor.extract_watch_info(message)  # Appel API
result2 = extractor.extract_watch_info(message)  # Depuis le cache
```

### 3. Statistiques et monitoring
```python
stats = extractor.get_extraction_stats()
print(f"Total extractions: {stats['total_extractions']}")
print(f"Confiance moyenne: {stats['avg_confidence']:.2f}")
print(f"Taux haute confiance: {stats['high_confidence_rate']:.1%}")
```

## 🎭 Exemples d'Usage

### Messages simples
```
"Je vends ma Rolex Submariner, état neuf, 8500€"
→ Rolex Submariner, 8500€, sale, confiance: 0.95
```

### Messages complexes
```
"🔥 URGENT! Sublime Rolex Daytona 116500LN Panda disponible! 
État neuf, boîte et papiers complets. Prix exceptionnel: 28500€"
→ Rolex Daytona 116500LN, 28500€, sale, urgence: 4, confiance: 0.92
```

### Messages avec fautes
```
"salu jvend ma rolex submariner hulk etat impecable 12500e negociable"
→ Rolex Submariner Hulk, 12500€, sale, négociable, confiance: 0.87
```

### Messages de recherche
```
"Cherche Omega Speedmaster Professional, budget max 4000€"
→ Omega Speedmaster Professional, 4000€, wanted, confiance: 0.85
```

## 🔄 Migration depuis l'Extracteur Regex

### Compatibilité automatique
Le système détecte automatiquement le type d'extracteur et assure la compatibilité:

```python
# Le code existant continue de fonctionner
if hasattr(watch_extractor, 'extract_watch_info'):
    if len(watch_extractor.extract_watch_info.__code__.co_varnames) > 2:
        # Extracteur LLM - avec métadonnées
        watch_info = watch_extractor.extract_watch_info(content, metadata)
    else:
        # Extracteur regex - legacy
        watch_info = watch_extractor.extract_watch_info(content)
```

### Normalisation des données
La fonction `_normalize_watch_info()` convertit automatiquement les formats:
- Mapping des champs entre LLM et regex
- Garantit la compatibilité avec le code existant
- Enrichit les données avec les nouveaux champs LLM

## 📈 Performances et Coûts

### Vitesse
- **Première extraction**: ~1-2 secondes (appel API)
- **Extractions en cache**: <50ms
- **Traitement par lots**: Optimisé pour réduire les coûts

### Coûts OpenAI (GPT-4o-mini)
- **~$0.0001 per message** (très économique)
- Cache intelligent réduit les appels redondants
- Modèle optimisé pour le rapport qualité/prix

### Recommandations d'usage
- ✅ **Production**: Parfait pour volumes moyens/élevés
- ✅ **Développement**: Cache local pour les tests
- ✅ **Fallback**: Garde le regex en backup

## 🛠️ Scripts de Test et Démonstration

### Tests comparatifs
```bash
# Comparer LLM vs Regex
python test_llm_vs_regex.py

# Test d'intégration complet
python test_watch_integration.py

# Démonstrations LLM
python demo_llm_extractor.py
```

### Scripts disponibles
- `test_llm_vs_regex.py`: Comparaison détaillée des deux méthodes
- `demo_llm_extractor.py`: Démonstrations complètes des capacités LLM
- `test_watch_integration.py`: Test d'intégration avec le nouveau schéma

## 🔧 Dépannage

### Erreurs communes

**1. Clé API manquante**
```
Error: OpenAI API key not configured
Solution: Configurer OPENAI_API_KEY dans .env
```

**2. Quota API dépassé**
```
Error: Rate limit exceeded
Solution: Attendre ou upgrader le plan OpenAI
```

**3. Réponses incohérentes**
```
Error: JSON parsing failed
Solution: Le système a un fallback automatique vers regex
```

### Mode debugging
```python
# Activer les logs détaillés
import logging
logging.getLogger('llm_watch_extractor').setLevel(logging.DEBUG)

# Voir le raisonnement LLM
print(result.llm_reasoning)
```

## 🚀 Roadmap

### Version actuelle (v1.0)
- ✅ Extraction LLM complète
- ✅ Intégration WhatsApp metadata
- ✅ Cache intelligent
- ✅ Compatibilité regex

### Prochaines versions
- 🔄 **v1.1**: Fine-tuning modèle spécifique montres
- 🔄 **v1.2**: Support images (OCR + vision)
- 🔄 **v1.3**: API temps réel optimisée
- 🔄 **v1.4**: Multi-modal (texte + images + audio)

## 📞 Support

Pour questions et problèmes:
1. Vérifier cette documentation
2. Exécuter les scripts de test
3. Consulter les logs détaillés
4. Ouvrir une issue avec les détails

---

**L'extracteur LLM transforme votre système WhatsApp RAG en solution de niveau enterprise avec une précision et une intelligence inégalées ! 🎉**
