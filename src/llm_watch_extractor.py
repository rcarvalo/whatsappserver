"""
🤖 Extracteur d'informations de montres basé sur LLM
Utilise OpenAI GPT pour extraire de manière précise et structurée les informations de vente de montres
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import openai
from openai import OpenAI

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMWatchInfo:
    """Structure enrichie pour les informations extraites par LLM"""
    # Informations de base de la montre
    brand: Optional[str] = None
    model: Optional[str] = None
    reference: Optional[str] = None
    collection: Optional[str] = None  # Ex: Submariner, Daytona, Speedmaster
    
    # Informations de prix
    price: Optional[float] = None
    currency: str = "EUR"
    price_type: Optional[str] = None  # 'asking', 'sold', 'negotiable', 'estimate'
    original_price: Optional[float] = None  # Prix d'achat original si mentionné
    
    # État et condition
    condition: Optional[str] = None
    condition_details: Optional[str] = None  # Détails sur l'état
    year: Optional[int] = None
    age_category: Optional[str] = None  # 'new', 'recent', 'vintage', 'antique'
    
    # Caractéristiques techniques
    size: Optional[str] = None
    movement_type: Optional[str] = None  # 'automatic', 'quartz', 'manual'
    material: Optional[str] = None  # 'steel', 'gold', 'platinum', 'ceramic'
    dial_color: Optional[str] = None
    
    # Accessoires et documents
    has_box: Optional[bool] = None
    has_papers: Optional[bool] = None
    has_warranty: Optional[bool] = None
    authenticity_mentioned: bool = False
    accessories_list: List[str] = None
    
    # Informations de vente
    seller_type: Optional[str] = None  # 'private', 'dealer', 'boutique'
    location: Optional[str] = None
    shipping_available: Optional[bool] = None
    shipping_details: Optional[str] = None
    
    # Classification du message
    message_type: str = 'general'  # 'sale', 'wanted', 'question', 'price_check', 'trade'
    urgency_level: int = 0  # 0-5
    negotiable: Optional[bool] = None
    
    # Sentiments et intentions
    seller_motivation: Optional[str] = None  # 'urgent', 'flexible', 'firm'
    investment_mention: Optional[bool] = None
    
    # Métadonnées d'extraction
    confidence_score: float = 0.0
    extraction_method: str = "llm"
    extracted_text_segments: List[str] = None
    llm_reasoning: Optional[str] = None
    
    def __post_init__(self):
        if self.accessories_list is None:
            self.accessories_list = []
        if self.extracted_text_segments is None:
            self.extracted_text_segments = []

class LLMWatchExtractor:
    """Extracteur d'informations de montres utilisant un LLM pour une précision maximale"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialise l'extracteur LLM
        
        Args:
            openai_api_key: Clé API OpenAI
            model: Modèle à utiliser (gpt-4o-mini recommandé pour le rapport qualité/prix)
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Cache pour éviter les appels répétés sur le même texte
        self._extraction_cache = {}
        
    def extract_watch_info(self, message_content: str, whatsapp_metadata: Dict = None) -> LLMWatchInfo:
        """
        Extrait les informations de montre depuis un message WhatsApp en utilisant un LLM
        
        Args:
            message_content: Contenu du message à analyser
            whatsapp_metadata: Métadonnées WhatsApp pour le contexte
            
        Returns:
            LLMWatchInfo avec toutes les informations extraites
        """
        try:
            # Vérifier le cache
            cache_key = self._generate_cache_key(message_content, whatsapp_metadata)
            if cache_key in self._extraction_cache:
                self.logger.debug("Résultat trouvé dans le cache")
                return self._extraction_cache[cache_key]
            
            # Créer le prompt d'extraction structuré
            extraction_prompt = self._create_extraction_prompt(message_content, whatsapp_metadata)
            
            # Appel au LLM avec structured output
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": extraction_prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Faible température pour la cohérence
                max_tokens=2000
            )
            
            # Parser la réponse JSON
            extraction_result = json.loads(response.choices[0].message.content)
            
            # Convertir en LLMWatchInfo
            watch_info = self._convert_llm_response_to_watch_info(extraction_result, message_content)
            
            # Mettre en cache
            self._extraction_cache[cache_key] = watch_info
            
            self.logger.info(f"Extraction LLM réussie: {watch_info.brand} {watch_info.model} - Confiance: {watch_info.confidence_score:.2f}")
            
            return watch_info
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction LLM: {e}")
            # Retourner un objet vide plutôt qu'une exception
            return LLMWatchInfo(
                confidence_score=0.0,
                llm_reasoning=f"Erreur d'extraction: {str(e)}"
            )
    
    def _get_system_prompt(self) -> str:
        """Retourne le prompt système pour l'extraction de montres"""
        return """Tu es un expert en horlogerie et ventes de montres de luxe. Ton rôle est d'extraire de manière précise et structurée toutes les informations pertinentes sur les montres depuis des messages WhatsApp en français.

CONTEXTE:
- Messages de groupes/particuliers vendant/cherchant des montres
- Souvent des montres de luxe (Rolex, Omega, Patek Philippe, etc.)
- Informations dispersées dans le texte, parfois avec fautes d'orthographe
- Abréviations et jargon horloger fréquents

INSTRUCTIONS:
1. Extrais TOUTES les informations disponibles sur la montre
2. Détermine le type de message (vente, recherche, question, etc.)
3. Évalue le niveau de confiance de tes extractions
4. Fournis un raisonnement sur tes choix
5. Réponds UNIQUEMENT en JSON valide

EXPERTISE REQUISE:
- Reconnaissance des références exactes (ex: 116610LV, 311.30.42.30)
- Surnoms populaires (Hulk, Panda, Speedmaster, etc.)
- Codes couleurs et matériaux
- Prix de marché approximatifs
- Accessoires standards (box, papers, warranty)"""

    def _create_extraction_prompt(self, message_content: str, whatsapp_metadata: Dict = None) -> str:
        """Crée le prompt d'extraction pour le message donné"""
        
        # Ajouter le contexte WhatsApp si disponible
        context_info = ""
        if whatsapp_metadata:
            context_info = f"""
CONTEXTE WHATSAPP:
- Expéditeur: {whatsapp_metadata.get('sender_profile_name', 'Inconnu')}
- Groupe: {'Oui' if whatsapp_metadata.get('is_group_message') else 'Non'}
- Intentions détectées: {whatsapp_metadata.get('semantic_metadata', {}).get('intent_signals', {})}
"""

        prompt = f"""Analyse ce message WhatsApp et extrais toutes les informations sur la montre:

MESSAGE À ANALYSER:
{message_content}

{context_info}

Réponds en JSON avec cette structure exacte:
{{
    "watch_details": {{
        "brand": "marque exacte ou null",
        "model": "modèle complet ou null", 
        "reference": "référence technique ou null",
        "collection": "collection/ligne (ex: Submariner) ou null",
        "price": valeur_numérique_ou_null,
        "currency": "devise (EUR/USD/CHF)",
        "price_type": "asking/sold/negotiable/estimate ou null",
        "condition": "état de la montre ou null",
        "condition_details": "détails sur l'état ou null",
        "year": année_ou_null,
        "size": "taille (ex: 40mm) ou null",
        "movement_type": "automatic/quartz/manual ou null",
        "material": "matériau principal ou null",
        "dial_color": "couleur du cadran ou null"
    }},
    "accessories": {{
        "has_box": true/false/null,
        "has_papers": true/false/null,
        "has_warranty": true/false/null,
        "authenticity_mentioned": true/false,
        "accessories_list": ["liste", "des", "accessoires"]
    }},
    "sale_info": {{
        "message_type": "sale/wanted/question/price_check/trade/general",
        "seller_type": "private/dealer/boutique ou null",
        "location": "lieu mentionné ou null",
        "shipping_available": true/false/null,
        "urgency_level": 0-5,
        "negotiable": true/false/null,
        "seller_motivation": "urgent/flexible/firm ou null"
    }},
    "extraction_metadata": {{
        "confidence_score": 0.0-1.0,
        "extracted_segments": ["segments", "de", "texte", "utilisés"],
        "reasoning": "explication de ton raisonnement et choix"
    }}
}}

RÈGLES IMPORTANTES:
- Si une information n'est pas claire, utilise null
- Pour les prix, extrait seulement les nombres (sans €, EUR, etc.)
- Pour message_type: "sale" si vente, "wanted" si recherche, "question" si demande d'info
- confidence_score: 0.8+ si très sûr, 0.5-0.8 si probable, <0.5 si incertain
- reasoning: explique pourquoi tu as fait ces choix
"""
        
        return prompt
    
    def _convert_llm_response_to_watch_info(self, llm_response: Dict, original_message: str) -> LLMWatchInfo:
        """Convertit la réponse LLM en objet LLMWatchInfo"""
        
        try:
            watch_details = llm_response.get('watch_details', {})
            accessories = llm_response.get('accessories', {})
            sale_info = llm_response.get('sale_info', {})
            metadata = llm_response.get('extraction_metadata', {})
            
            return LLMWatchInfo(
                # Informations de base
                brand=watch_details.get('brand'),
                model=watch_details.get('model'),
                reference=watch_details.get('reference'),
                collection=watch_details.get('collection'),
                
                # Prix
                price=watch_details.get('price'),
                currency=watch_details.get('currency', 'EUR'),
                price_type=watch_details.get('price_type'),
                
                # État
                condition=watch_details.get('condition'),
                condition_details=watch_details.get('condition_details'),
                year=watch_details.get('year'),
                
                # Caractéristiques
                size=watch_details.get('size'),
                movement_type=watch_details.get('movement_type'),
                material=watch_details.get('material'),
                dial_color=watch_details.get('dial_color'),
                
                # Accessoires
                has_box=accessories.get('has_box'),
                has_papers=accessories.get('has_papers'),
                has_warranty=accessories.get('has_warranty'),
                authenticity_mentioned=accessories.get('authenticity_mentioned', False),
                accessories_list=accessories.get('accessories_list', []),
                
                # Vente
                message_type=sale_info.get('message_type', 'general'),
                seller_type=sale_info.get('seller_type'),
                location=sale_info.get('location'),
                shipping_available=sale_info.get('shipping_available'),
                urgency_level=sale_info.get('urgency_level', 0),
                negotiable=sale_info.get('negotiable'),
                seller_motivation=sale_info.get('seller_motivation'),
                
                # Métadonnées
                confidence_score=metadata.get('confidence_score', 0.0),
                extraction_method="llm",
                extracted_text_segments=metadata.get('extracted_segments', []),
                llm_reasoning=metadata.get('reasoning')
            )
            
        except Exception as e:
            self.logger.error(f"Erreur conversion réponse LLM: {e}")
            return LLMWatchInfo(
                confidence_score=0.0,
                llm_reasoning=f"Erreur de conversion: {str(e)}"
            )
    
    def _generate_cache_key(self, message_content: str, whatsapp_metadata: Dict = None) -> str:
        """Génère une clé de cache pour éviter les appels répétés"""
        import hashlib
        
        # Créer un hash du contenu + métadonnées importantes
        cache_content = message_content
        if whatsapp_metadata:
            # Ajouter seulement les métadonnées qui influencent l'extraction
            cache_content += str(whatsapp_metadata.get('sender_profile_name', ''))
            cache_content += str(whatsapp_metadata.get('is_group_message', False))
        
        return hashlib.md5(cache_content.encode()).hexdigest()
    
    def extract_batch(self, messages: List[Dict]) -> List[LLMWatchInfo]:
        """
        Extrait les informations de plusieurs messages en batch pour optimiser les coûts
        
        Args:
            messages: Liste de messages avec 'content' et optionnellement 'metadata'
            
        Returns:
            Liste des LLMWatchInfo extraites
        """
        results = []
        
        for message in messages:
            content = message.get('content', '')
            metadata = message.get('metadata', {})
            
            if not content.strip():
                results.append(LLMWatchInfo(confidence_score=0.0))
                continue
            
            try:
                watch_info = self.extract_watch_info(content, metadata)
                results.append(watch_info)
            except Exception as e:
                self.logger.error(f"Erreur extraction batch: {e}")
                results.append(LLMWatchInfo(
                    confidence_score=0.0,
                    llm_reasoning=f"Erreur batch: {str(e)}"
                ))
        
        return results
    
    def get_extraction_stats(self) -> Dict:
        """Retourne des statistiques sur les extractions effectuées"""
        total_extractions = len(self._extraction_cache)
        
        if total_extractions == 0:
            return {"total_extractions": 0}
        
        # Analyser les résultats en cache
        confidence_scores = [info.confidence_score for info in self._extraction_cache.values()]
        message_types = [info.message_type for info in self._extraction_cache.values()]
        brands = [info.brand for info in self._extraction_cache.values() if info.brand]
        
        return {
            "total_extractions": total_extractions,
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "high_confidence_rate": len([s for s in confidence_scores if s > 0.7]) / len(confidence_scores) if confidence_scores else 0,
            "message_types_distribution": {mt: message_types.count(mt) for mt in set(message_types)},
            "top_brands": list(set(brands)),
            "cache_hit_rate": "N/A"  # Nécessiterait un tracking plus sophistiqué
        }
    
    def clear_cache(self):
        """Vide le cache d'extraction"""
        self._extraction_cache.clear()
        self.logger.info("Cache d'extraction vidé")

# Fonction de compatibilité pour remplacer l'ancien extracteur
def create_llm_extractor(openai_api_key: str) -> LLMWatchExtractor:
    """Crée une instance de l'extracteur LLM"""
    return LLMWatchExtractor(openai_api_key=openai_api_key)
