"""
🕰️ Extracteur d'informations spécialisé pour les ventes de montres
Analyse automatique des messages WhatsApp pour extraire les détails des montres
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WatchInfo:
    """Structure pour les informations extraites d'une montre"""
    brand: Optional[str] = None
    model: Optional[str] = None
    reference: Optional[str] = None
    price: Optional[float] = None
    currency: str = "EUR"
    price_type: Optional[str] = None  # 'asking', 'sold', 'negotiable'
    condition: Optional[str] = None
    year: Optional[int] = None
    size: Optional[str] = None
    movement_type: Optional[str] = None
    authenticity_mentioned: bool = False
    location: Optional[str] = None
    message_type: str = "general"  # 'sale', 'wanted', 'question', 'trade'
    keywords: List[str] = None
    confidence_score: float = 0.0

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

class WatchInfoExtractor:
    """Extracteur intelligent d'informations sur les montres"""
    
    def __init__(self):
        # 🏷️ Marques de montres reconnues
        self.watch_brands = {
            'rolex', 'omega', 'seiko', 'casio', 'citizen', 'tissot', 
            'tag heuer', 'breitling', 'iwc', 'cartier', 'patek philippe',
            'audemars piguet', 'vacheron constantin', 'jaeger-lecoultre',
            'panerai', 'hublot', 'zenith', 'tudor', 'longines', 'hamilton',
            'oris', 'frederique constant', 'mont blanc', 'baume mercier',
            'chopard', 'maurice lacroix', 'mido', 'swatch', 'fossil',
            'diesel', 'armani', 'michael kors', 'daniel wellington',
            'mvmt', 'garmin', 'suunto', 'apple watch', 'samsung gear'
        }
        
        # 💰 Patterns pour détecter les prix
        self.price_patterns = [
            r'(\d{1,6})\s*€',  # Prix en euros
            r'€\s*(\d{1,6})',
            r'(\d{1,6})\s*eur',
            r'(\d{1,6})\s*dollars?',
            r'\$\s*(\d{1,6})',
            r'(\d{1,6})\s*chf',
            r'(\d{1,6})\s*£',
            r'£\s*(\d{1,6})',
            r'prix[:\s]*(\d{1,6})',
            r'(\d{1,6})[,.](\d{2})\s*€',  # Prix avec centimes
        ]
        
        # 🏷️ Patterns pour les conditions
        self.condition_patterns = {
            'neuf': ['neuf', 'new', 'jamais porté', 'unworn', 'bnib', 'brand new'],
            'excellent': ['excellent', 'très bon état', 'excellent condition', 'mint'],
            'bon': ['bon état', 'good condition', 'bel état', 'bien conservé'],
            'occasion': ['occasion', 'used', 'porté', 'worn', 'pre-owned'],
            'vintage': ['vintage', 'ancien', 'collection', 'rare'],
            'box_papers': ['box', 'papers', 'boite', 'papiers', 'certificat', 'garantie']
        }
        
        # 🔧 Patterns pour les tailles
        self.size_patterns = [
            r'(\d{2})\s*mm',
            r'(\d{2})mm',
            r'diamètre[:\s]*(\d{2})',
            r'taille[:\s]*(\d{2})'
        ]
        
        # ⚙️ Types de mouvement
        self.movement_patterns = {
            'automatic': ['automatique', 'automatic', 'auto', 'self-winding'],
            'quartz': ['quartz', 'électronique', 'electronic'],
            'manual': ['manuel', 'manual', 'mécanique', 'mechanical', 'hand-wind']
        }
        
        # 📍 Patterns pour les lieux
        self.location_patterns = [
            r'(?:à|in|from|de)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'livraison[:\s]*([A-Z][a-z]+)',
            r'shipping[:\s]*([A-Z][a-z]+)',
            r'dispo[:\s]*([A-Z][a-z]+)'
        ]

    def extract_watch_info(self, message: str, group_name: str = None) -> WatchInfo:
        """
        Extrait les informations de montre d'un message
        
        Args:
            message: Contenu du message
            group_name: Nom du groupe WhatsApp
            
        Returns:
            WatchInfo: Informations extraites
        """
        message_lower = message.lower()
        info = WatchInfo()
        
        try:
            # 🕰️ Détecter la marque
            info.brand = self._extract_brand(message_lower)
            
            # 🏷️ Détecter le modèle (après la marque)
            info.model = self._extract_model(message, info.brand)
            
            # 💰 Détecter le prix
            price_data = self._extract_price(message)
            info.price = price_data.get('amount')
            info.currency = price_data.get('currency', 'EUR')
            info.price_type = price_data.get('type', 'asking')
            
            # 📊 Détecter la condition
            info.condition = self._extract_condition(message_lower)
            
            # 📏 Détecter la taille
            info.size = self._extract_size(message)
            
            # ⚙️ Détecter le type de mouvement
            info.movement_type = self._extract_movement(message_lower)
            
            # 📅 Détecter l'année
            info.year = self._extract_year(message)
            
            # 📍 Détecter le lieu
            info.location = self._extract_location(message)
            
            # 🔍 Classifier le type de message
            info.message_type = self._classify_message_type(message_lower)
            
            # 🏷️ Extraire les mots-clés
            info.keywords = self._extract_keywords(message_lower)
            
            # 🎯 Calculer le score de confiance
            info.confidence_score = self._calculate_confidence_score(info)
            
            # 📜 Détecter mention d'authenticité
            info.authenticity_mentioned = self._detect_authenticity(message_lower)
            
            logger.info(f"✅ Extraction terminée - Marque: {info.brand}, Prix: {info.price}, Type: {info.message_type}")
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction: {e}")
            
        return info

    def _extract_brand(self, message: str) -> Optional[str]:
        """Extrait la marque de la montre"""
        for brand in self.watch_brands:
            if brand in message:
                # Retourner la marque avec la casse correcte
                return brand.title()
        return None

    def _extract_model(self, message: str, brand: str) -> Optional[str]:
        """Extrait le modèle de la montre"""
        if not brand:
            return None
            
        # Chercher après la marque
        brand_pos = message.lower().find(brand.lower())
        if brand_pos == -1:
            return None
            
        # Prendre les 2-3 mots suivant la marque
        after_brand = message[brand_pos + len(brand):].strip()
        words = after_brand.split()[:3]
        
        # Filtrer les mots non pertinents
        model_words = []
        for word in words:
            if not any(skip in word.lower() for skip in ['prix', 'euro', '€', 'vend', 'cherche']):
                model_words.append(word)
                
        return ' '.join(model_words) if model_words else None

    def _extract_price(self, message: str) -> Dict:
        """Extrait le prix du message"""
        for pattern in self.price_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1).replace(',', '.'))
                    
                    # Détecter la devise
                    currency = 'EUR'
                    if '€' in match.group(0) or 'eur' in match.group(0).lower():
                        currency = 'EUR'
                    elif '$' in match.group(0) or 'dollar' in match.group(0).lower():
                        currency = 'USD'
                    elif '£' in match.group(0):
                        currency = 'GBP'
                    elif 'chf' in match.group(0).lower():
                        currency = 'CHF'
                    
                    # Détecter le type de prix
                    price_type = 'asking'
                    if 'vendu' in message.lower() or 'sold' in message.lower():
                        price_type = 'sold'
                    elif 'négociable' in message.lower() or 'obo' in message.lower():
                        price_type = 'negotiable'
                    
                    return {
                        'amount': price,
                        'currency': currency,
                        'type': price_type
                    }
                except ValueError:
                    continue
                    
        return {'amount': None, 'currency': 'EUR', 'type': 'asking'}

    def _extract_condition(self, message: str) -> Optional[str]:
        """Extrait la condition de la montre"""
        for condition, keywords in self.condition_patterns.items():
            if any(keyword in message for keyword in keywords):
                return condition
        return None

    def _extract_size(self, message: str) -> Optional[str]:
        """Extrait la taille de la montre"""
        for pattern in self.size_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return f"{match.group(1)}mm"
        return None

    def _extract_movement(self, message: str) -> Optional[str]:
        """Extrait le type de mouvement"""
        for movement, keywords in self.movement_patterns.items():
            if any(keyword in message for keyword in keywords):
                return movement
        return None

    def _extract_year(self, message: str) -> Optional[int]:
        """Extrait l'année de fabrication"""
        # Chercher des années entre 1900 et 2030
        year_pattern = r'\b(19\d{2}|20[0-3]\d)\b'
        match = re.search(year_pattern, message)
        if match:
            year = int(match.group(1))
            # Vérifier que c'est plausible pour une montre
            if 1900 <= year <= 2030:
                return year
        return None

    def _extract_location(self, message: str) -> Optional[str]:
        """Extrait le lieu mentionné"""
        for pattern in self.location_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _classify_message_type(self, message: str) -> str:
        """Classifie le type de message"""
        sale_keywords = ['vend', 'vends', 'à vendre', 'for sale', 'sell', 'prix']
        wanted_keywords = ['cherche', 'recherche', 'wanted', 'wtb', 'looking for']
        trade_keywords = ['échange', 'trade', 'swap', 'troc']
        question_keywords = ['?', 'question', 'avis', 'opinion', 'help']
        
        if any(keyword in message for keyword in sale_keywords):
            return 'sale'
        elif any(keyword in message for keyword in wanted_keywords):
            return 'wanted'
        elif any(keyword in message for keyword in trade_keywords):
            return 'trade'
        elif any(keyword in message for keyword in question_keywords):
            return 'question'
        else:
            return 'general'

    def _extract_keywords(self, message: str) -> List[str]:
        """Extrait les mots-clés pertinents"""
        keywords = []
        
        # Mots-clés horlogers
        watch_keywords = [
            'cadran', 'bracelet', 'boitier', 'lunette', 'couronne',
            'dial', 'strap', 'case', 'bezel', 'crown', 'crystal',
            'chrono', 'gmt', 'diver', 'dress', 'sport', 'limited',
            'édition limitée', 'rare', 'collection'
        ]
        
        for keyword in watch_keywords:
            if keyword in message:
                keywords.append(keyword)
                
        return keywords

    def _detect_authenticity(self, message: str) -> bool:
        """Détecte si l'authenticité est mentionnée"""
        auth_keywords = [
            'certificat', 'authentique', 'authentic', 'genuine',
            'papers', 'papiers', 'garantie', 'warranty', 'box'
        ]
        return any(keyword in message for keyword in auth_keywords)

    def _calculate_confidence_score(self, info: WatchInfo) -> float:
        """Calcule un score de confiance pour l'extraction"""
        score = 0.0
        
        # Points pour chaque information trouvée
        if info.brand:
            score += 0.3
        if info.model:
            score += 0.2
        if info.price:
            score += 0.2
        if info.condition:
            score += 0.1
        if info.size:
            score += 0.1
        if info.year:
            score += 0.1
        
        return min(score, 1.0)

# 🧪 Fonction de test
def test_extractor():
    """Teste l'extracteur avec des exemples"""
    extractor = WatchInfoExtractor()
    
    test_messages = [
        "Vends Rolex Submariner 40mm automatique, excellent état, 8500€ avec boite et papiers",
        "Cherche Omega Speedmaster Professional pour collection, budget 3000€ max",
        "À vendre Seiko SKX007 plongée automatique 200m, porté quelques fois, 180€ négociable",
        "Rolex GMT Master II Pepsi 2019 neuf jamais porté 12000€ livraison possible Paris"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n🧪 Test {i}: {message}")
        info = extractor.extract_watch_info(message)
        print(f"   Marque: {info.brand}")
        print(f"   Modèle: {info.model}")
        print(f"   Prix: {info.price} {info.currency}")
        print(f"   Condition: {info.condition}")
        print(f"   Type: {info.message_type}")
        print(f"   Confiance: {info.confidence_score:.2f}")

if __name__ == "__main__":
    test_extractor()
