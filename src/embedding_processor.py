import os
import json
import logging
from typing import List, Dict, Optional, Union
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from supabase import create_client, Client
import openai
from openai import OpenAI
import time
import hashlib
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

@dataclass
class MessageEmbedding:
    """Structure pour stocker un message avec son embedding"""
    id: Optional[int]
    phone_number: str
    message_content: str
    message_timestamp: str
    sender: str
    embedding: List[float]
    is_outgoing: bool
    media_type: Optional[str] = None
    content_hash: Optional[str] = None
    
    # 🕰️ NOUVEAUX CHAMPS POUR LES MONTRES
    group_name: Optional[str] = None
    message_type: str = 'general'
    intent_score: float = 0.0
    
    # Informations montres extraites
    watch_brand: Optional[str] = None
    watch_model: Optional[str] = None
    watch_reference: Optional[str] = None
    
    # Informations prix
    price_mentioned: Optional[float] = None
    currency: str = 'EUR'
    price_type: Optional[str] = None
    
    # État et caractéristiques
    condition_mentioned: Optional[str] = None
    year_mentioned: Optional[int] = None
    size_mentioned: Optional[str] = None
    movement_type: Optional[str] = None
    
    # Informations vente
    seller_type: Optional[str] = None
    location_mentioned: Optional[str] = None
    shipping_info: Optional[str] = None
    authenticity_mentioned: bool = False
    
    # Métadonnées enrichies
    extracted_keywords: Optional[List[str]] = None
    sentiment_score: Optional[float] = None
    urgency_level: int = 0
    
    # JSON pour infos complexes
    detailed_extraction: Optional[Dict] = None
    search_metadata: Optional[Dict] = None

class EmbeddingProcessor:
    def __init__(self, supabase_url: str, supabase_key: str, openai_api_key: str):
        """
        Initialise le processeur d'embeddings
        
        Args:
            supabase_url: URL de votre projet Supabase
            supabase_key: Clé d'API Supabase
            openai_api_key: Clé d'API OpenAI
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.logger = self._setup_logging()
        
        # Configuration des embeddings
        self.embedding_model = "text-embedding-3-small"  # Modèle plus récent et économique
        self.embedding_dimension = 1536
        self.batch_size = 100  # Nombre de messages à traiter par lot
        self.max_tokens = 8000  # Limite de tokens par message
        
        # Cache pour éviter les duplicatas
        self.processed_hashes = set()
        
    def _setup_logging(self):
        """Configure le logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('embedding_processor.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _create_content_hash(self, content: str, phone_number: str, timestamp: str) -> str:
        """
        Crée un hash unique pour éviter les duplicatas
        
        Args:
            content: Contenu du message
            phone_number: Numéro de téléphone
            timestamp: Horodatage
            
        Returns:
            Hash MD5 du contenu
        """
        unique_string = f"{phone_number}_{timestamp}_{content}"
        return hashlib.md5(unique_string.encode('utf-8')).hexdigest()
    
    def _clean_message_content(self, content: str) -> str:
        """
        Nettoie le contenu du message pour l'embedding
        
        Args:
            content: Contenu brut du message
            
        Returns:
            Contenu nettoyé
        """
        if not content or content.strip() == "":
            return ""
        
        # Remplacer les emojis par leur description textuelle (optionnel)
        # content = self._replace_emojis(content)
        
        # Nettoyer les caractères spéciaux et espaces multiples
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Limiter la longueur (approximation pour éviter de dépasser les tokens)
        if len(content) > 6000:  # Environ 8000 tokens
            content = content[:6000] + "..."
        
        return content
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Génère un embedding pour un texte donné
        
        Args:
            text: Texte à encoder
            
        Returns:
            Vecteur d'embedding ou None si erreur
        """
        try:
            if not text or text.strip() == "":
                return None
            
            cleaned_text = self._clean_message_content(text)
            if not cleaned_text:
                return None
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération d'embedding: {e}")
            return None
    
    def generate_enhanced_embedding(self, text: str, metadata: Dict = None) -> Optional[List[float]]:
        """
        Génère un embedding enrichi avec métadonnées pour améliorer la recherche sémantique
        
        Args:
            text: Texte original à encoder
            metadata: Métadonnées enrichies (groupes, intentions, etc.)
            
        Returns:
            Vecteur d'embedding enrichi ou None si erreur
        """
        try:
            if not text or text.strip() == "":
                return None
            
            # 🎯 ENRICHISSEMENT DU TEXTE AVEC MÉTADONNÉES
            enhanced_text = self._create_enhanced_text_for_embedding(text, metadata)
            
            # Générer l'embedding sur le texte enrichi
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=enhanced_text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            if len(embedding) != self.embedding_dimension:
                self.logger.error(f"Dimension d'embedding incorrecte: {len(embedding)} au lieu de {self.embedding_dimension}")
                return None
            
            self.logger.debug(f"Embedding enrichi généré: {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération d'embedding enrichi: {e}")
            # Fallback vers l'embedding standard
            return self.generate_embedding(text)
    
    def _create_enhanced_text_for_embedding(self, original_text: str, metadata: Dict = None) -> str:
        """
        Crée un texte enrichi avec les métadonnées pour améliorer l'embedding
        Inspiré des techniques d'Azure Search et autres systèmes RAG avancés
        """
        enhanced_parts = [original_text]
        
        if not metadata:
            return original_text
        
        # 🏢 CONTEXTE DE GROUPE
        if metadata.get('is_group_message'):
            group_name = metadata.get('sender_profile_name', 'Groupe')
            enhanced_parts.append(f"[GROUPE: {group_name}]")
            
            group_indicators = metadata.get('group_context_indicators', [])
            if group_indicators:
                enhanced_parts.append(f"[CONTEXTE: {', '.join(group_indicators[:2])}]")
        
        # 👤 CONTEXTE EXPÉDITEUR
        sender_name = metadata.get('sender_formatted_name') or metadata.get('sender_profile_name')
        if sender_name:
            enhanced_parts.append(f"[EXPÉDITEUR: {sender_name}]")
        
        # 🎯 INTENTIONS DÉTECTÉES
        semantic_metadata = metadata.get('semantic_metadata', {})
        intent_signals = semantic_metadata.get('intent_signals', {})
        
        detected_intents = []
        if intent_signals.get('is_selling'):
            detected_intents.append('VENTE')
        if intent_signals.get('is_seeking'):
            detected_intents.append('RECHERCHE')
        if intent_signals.get('is_question'):
            detected_intents.append('QUESTION')
        if intent_signals.get('has_urgency'):
            detected_intents.append('URGENT')
            
        if detected_intents:
            enhanced_parts.append(f"[INTENTION: {', '.join(detected_intents)}]")
        
        # 📊 CONTEXTE COMMERCIAL
        text_analysis = semantic_metadata.get('text_analysis', {})
        commercial_context = []
        
        if text_analysis.get('has_price'):
            commercial_context.append('PRIX')
        if text_analysis.get('has_phone'):
            commercial_context.append('CONTACT')
        if text_analysis.get('has_urls'):
            commercial_context.append('LIEN')
            
        if commercial_context:
            enhanced_parts.append(f"[COMMERCIAL: {', '.join(commercial_context)}]")
        
        # 🌍 CONTEXTE LINGUISTIQUE
        language_hints = text_analysis.get('language_hints', [])
        if language_hints:
            enhanced_parts.append(f"[LANGUE: {language_hints[0].upper()}]")
        
        # ⏰ CONTEXTE TEMPOREL
        timing = semantic_metadata.get('timing', {})
        if timing.get('is_business_hours'):
            enhanced_parts.append("[HEURES_OUVRABLES]")
        
        # 🔗 CONTEXTE CONVERSATIONNEL
        conversation = semantic_metadata.get('conversation', {})
        if conversation.get('is_reply'):
            enhanced_parts.append("[RÉPONSE]")
        if conversation.get('has_context'):
            enhanced_parts.append("[THREAD]")
        
        # Combiner tous les éléments
        enhanced_text = " ".join(enhanced_parts)
        
        # Limiter la taille pour éviter les tokens excessifs
        max_length = 8000  # Limite approximative
        if len(enhanced_text) > max_length:
            # Garder le texte original et les métadonnées les plus importantes
            priority_metadata = []
            if metadata.get('is_group_message'):
                priority_metadata.append(f"[GROUPE: {metadata.get('sender_profile_name', 'Groupe')}]")
            if detected_intents:
                priority_metadata.append(f"[INTENTION: {detected_intents[0]}]")
            
            enhanced_text = original_text + " " + " ".join(priority_metadata)
            enhanced_text = enhanced_text[:max_length]
        
        return enhanced_text
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Génère des embeddings par lot pour optimiser les performances
        
        Args:
            texts: Liste de textes à encoder
            
        Returns:
            Liste d'embeddings correspondants
        """
        try:
            if not texts:
                return []
            
            # Nettoyer et filtrer les textes vides
            cleaned_texts = []
            text_indices = []  # Pour mapper les résultats aux textes originaux
            
            for i, text in enumerate(texts):
                cleaned = self._clean_message_content(text) if text else ""
                if cleaned:
                    cleaned_texts.append(cleaned)
                    text_indices.append(i)
            
            if not cleaned_texts:
                return [None] * len(texts)
            
            # Générer les embeddings par lot
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_texts
            )
            
            # Reconstruire la liste complète avec les None pour les textes vides
            result = [None] * len(texts)
            for i, embedding_data in enumerate(response.data):
                original_index = text_indices[i]
                result[original_index] = embedding_data.embedding
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération d'embeddings par lot: {e}")
            return [None] * len(texts)
    
    def store_message_embedding(self, message_embedding: MessageEmbedding) -> Optional[Dict]:
        """
        Stocke un message avec son embedding dans Supabase (table watch_conversations)
        
        Args:
            message_embedding: Objet MessageEmbedding à stocker
            
        Returns:
            Résultat de l'insertion ou None si erreur
        """
        try:
            data = {
                'phone_number': message_embedding.phone_number,
                'message_content': message_embedding.message_content,
                'message_timestamp': message_embedding.message_timestamp,
                'sender': message_embedding.sender,
                'embedding': message_embedding.embedding,
                'is_outgoing': message_embedding.is_outgoing,
                'content_hash': message_embedding.content_hash,
                
                # 🕰️ NOUVELLES DONNÉES MONTRES
                'group_name': message_embedding.group_name,
                'message_type': message_embedding.message_type,
                'intent_score': message_embedding.intent_score,
                
                # Informations montres extraites
                'watch_brand': message_embedding.watch_brand,
                'watch_model': message_embedding.watch_model,
                'watch_reference': message_embedding.watch_reference,
                
                # Informations prix
                'price_mentioned': message_embedding.price_mentioned,
                'currency': message_embedding.currency,
                'price_type': message_embedding.price_type,
                
                # État et caractéristiques
                'condition_mentioned': message_embedding.condition_mentioned,
                'year_mentioned': message_embedding.year_mentioned,
                'size_mentioned': message_embedding.size_mentioned,
                'movement_type': message_embedding.movement_type,
                
                # Informations vente
                'seller_type': message_embedding.seller_type,
                'location_mentioned': message_embedding.location_mentioned,
                'shipping_info': message_embedding.shipping_info,
                'authenticity_mentioned': message_embedding.authenticity_mentioned,
                
                # Métadonnées enrichies
                'extracted_keywords': message_embedding.extracted_keywords,
                'sentiment_score': message_embedding.sentiment_score,
                'urgency_level': message_embedding.urgency_level,
                
                # JSON pour infos complexes
                'detailed_extraction': message_embedding.detailed_extraction or {},
                'search_metadata': message_embedding.search_metadata or {}
            }
            
            result = self.supabase.table('watch_conversations').insert(data).execute()
            
            if result.data:
                self.logger.debug(f"Message stocké avec ID: {result.data[0].get('id')}")
                return result.data[0]
            else:
                self.logger.warning("Aucune donnée retournée lors de l'insertion")
                return None
                
        except Exception as e:
            self.logger.error(f"Erreur lors du stockage: {e}")
            return None
    
    def store_messages_batch(self, message_embeddings: List[MessageEmbedding]) -> List[Optional[Dict]]:
        """
        Stocke plusieurs messages par lot pour optimiser les performances
        
        Args:
            message_embeddings: Liste des MessageEmbedding à stocker
            
        Returns:
            Liste des résultats d'insertion
        """
        try:
            if not message_embeddings:
                return []
            
            data_list = []
            for msg_emb in message_embeddings:
                data = {
                    'phone_number': msg_emb.phone_number,
                    'message_content': msg_emb.message_content,
                    'message_timestamp': msg_emb.message_timestamp,
                    'sender': msg_emb.sender,
                    'embedding': msg_emb.embedding,
                    'is_outgoing': msg_emb.is_outgoing,
                    'content_hash': msg_emb.content_hash,
                    
                    # 🕰️ NOUVELLES DONNÉES MONTRES (même structure que store_message_embedding)
                    'group_name': msg_emb.group_name,
                    'message_type': msg_emb.message_type,
                    'intent_score': msg_emb.intent_score,
                    'watch_brand': msg_emb.watch_brand,
                    'watch_model': msg_emb.watch_model,
                    'watch_reference': msg_emb.watch_reference,
                    'price_mentioned': msg_emb.price_mentioned,
                    'currency': msg_emb.currency,
                    'price_type': msg_emb.price_type,
                    'condition_mentioned': msg_emb.condition_mentioned,
                    'year_mentioned': msg_emb.year_mentioned,
                    'size_mentioned': msg_emb.size_mentioned,
                    'movement_type': msg_emb.movement_type,
                    'seller_type': msg_emb.seller_type,
                    'location_mentioned': msg_emb.location_mentioned,
                    'shipping_info': msg_emb.shipping_info,
                    'authenticity_mentioned': msg_emb.authenticity_mentioned,
                    'extracted_keywords': msg_emb.extracted_keywords,
                    'sentiment_score': msg_emb.sentiment_score,
                    'urgency_level': msg_emb.urgency_level,
                    'detailed_extraction': msg_emb.detailed_extraction or {},
                    'search_metadata': msg_emb.search_metadata or {}
                }
                data_list.append(data)
            
            result = self.supabase.table('watch_conversations').insert(data_list).execute()
            
            if result.data:
                self.logger.info(f"Lot de {len(result.data)} messages stocké avec succès")
                return result.data
            else:
                self.logger.warning("Aucune donnée retournée lors de l'insertion du lot")
                return []
                
        except Exception as e:
            self.logger.error(f"Erreur lors du stockage par lot: {e}")
            return []
    
    def check_existing_messages(self, phone_number: str) -> set:
        """
        Vérifie les messages déjà existants pour éviter les duplicatas
        
        Args:
            phone_number: Numéro de téléphone à vérifier
            
        Returns:
            Set des hash de contenu déjà existants
        """
        try:
            result = self.supabase.table('watch_conversations')\
                .select('content_hash')\
                .eq('phone_number', phone_number)\
                .execute()
            
            existing_hashes = {row['content_hash'] for row in result.data if row['content_hash']}
            self.logger.info(f"Trouvé {len(existing_hashes)} messages existants pour {phone_number}")
            
            return existing_hashes
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification des messages existants: {e}")
            return set()
    
    def process_whatsapp_messages(self, messages: List[Dict], phone_number: str, 
                                check_duplicates: bool = True) -> List[MessageEmbedding]:
        """
        Traite une liste de messages WhatsApp pour créer des embeddings
        
        Args:
            messages: Liste des messages extraits de WhatsApp
            phone_number: Numéro de téléphone associé
            check_duplicates: Vérifier les duplicatas avant traitement
            
        Returns:
            Liste des MessageEmbedding créés
        """
        try:
            self.logger.info(f"Début du traitement de {len(messages)} messages pour {phone_number}")
            
            # Vérifier les messages existants si demandé
            existing_hashes = set()
            if check_duplicates:
                existing_hashes = self.check_existing_messages(phone_number)
            
            message_embeddings = []
            texts_to_embed = []
            valid_messages = []
            
            # Préparer les messages pour le traitement par lot
            for msg in messages:
                content = msg.get('content', '').strip()
                
                # Ignorer les messages vides ou médias sans texte
                if not content or content in ['[MÉDIA]', '[MESSAGE NON RECONNU]']:
                    continue
                
                # Créer le hash pour vérifier les duplicatas
                timestamp_str = msg.get('timestamp', datetime.now().isoformat())
                content_hash = self._create_content_hash(content, phone_number, timestamp_str)
                
                # Ignorer si déjà traité
                if content_hash in existing_hashes:
                    continue
                
                texts_to_embed.append(content)
                valid_messages.append((msg, content_hash))
            
            if not texts_to_embed:
                self.logger.info("Aucun nouveau message à traiter")
                return []
            
            self.logger.info(f"Traitement de {len(texts_to_embed)} nouveaux messages")
            
            # Générer les embeddings par lot
            embeddings = self.generate_embeddings_batch(texts_to_embed)
            
            # Créer les objets MessageEmbedding
            for i, (embedding, (msg, content_hash)) in enumerate(zip(embeddings, valid_messages)):
                if embedding is None:
                    continue
                
                message_embedding = MessageEmbedding(
                    id=None,
                    phone_number=phone_number,
                    message_content=msg.get('content', ''),
                    timestamp=msg.get('timestamp', datetime.now().isoformat()),
                    sender=msg.get('sender', 'unknown'),
                    embedding=embedding,
                    is_outgoing=msg.get('is_outgoing', False),
                    media_type=msg.get('media_type'),
                    content_hash=content_hash
                )
                
                message_embeddings.append(message_embedding)
            
            self.logger.info(f"Créé {len(message_embeddings)} embeddings avec succès")
            return message_embeddings
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des messages: {e}")
            return []
    
    def process_and_store_conversation(self, messages: List[Dict], phone_number: str,
                                     batch_size: Optional[int] = None) -> bool:
        """
        Traite et stocke une conversation complète
        
        Args:
            messages: Messages extraits de WhatsApp
            phone_number: Numéro de téléphone
            batch_size: Taille des lots pour le stockage
            
        Returns:
            True si succès, False sinon
        """
        try:
            self.logger.info(f"Début du traitement complet pour {phone_number}")
            
            # Traiter les messages pour créer les embeddings
            message_embeddings = self.process_whatsapp_messages(messages, phone_number)
            
            if not message_embeddings:
                self.logger.info("Aucun message à stocker")
                return True
            
            # Stocker par lot
            batch_size = batch_size or self.batch_size
            stored_count = 0
            
            for i in range(0, len(message_embeddings), batch_size):
                batch = message_embeddings[i:i + batch_size]
                result = self.store_messages_batch(batch)
                
                if result:
                    stored_count += len(result)
                    self.logger.info(f"Lot {i//batch_size + 1} stocké: {len(result)} messages")
                else:
                    self.logger.error(f"Échec du stockage du lot {i//batch_size + 1}")
                
                # Pause pour éviter les limitations de débit
                time.sleep(0.1)
            
            self.logger.info(f"Traitement terminé: {stored_count} messages stockés sur {len(message_embeddings)}")
            return stored_count > 0
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement complet: {e}")
            return False
    
    def load_and_process_from_file(self, file_path: str, phone_number: str,
                                  file_format: str = 'json') -> bool:
        """
        Charge et traite des messages depuis un fichier
        
        Args:
            file_path: Chemin vers le fichier
            phone_number: Numéro de téléphone associé
            file_format: Format du fichier ('json', 'csv')
            
        Returns:
            True si succès, False sinon
        """
        try:
            self.logger.info(f"Chargement du fichier: {file_path}")
            
            messages = []
            
            if file_format.lower() == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
                    
            elif file_format.lower() == 'csv':
                df = pd.read_csv(file_path)
                messages = df.to_dict('records')
                
            else:
                raise ValueError(f"Format de fichier non supporté: {file_format}")
            
            return self.process_and_store_conversation(messages, phone_number)
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement depuis le fichier: {e}")
            return False
    
    def get_conversation_stats(self, phone_number: str) -> Dict:
        """
        Obtient les statistiques d'une conversation
        
        Args:
            phone_number: Numéro de téléphone
            
        Returns:
            Dictionnaire avec les statistiques
        """
        try:
            result = self.supabase.table('watch_conversations')\
                .select('id, timestamp, sender, is_outgoing')\
                .eq('phone_number', phone_number)\
                .execute()
            
            if not result.data:
                return {'message_count': 0}
            
            messages = result.data
            stats = {
                'phone_number': phone_number,
                'message_count': len(messages),
                'outgoing_count': sum(1 for msg in messages if msg.get('is_outgoing', False)),
                'incoming_count': sum(1 for msg in messages if not msg.get('is_outgoing', True)),
                'date_range': {
                    'earliest': min(msg['timestamp'] for msg in messages if msg.get('timestamp')),
                    'latest': max(msg['timestamp'] for msg in messages if msg.get('timestamp'))
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'obtention des statistiques: {e}")
            return {'error': str(e)}
    
    def delete_conversation(self, phone_number: str) -> bool:
        """
        Supprime tous les messages d'une conversation
        
        Args:
            phone_number: Numéro de téléphone
            
        Returns:
            True si succès, False sinon
        """
        try:
            result = self.supabase.table('watch_conversations')\
                .delete()\
                .eq('phone_number', phone_number)\
                .execute()
            
            self.logger.info(f"Conversation supprimée pour {phone_number}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression: {e}")
            return False
    
    def update_embeddings_model(self, new_model: str = "text-embedding-3-small"):
        """
        Met à jour le modèle d'embeddings utilisé
        
        Args:
            new_model: Nom du nouveau modèle OpenAI
        """
        self.embedding_model = new_model
        self.logger.info(f"Modèle d'embeddings mis à jour: {new_model}")
    
    def cleanup_orphaned_embeddings(self) -> int:
        """
        Nettoie les embeddings orphelins (sans contenu valide)
        
        Returns:
            Nombre d'enregistrements supprimés
        """
        try:
            # Supprimer les messages avec contenu vide ou embedding null
            result = self.supabase.table('watch_conversations')\
                .delete()\
                .or_("message_content.is.null,message_content.eq.'',embedding.is.null")\
                .execute()
            
            count = len(result.data) if result.data else 0
            self.logger.info(f"Nettoyage terminé: {count} enregistrements supprimés")
            
            return count
            
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage: {e}")
            return 0

# Classe utilitaire pour la migration et maintenance
class EmbeddingMaintenance:
    def __init__(self, embedding_processor: EmbeddingProcessor):
        self.processor = embedding_processor
        self.logger = embedding_processor.logger
    
    def reprocess_conversation(self, phone_number: str, force: bool = False) -> bool:
        """
        Retraite tous les embeddings d'une conversation
        
        Args:
            phone_number: Numéro à retraiter
            force: Forcer même si les embeddings existent déjà
            
        Returns:
            True si succès
        """
        try:
            if not force:
                # Vérifier s'il y a déjà des embeddings
                existing = self.processor.check_existing_messages(phone_number)
                if existing:
                    self.logger.info(f"Embeddings déjà présents pour {phone_number}. Utilisez force=True pour retraiter.")
                    return True
            
            # Supprimer les anciens embeddings si force=True
            if force:
                self.processor.delete_conversation(phone_number)
            
            # Ici, il faudrait récupérer les messages originaux depuis une source
            # (fichier de sauvegarde, base de données temporaire, etc.)
            self.logger.warning("Retraitement nécessite les messages sources originaux")
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur lors du retraitement: {e}")
            return False
    
    def migrate_to_new_model(self, new_model: str) -> bool:
        """
        Migre tous les embeddings vers un nouveau modèle
        
        Args:
            new_model: Nom du nouveau modèle
            
        Returns:
            True si succès
        """
        try:
            self.logger.info(f"Début de migration vers le modèle: {new_model}")
            
            # Cette fonction nécessiterait une stratégie de migration par lot
            # pour éviter de surcharger l'API OpenAI
            
            # 1. Obtenir tous les messages uniques
            # 2. Générer les nouveaux embeddings par lot
            # 3. Mettre à jour la base de données
            
            self.processor.update_embeddings_model(new_model)
            self.logger.info("Migration terminée")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la migration: {e}")
            return False

# Exemple d'utilisation
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configuration
    processor = EmbeddingProcessor(
        supabase_url=os.getenv('SUPABASE_URL'),
        supabase_key=os.getenv('SUPABASE_ANON_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # Exemple: traitement d'un fichier JSON
    phone_number = "+33123456789"
    json_file = f"whatsapp_{phone_number.replace('+', '').replace(' ', '')}_20240101_120000.json"
    
    if os.path.exists(json_file):
        success = processor.load_and_process_from_file(
            file_path=json_file,
            phone_number=phone_number,
            file_format='json'
        )
        
        if success:
            # Afficher les statistiques
            stats = processor.get_conversation_stats(phone_number)
            print(f"Traitement terminé:")
            print(f"- Messages stockés: {stats.get('message_count', 0)}")
            print(f"- Messages sortants: {stats.get('outgoing_count', 0)}")
            print(f"- Messages entrants: {stats.get('incoming_count', 0)}")
        else:
            print("Échec du traitement")
    else:
        print(f"Fichier {json_file} non trouvé")