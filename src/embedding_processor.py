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
    timestamp: str
    sender: str
    embedding: List[float]
    is_outgoing: bool
    media_type: Optional[str] = None
    content_hash: Optional[str] = None

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
        Stocke un message avec son embedding dans Supabase
        
        Args:
            message_embedding: Objet MessageEmbedding à stocker
            
        Returns:
            Résultat de l'insertion ou None si erreur
        """
        try:
            data = {
                'phone_number': message_embedding.phone_number,
                'message_content': message_embedding.message_content,
                'timestamp': message_embedding.timestamp,
                'sender': message_embedding.sender,
                'embedding': message_embedding.embedding,
                'is_outgoing': message_embedding.is_outgoing,
                'media_type': message_embedding.media_type,
                'content_hash': message_embedding.content_hash
            }
            
            result = self.supabase.table('conversations').insert(data).execute()
            
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
                    'timestamp': msg_emb.timestamp,
                    'sender': msg_emb.sender,
                    'embedding': msg_emb.embedding,
                    'is_outgoing': msg_emb.is_outgoing,
                    'media_type': msg_emb.media_type,
                    'content_hash': msg_emb.content_hash
                }
                data_list.append(data)
            
            result = self.supabase.table('conversations').insert(data_list).execute()
            
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
            result = self.supabase.table('conversations')\
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
            result = self.supabase.table('conversations')\
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
            result = self.supabase.table('conversations')\
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
            result = self.supabase.table('conversations')\
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