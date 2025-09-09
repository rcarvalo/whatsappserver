import os
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
from supabase import create_client, Client
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class SearchResult:
    """Structure pour un résultat de recherche"""
    id: int
    phone_number: str
    message_content: str
    timestamp: str
    sender: str
    similarity: float
    is_outgoing: bool
    media_type: Optional[str] = None

@dataclass
class ConversationContext:
    """Structure pour le contexte d'une conversation"""
    phone_number: str
    messages: List[SearchResult]
    total_messages: int
    date_range: Dict[str, str]
    summary: Optional[str] = None

class RAGSearcher:
    def __init__(self, supabase_url: str, supabase_key: str, openai_api_key: str):
        """
        Initialise le système de recherche RAG
        
        Args:
            supabase_url: URL Supabase
            supabase_key: Clé API Supabase
            openai_api_key: Clé API OpenAI
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.logger = self._setup_logging()
        
        # Configuration de la recherche
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o"
        self.default_similarity_threshold = 0.7
        self.max_context_messages = 10
        self.max_response_tokens = 500
        
        # Cache pour les embeddings récents
        self.embedding_cache = {}
        
    def _setup_logging(self):
        """Configure le logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rag_searcher.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Génère un embedding pour la requête de recherche
        
        Args:
            query: Texte de la requête
            
        Returns:
            Vecteur d'embedding ou None
        """
        try:
            # Vérifier le cache
            if query in self.embedding_cache:
                return self.embedding_cache[query]
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query.strip()
            )
            
            embedding = response.data[0].embedding
            
            # Ajouter au cache (limité à 100 entrées)
            if len(self.embedding_cache) >= 100:
                # Supprimer le plus ancien
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
            
            self.embedding_cache[query] = embedding
            return embedding
            
        except Exception as e:
            self.logger.error(f"Erreur génération embedding requête: {e}")
            return None
    
    def semantic_search(self, query: str, phone_number: Optional[str] = None,
                       similarity_threshold: float = None, limit: int = 5,
                       date_from: Optional[str] = None, date_to: Optional[str] = None) -> List[SearchResult]:
        """
        Effectue une recherche sémantique dans les conversations
        
        Args:
            query: Requête de recherche
            phone_number: Numéro spécifique à rechercher (optionnel)
            similarity_threshold: Seuil de similarité minimum
            limit: Nombre maximum de résultats
            date_from: Date de début (ISO format)
            date_to: Date de fin (ISO format)
            
        Returns:
            Liste des résultats de recherche
        """
        try:
            self.logger.info(f"Recherche sémantique: '{query}'" + (f" pour {phone_number}" if phone_number else ""))
            
            # Générer l'embedding de la requête
            query_embedding = self.generate_query_embedding(query)
            if not query_embedding:
                return []
            
            threshold = similarity_threshold or self.default_similarity_threshold
            logger.info(f"query_embedding: {query_embedding}")
            logger.info(f"match_threshold: {threshold}")
            logger.info(f"limit: {limit}")
            # Appeler la fonction PostgreSQL de recherche
            rpc_params = {
                'query_embedding': query_embedding,
                'match_threshold': threshold,
                'match_count': limit
            }
            logger.info(f"RPC params: {rpc_params}")
            if phone_number:
                rpc_params['phone_filter'] = phone_number
            
            if date_from:
                rpc_params['date_from'] = date_from
                
            if date_to:
                rpc_params['date_to'] = date_to
            
            result = self.supabase.rpc('search_watch_conversations', rpc_params).execute()
            
            if not result.data:
                self.logger.info("Aucun résultat trouvé")
                return []
            
            # Convertir en objets SearchResult
            search_results = []
            for row in result.data:
                search_result = SearchResult(
                    id=row['id'],
                    phone_number=row['phone_number'],
                    message_content=row['message_content'],
                    timestamp=row['timestamp'],
                    sender=row['sender'],
                    similarity=row['similarity'],
                    is_outgoing=row['is_outgoing'],
                    media_type=row.get('media_type')
                )
                search_results.append(search_result)
            
            self.logger.info(f"Trouvé {len(search_results)} résultats")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Erreur recherche sémantique: {e}")
            return []
    
    def keyword_search(self, keywords: List[str], phone_number: Optional[str] = None,
                      limit: int = 10, case_sensitive: bool = False) -> List[SearchResult]:
        """
        Effectue une recherche par mots-clés
        
        Args:
            keywords: Liste de mots-clés
            phone_number: Numéro spécifique (optionnel)
            limit: Nombre maximum de résultats
            case_sensitive: Recherche sensible à la casse
            
        Returns:
            Liste des résultats
        """
        try:
            self.logger.info(f"Recherche mots-clés: {keywords}" + (f" pour {phone_number}" if phone_number else ""))
            
            # Construire la requête SQL
            query_builder = self.supabase.table('watch_conversations').select('*')
            
            if phone_number:
                query_builder = query_builder.eq('phone_number', phone_number)
            
            # Recherche dans le contenu du message
            if not case_sensitive:
                keywords = [kw.lower() for kw in keywords]
                for keyword in keywords:
                    query_builder = query_builder.ilike('message_content', f'%{keyword}%')
            else:
                for keyword in keywords:
                    query_builder = query_builder.like('message_content', f'%{keyword}%')
            
            query_builder = query_builder.order('timestamp', desc=True).limit(limit)
            
            result = query_builder.execute()
            
            if not result.data:
                return []
            
            # Convertir en SearchResult
            search_results = []
            for row in result.data:
                search_result = SearchResult(
                    id=row['id'],
                    phone_number=row['phone_number'],
                    message_content=row['message_content'],
                    timestamp=row['timestamp'],
                    sender=row['sender'],
                    similarity=1.0,  # Correspondance exacte pour les mots-clés
                    is_outgoing=row['is_outgoing'],
                    media_type=row.get('media_type')
                )
                search_results.append(search_result)
            
            self.logger.info(f"Trouvé {len(search_results)} résultats par mots-clés")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Erreur recherche mots-clés: {e}")
            return []
    
    def get_conversation_context(self, phone_number: str, around_timestamp: Optional[str] = None,
                               context_size: int = 5) -> ConversationContext:
        """
        Récupère le contexte d'une conversation autour d'un timestamp
        
        Args:
            phone_number: Numéro de téléphone
            around_timestamp: Timestamp central (optionnel)
            context_size: Nombre de messages avant/après
            
        Returns:
            Contexte de la conversation
        """
        try:
            query_builder = self.supabase.table('watch_conversations')\
                .select('*')\
                .eq('phone_number', phone_number)\
                .order('timestamp', desc=False)
            
            if around_timestamp:
                # Récupérer les messages autour du timestamp
                query_builder = query_builder.gte('timestamp', around_timestamp)\
                    .limit(context_size * 2)
            else:
                # Récupérer les messages les plus récents
                query_builder = query_builder.limit(context_size)
            
            result = query_builder.execute()
            
            if not result.data:
                return ConversationContext(
                    phone_number=phone_number,
                    messages=[],
                    total_messages=0,
                    date_range={}
                )
            
            # Convertir en SearchResult
            messages = []
            for row in result.data:
                search_result = SearchResult(
                    id=row['id'],
                    phone_number=row['phone_number'],
                    message_content=row['message_content'],
                    timestamp=row['timestamp'],
                    sender=row['sender'],
                    similarity=1.0,
                    is_outgoing=row['is_outgoing'],
                    media_type=row.get('media_type')
                )
                messages.append(search_result)
            
            # Calculer les statistiques
            timestamps = [msg.timestamp for msg in messages]
            date_range = {
                'earliest': min(timestamps) if timestamps else '',
                'latest': max(timestamps) if timestamps else ''
            }
            
            return ConversationContext(
                phone_number=phone_number,
                messages=messages,
                total_messages=len(messages),
                date_range=date_range
            )
            
        except Exception as e:
            self.logger.error(f"Erreur récupération contexte: {e}")
            return ConversationContext(
                phone_number=phone_number,
                messages=[],
                total_messages=0,
                date_range={}
            )
    
    def generate_response(self, query: str, search_results: List[SearchResult],
                         conversation_context: Optional[ConversationContext] = None,
                         response_language: str = "français") -> str:
        """
        Génère une réponse basée sur les résultats de recherche
        
        Args:
            query: Requête originale
            search_results: Résultats de la recherche sémantique
            conversation_context: Contexte additionnel
            response_language: Langue de la réponse
            
        Returns:
            Réponse générée
        """
        try:
            if not search_results:
                return f"Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question : '{query}'"
            
            # Construire le contexte pour le prompt
            context_messages = []
            
            # Ajouter les résultats de recherche
            for result in search_results[:self.max_context_messages]:
                timestamp_formatted = self._format_timestamp(result.timestamp)
                sender_label = "Vous" if result.is_outgoing else "Contact"
                
                context_msg = f"[{timestamp_formatted}] {sender_label}: {result.message_content}"
                context_messages.append(context_msg)
            
            # Ajouter le contexte conversationnel si disponible
            if conversation_context and conversation_context.messages:
                for ctx_msg in conversation_context.messages[-3:]:  # Derniers messages pour contexte
                    timestamp_formatted = self._format_timestamp(ctx_msg.timestamp)
                    sender_label = "Vous" if ctx_msg.is_outgoing else "Contact"
                    context_msg = f"[Contexte - {timestamp_formatted}] {sender_label}: {ctx_msg.message_content}"
                    context_messages.append(context_msg)
            
            context_text = "\n\n".join(context_messages)
            
            # Construire le prompt pour GPT
            system_prompt = f"""Vous êtes un assistant intelligent qui analyse des conversations WhatsApp.
Répondez en {response_language} de manière naturelle et utile.

INSTRUCTIONS:
- Basez votre réponse UNIQUEMENT sur les informations fournies dans le contexte
- Si l'information n'est pas disponible, dites-le clairement
- Soyez précis et concis
- Mentionnez les dates/heures pertinentes si importantes
- Si plusieurs personnes sont mentionnées, précisez qui a dit quoi
- Gardez un ton professionnel mais accessible"""

            user_prompt = f"""Contexte des conversations WhatsApp:
{context_text}

Question: {query}

Répondez en vous basant sur le contexte fourni ci-dessus."""

            # Générer la réponse avec OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_response_tokens,
                temperature=0.7
            )
            
            generated_response = response.choices[0].message.content
            
            # Ajouter des métadonnées sur les sources
            sources_info = f"\n\n📊 Sources: {len(search_results)} message(s) analysé(s)"
            if search_results:
                phone_numbers = list(set(result.phone_number for result in search_results))
                if len(phone_numbers) == 1:
                    sources_info += f" de {phone_numbers[0]}"
                else:
                    sources_info += f" de {len(phone_numbers)} contact(s)"
            
            return generated_response + sources_info
            
        except Exception as e:
            self.logger.error(f"Erreur génération réponse: {e}")
            return f"Désolé, j'ai rencontré une erreur en générant la réponse pour: '{query}'"
    
    def _format_timestamp(self, timestamp_str: str) -> str:
        """
        Formate un timestamp pour l'affichage
        
        Args:
            timestamp_str: Timestamp au format ISO
            
        Returns:
            Timestamp formaté
        """
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime("%d/%m/%Y %H:%M")
        except:
            return timestamp_str
    
    def search_and_respond(self, query: str, phone_number: Optional[str] = None,
                          include_context: bool = True) -> Dict:
        """
        Effectue une recherche complète et génère une réponse
        
        Args:
            query: Requête de recherche
            phone_number: Numéro spécifique (optionnel)
            include_context: Inclure le contexte conversationnel
            
        Returns:
            Dictionnaire avec résultats et réponse
        """
        try:
            # Recherche sémantique
            search_results = self.semantic_search(query, phone_number)
            
            # Récupérer le contexte si demandé
            conversation_context = None
            if include_context and phone_number:
                conversation_context = self.get_conversation_context(phone_number)
            
            # Générer la réponse
            response = self.generate_response(query, search_results, conversation_context)
            
            return {
                'query': query,
                'response': response,
                'search_results': [
                    {
                        'phone_number': result.phone_number,
                        'message': result.message_content,
                        'timestamp': result.timestamp,
                        'sender': result.sender,
                        'similarity': result.similarity,
                        'is_outgoing': result.is_outgoing
                    }
                    for result in search_results
                ],
                'context_used': conversation_context is not None,
                'total_results': len(search_results)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur recherche complète: {e}")
            return {
                'query': query,
                'response': f"Erreur lors de la recherche: {str(e)}",
                'search_results': [],
                'context_used': False,
                'total_results': 0,
                'error': str(e)
            }
    
    def get_conversation_summary(self, phone_number: str, days_back: int = 7) -> str:
        """
        Génère un résumé d'une conversation sur une période
        
        Args:
            phone_number: Numéro de téléphone
            days_back: Nombre de jours à analyser
            
        Returns:
            Résumé de la conversation
        """
        try:
            # Calculer la date de début
            date_from = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            # Récupérer les messages récents
            result = self.supabase.table('conversations')\
                .select('message_content, timestamp, sender, is_outgoing')\
                .eq('phone_number', phone_number)\
                .gte('timestamp', date_from)\
                .order('timestamp', desc=False)\
                .execute()
            
            if not result.data or len(result.data) == 0:
                return f"Aucune conversation trouvée avec {phone_number} dans les {days_back} derniers jours."
            
            # Préparer le contexte pour le résumé
            messages_text = []
            for msg in result.data:
                sender_label = "Vous" if msg['is_outgoing'] else "Contact"
                timestamp_formatted = self._format_timestamp(msg['timestamp'])
                messages_text.append(f"[{timestamp_formatted}] {sender_label}: {msg['message_content']}")
            
            context_text = "\n".join(messages_text)
            
            # Générer le résumé
            system_prompt = """Vous êtes un assistant qui résume des conversations WhatsApp.
Créez un résumé concis et informatif en français qui capture:
- Les sujets principaux discutés
- Les décisions prises ou accords conclus
- Les événements importants mentionnés
- Le ton général de la conversation
Gardez le résumé objectif et factuel."""

            user_prompt = f"""Conversation WhatsApp à résumer:
{context_text}

Créez un résumé de cette conversation des {days_back} derniers jours."""

            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.5
            )
            
            summary = response.choices[0].message.content
            
            # Ajouter les statistiques
            stats = f"\n\n📊 Statistiques: {len(result.data)} messages sur {days_back} jours"
            
            return summary + stats
            
        except Exception as e:
            self.logger.error(f"Erreur génération résumé: {e}")
            return f"Erreur lors de la génération du résumé pour {phone_number}"
    
    def find_similar_conversations(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Trouve des conversations similaires à travers tous les contacts
        
        Args:
            query: Requête de recherche
            limit: Nombre de conversations à retourner
            
        Returns:
            Liste des conversations similaires groupées par numéro
        """
        try:
            # Recherche sémantique globale
            search_results = self.semantic_search(query, limit=limit*3)  # Plus de résultats pour grouper
            
            # Grouper par numéro de téléphone
            conversations = {}
            for result in search_results:
                phone = result.phone_number
                if phone not in conversations:
                    conversations[phone] = {
                        'phone_number': phone,
                        'messages': [],
                        'max_similarity': 0,
                        'message_count': 0
                    }
                
                conversations[phone]['messages'].append({
                    'content': result.message_content,
                    'timestamp': result.timestamp,
                    'similarity': result.similarity,
                    'sender': result.sender
                })
                
                if result.similarity > conversations[phone]['max_similarity']:
                    conversations[phone]['max_similarity'] = result.similarity
                
                conversations[phone]['message_count'] += 1
            
            # Trier par pertinence
            sorted_conversations = sorted(
                conversations.values(),
                key=lambda x: x['max_similarity'],
                reverse=True
            )[:limit]
            
            return sorted_conversations
            
        except Exception as e:
            self.logger.error(f"Erreur recherche conversations similaires: {e}")
            return []
    
    def advanced_search(self, query: str, filters: Dict) -> List[SearchResult]:
        """
        Recherche avancée avec filtres multiples
        
        Args:
            query: Requête de recherche
            filters: Dictionnaire de filtres
                - phone_numbers: Liste de numéros
                - date_from/date_to: Plage de dates
                - senders: Liste de types d'expéditeurs ('me', 'contact')
                - similarity_threshold: Seuil de similarité
                - exclude_media: Exclure les messages média
                - keywords: Mots-clés obligatoires
                
        Returns:
            Résultats filtrés
        """
        try:
            # Recherche sémantique de base
            search_results = self.semantic_search(
                query=query,
                phone_number=filters.get('phone_number'),
                similarity_threshold=filters.get('similarity_threshold'),
                limit=filters.get('limit', 20),
                date_from=filters.get('date_from'),
                date_to=filters.get('date_to')
            )
            
            # Appliquer les filtres additionnels
            filtered_results = []
            
            for result in search_results:
                # Filtre par numéros de téléphone
                if filters.get('phone_numbers'):
                    if result.phone_number not in filters['phone_numbers']:
                        continue
                
                # Filtre par type d'expéditeur
                if filters.get('senders'):
                    sender_type = 'me' if result.is_outgoing else 'contact'
                    if sender_type not in filters['senders']:
                        continue
                
                # Exclure les médias
                if filters.get('exclude_media', False):
                    if result.media_type:
                        continue
                
                # Mots-clés obligatoires
                if filters.get('keywords'):
                    content_lower = result.message_content.lower()
                    if not all(keyword.lower() in content_lower for keyword in filters['keywords']):
                        continue
                
                filtered_results.append(result)
            
            self.logger.info(f"Recherche avancée: {len(filtered_results)} résultats après filtrage")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Erreur recherche avancée: {e}")
            return []
    
    def get_search_suggestions(self, partial_query: str, phone_number: Optional[str] = None) -> List[str]:
        """
        Génère des suggestions de recherche basées sur le contenu existant
        
        Args:
            partial_query: Début de requête
            phone_number: Numéro spécifique (optionnel)
            
        Returns:
            Liste de suggestions
        """
        try:
            # Cette fonction pourrait être améliorée avec une vraie analyse des fréquences de mots
            # ou des modèles de suggestion plus sophistiqués
            
            suggestions = []
            
            # Suggestions génériques basées sur des patterns courants
            common_patterns = [
                "réunion", "rendez-vous", "travail", "projet", "famille",
                "weekend", "vacances", "restaurant", "film", "livre",
                "santé", "médecin", "sport", "voyage", "argent"
            ]
            
            partial_lower = partial_query.lower()
            for pattern in common_patterns:
                if pattern.startswith(partial_lower) and len(partial_lower) >= 2:
                    suggestions.append(f"Messages contenant '{pattern}'")
            
            # Suggestions basées sur les requêtes fréquentes
            if len(partial_lower) >= 3:
                suggestions.extend([
                    f"Conversations récentes sur {partial_query}",
                    f"Messages de la semaine dernière concernant {partial_query}",
                    f"Tous les échanges sur {partial_query}"
                ])
            
            return suggestions[:5]  # Limiter à 5 suggestions
            
        except Exception as e:
            self.logger.error(f"Erreur génération suggestions: {e}")
            return []

# Exemple d'utilisation et tests
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configuration
    searcher = RAGSearcher(
        supabase_url=os.getenv('SUPABASE_URL'),
        supabase_key=os.getenv('SUPABASE_ANON_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # Exemple de recherche simple
    query = "Parle-moi des projets dont nous avons discuté"
    phone_number = "+33123456789"
    
    result = searcher.search_and_respond(query, phone_number)
    
    print("=== RÉSULTAT DE RECHERCHE ===")
    print(f"Requête: {result['query']}")
    print(f"Réponse: {result['response']}")
    print(f"Nombre de résultats: {result['total_results']}")
    
    if result['search_results']:
        print("\n=== MESSAGES TROUVÉS ===")
        for i, msg in enumerate(result['search_results'][:3], 1):
            print(f"{i}. [{msg['timestamp']}] {msg['sender']}: {msg['message'][:100]}...")
            print(f"   Similarité: {msg['similarity']:.2%}")
    
    # Exemple de recherche avancée
    print("\n=== RECHERCHE AVANCÉE ===")
    filters = {
        'date_from': '2024-01-01',
        'senders': ['contact'],  # Seulement les messages reçus
        'similarity_threshold': 0.8,
        'exclude_media': True,
        'limit': 10
    }
    
    advanced_results = searcher.advanced_search("réunion projet", filters)
    print(f"Résultats avancés: {len(advanced_results)} messages")
    
    # Exemple de résumé de conversation
    print("\n=== RÉSUMÉ DE CONVERSATION ===")
    summary = searcher.get_conversation_summary(phone_number, days_back=7)
    print(summary)