"""
WhatsApp Business API - Intégration temps réel avec RAG
Remplace l'extraction Selenium par une API temps réel via webhooks
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone
import requests
import hashlib
import hmac
from dataclasses import dataclass, asdict
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
import uvicorn
from threading import Thread
import time

# Import des modules du projet
from .embedding_processor import EmbeddingProcessor
from .rag_searcher import RAGSearcher

@dataclass
class WhatsAppMessage:
    """Structure pour un message WhatsApp"""
    id: str
    phone_number: str  # Numéro avec indicatif complet
    content: str
    timestamp: str
    sender: str  # 'contact' ou 'me'
    is_outgoing: bool
    media_type: Optional[str] = None
    media_url: Optional[str] = None
    message_type: str = "text"
    status: Optional[str] = None  # delivered, read, etc.

class WhatsAppRealtimeAPI:
    def __init__(self, access_token: str, phone_number_id: str, verify_token: str, 
                 webhook_secret: str, supabase_url: str, supabase_key: str, openai_api_key: str):
        """
        Initialise l'API WhatsApp Business avec intégration RAG temps réel
        
        Args:
            access_token: Token d'accès Facebook
            phone_number_id: ID du numéro WhatsApp Business
            verify_token: Token de vérification webhook
            webhook_secret: Secret pour valider les webhooks
            supabase_url: URL Supabase
            supabase_key: Clé Supabase
            openai_api_key: Clé OpenAI
        """
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.verify_token = verify_token
        self.webhook_secret = webhook_secret
        self.base_url = f"https://graph.facebook.com/v19.0/{phone_number_id}"
        self.graph_url = "https://graph.facebook.com/v19.0"
        
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Initialisation des composants RAG
        self.embedding_processor = EmbeddingProcessor(supabase_url, supabase_key, openai_api_key)
        self.rag_searcher = RAGSearcher(supabase_url, supabase_key, openai_api_key)
        
        # Configuration logging
        self.logger = self._setup_logging()
        
        # Cache des contacts actifs et leurs contextes
        self.active_contacts = {}
        self.message_handlers = []
        
        # Configuration pour les réponses automatiques
        self.auto_response_enabled = False
        self.response_delay = 2  # secondes
        
        # Configuration FastAPI pour les webhooks
        self.app = FastAPI(title="WhatsApp RAG Webhook", version="1.0.0")
        self._setup_webhook_routes()
        
        self.logger.info("WhatsApp Realtime API RAG initialisé")
    
    def _setup_logging(self):
        """Configure le logging"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/whatsapp_realtime.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _setup_webhook_routes(self):
        """Configure les routes FastAPI pour les webhooks"""
        
        @self.app.get("/webhook")
        async def verify_webhook(request: Request):
            """Vérification du webhook Facebook"""
            mode = request.query_params.get("hub.mode")
            token = request.query_params.get("hub.verify_token")
            challenge = request.query_params.get("hub.challenge")
            
            if mode == "subscribe" and token == self.verify_token:
                self.logger.info("✅ Webhook vérifié avec succès")
                return PlainTextResponse(challenge)
            else:
                self.logger.warning("❌ Échec de vérification webhook")
                raise HTTPException(status_code=403, detail="Forbidden")
        
        @self.app.post("/webhook")
        async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
            """Réception des webhooks WhatsApp"""
            try:
                body = await request.body()
                signature = request.headers.get("X-Hub-Signature-256", "")
                
                # Vérifier la signature si secret défini
                if self.webhook_secret and not self._verify_webhook_signature(body, signature):
                    raise HTTPException(status_code=401, detail="Invalid signature")
                
                data = json.loads(body.decode())
                
                # Traiter les messages en arrière-plan
                background_tasks.add_task(self._process_webhook_data, data)
                
                return {"status": "received"}
                
            except Exception as e:
                self.logger.error(f"Erreur webhook: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/health")
        async def health_check():
            """Point de santé pour monitoring"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "active_contacts": len(self.active_contacts),
                "message_handlers": len(self.message_handlers)
            }
        
        @self.app.get("/stats")
        async def get_stats():
            """Statistiques de l'API"""
            try:
                # Récupérer des stats depuis la DB
                total_messages = 0
                contacts_count = 0
                
                # Vous pouvez ajouter d'autres statistiques ici
                
                return {
                    "total_messages": total_messages,
                    "total_contacts": contacts_count,
                    "active_contacts": len(self.active_contacts),
                    "auto_response": self.auto_response_enabled
                }
            except Exception as e:
                self.logger.error(f"Erreur stats: {e}")
                return {"error": str(e)}
    
    def _verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Vérifie la signature du webhook"""
        if not signature.startswith("sha256="):
            return False
        
        expected_signature = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(f"sha256={expected_signature}", signature)
    
    async def _process_webhook_data(self, data: Dict):
        """Traite les données du webhook de façon asynchrone"""
        try:
            self.logger.debug(f"Webhook reçu: {json.dumps(data, indent=2)}")
            
            messages = self._extract_messages_from_webhook(data)
            
            for message in messages:
                await self._handle_incoming_message(message)
                
        except Exception as e:
            self.logger.error(f"Erreur traitement webhook: {e}")
    
    def _extract_messages_from_webhook(self, webhook_data: Dict) -> List[WhatsAppMessage]:
        """Extrait les messages du webhook"""
        messages = []
        
        if webhook_data.get("object") != "whatsapp_business_account":
            return messages
        
        for entry in webhook_data.get("entry", []):
            for change in entry.get("changes", []):
                if change.get("field") == "messages":
                    value = change.get("value", {})
                    
                    # Messages entrants
                    for msg in value.get("messages", []):
                        whatsapp_msg = self._convert_to_whatsapp_message(msg, is_outgoing=False)
                        if whatsapp_msg:
                            messages.append(whatsapp_msg)
                    
                    # Statuts des messages sortants
                    for status in value.get("statuses", []):
                        self._process_message_status(status)
        
        return messages
    
    def _convert_to_whatsapp_message(self, raw_message: Dict, is_outgoing: bool) -> Optional[WhatsAppMessage]:
        """Convertit un message brut en WhatsAppMessage"""
        try:
            message_id = raw_message.get("id")
            phone_number = raw_message.get("from") if not is_outgoing else raw_message.get("to")
            timestamp = datetime.fromtimestamp(int(raw_message.get("timestamp"))).isoformat()
            message_type = raw_message.get("type", "text")
            
            # Normaliser le numéro de téléphone
            phone_number = self._normalize_phone_number(phone_number)
            
            # Extraire le contenu selon le type
            content = ""
            media_url = None
            media_type = None
            
            if message_type == "text":
                content = raw_message.get("text", {}).get("body", "")
            elif message_type in ["image", "audio", "video", "document"]:
                media_type = message_type
                media_url = self._get_media_url(raw_message)
                content = f"[{message_type.upper()}]"
                
                # Ajouter la caption si disponible
                caption = raw_message.get(message_type, {}).get("caption", "")
                if caption:
                    content += f" {caption}"
            elif message_type == "voice":
                media_type = "voice"
                media_url = self._get_media_url(raw_message)
                content = "[MESSAGE VOCAL]"
            elif message_type == "location":
                location = raw_message.get("location", {})
                lat = location.get("latitude")
                lng = location.get("longitude")
                content = f"[LOCALISATION] Lat: {lat}, Lng: {lng}"
                if location.get("name"):
                    content += f" - {location['name']}"
            elif message_type == "contacts":
                contacts = raw_message.get("contacts", [])
                content = f"[CONTACT] {len(contacts)} contact(s) partagé(s)"
            else:
                content = f"[{message_type.upper()}]"
            
            return WhatsAppMessage(
                id=message_id,
                phone_number=phone_number,
                content=content,
                timestamp=timestamp,
                sender="contact" if not is_outgoing else "me",
                is_outgoing=is_outgoing,
                media_type=media_type,
                media_url=media_url,
                message_type=message_type
            )
            
        except Exception as e:
            self.logger.error(f"Erreur conversion message: {e}")
            return None
    
    def _normalize_phone_number(self, phone: str) -> str:
        """Normalise le numéro de téléphone au format international"""
        if not phone:
            return ""
        
        # Supprimer les espaces et caractères spéciaux
        clean_phone = ''.join(filter(str.isdigit, phone))
        
        # Ajouter le + si pas présent
        if not phone.startswith('+'):
            return f"+{clean_phone}"
        
        return phone
    
    def _get_media_url(self, message: Dict) -> Optional[str]:
        """Récupère l'URL du média"""
        try:
            message_type = message.get("type")
            if message_type in message:
                media_id = message[message_type].get("id")
                if media_id:
                    # Récupérer l'URL via l'API Graph
                    url = f"{self.graph_url}/{media_id}"
                    response = requests.get(url, headers=self.headers)
                    if response.status_code == 200:
                        return response.json().get("url")
        except Exception as e:
            self.logger.error(f"Erreur récupération média: {e}")
        
        return None
    
    async def _handle_incoming_message(self, message: WhatsAppMessage):
        """Traite un message entrant en temps réel"""
        try:
            self.logger.info(f"📨 Message reçu de {message.phone_number}: {message.content[:50]}...")
            
            # 1. Marquer le message comme lu
            await self._mark_message_as_read(message.id)
            
            # 2. Stocker le message dans la base de données avec embedding
            await self._store_message_realtime(message)
            
            # 3. Traiter avec le RAG si approprié
            if message.message_type == "text" and message.content.strip():
                await self._process_with_rag_realtime(message)
            
            # 4. Mettre à jour le contexte du contact
            self._update_contact_context(message.phone_number, message)
            
            # 5. Notifier les handlers personnalisés
            for handler in self.message_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    self.logger.error(f"Erreur handler: {e}")
                    
        except Exception as e:
            self.logger.error(f"Erreur traitement message: {e}")
    
    async def _mark_message_as_read(self, message_id: str):
        """Marque un message comme lu"""
        try:
            url = f"{self.base_url}/messages"
            payload = {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                self.logger.debug(f"Message {message_id} marqué comme lu")
                
        except Exception as e:
            self.logger.error(f"Erreur marquage lu: {e}")
    
    async def _store_message_realtime(self, message: WhatsAppMessage):
        """Stocke un message en temps réel avec embedding"""
        try:
            # Convertir en format compatible avec embedding_processor
            message_dict = {
                "id": message.id,
                "content": message.content,
                "timestamp": message.timestamp,
                "sender": message.sender,
                "is_outgoing": message.is_outgoing,
                "media_type": message.media_type
            }
            
            # Traitement et stockage asynchrone
            success = self.embedding_processor.process_and_store_conversation(
                messages=[message_dict],
                phone_number=message.phone_number
            )
            
            if success:
                self.logger.info(f"✅ Message stocké et indexé: {message.id}")
            else:
                self.logger.error(f"❌ Échec stockage message: {message.id}")
                
        except Exception as e:
            self.logger.error(f"Erreur stockage temps réel: {e}")
    
    async def _process_with_rag_realtime(self, message: WhatsAppMessage):
        """Traite un message avec le système RAG en temps réel"""
        try:
            content = message.content.lower().strip()
            
            # Détection intelligente de questions ou demandes
            question_indicators = [
                '?', 'quoi', 'comment', 'pourquoi', 'quand', 'où', 'qui',
                'peux-tu', 'pourrais-tu', 'aide', 'explique', 'dis-moi',
                'recherche', 'trouve', 'montre', 'rappel'
            ]
            
            is_query = any(indicator in content for indicator in question_indicators)
            
            # Ou si le message commence par certains mots clés
            query_starters = ['recherche', 'trouve', 'dis-moi', 'explique', 'rappel', 'quand']
            starts_with_query = any(content.startswith(starter) for starter in query_starters)
            
            if is_query or starts_with_query:
                self.logger.info(f"🤖 Traitement RAG pour: {message.content[:30]}...")
                
                # Recherche RAG avec contexte
                result = self.rag_searcher.search_and_respond(
                    query=message.content,
                    phone_number=message.phone_number,
                    include_context=True
                )
                
                if result.get('response') and not result.get('error'):
                    response_text = result['response']
                    
                    # Ajouter des métadonnées sur les résultats trouvés
                    if result.get('search_results'):
                        response_text += f"\n\n📊 Basé sur {len(result['search_results'])} message(s) trouvé(s)"
                    
                    # Envoyer la réponse automatique si activée
                    if self.auto_response_enabled:
                        await asyncio.sleep(self.response_delay)  # Délai naturel
                        await self._send_response_async(message.phone_number, response_text)
                    
                    self.logger.info(f"✅ Réponse RAG générée pour {message.phone_number}")
                else:
                    self.logger.warning(f"⚠️ Aucune réponse RAG générée: {result.get('error', 'Raison inconnue')}")
                
        except Exception as e:
            self.logger.error(f"Erreur RAG temps réel: {e}")
    
    def _update_contact_context(self, phone_number: str, message: WhatsAppMessage):
        """Met à jour le contexte d'un contact"""
        if phone_number not in self.active_contacts:
            self.active_contacts[phone_number] = {
                'last_message_time': message.timestamp,
                'message_count': 0,
                'last_messages': []
            }
        
        context = self.active_contacts[phone_number]
        context['last_message_time'] = message.timestamp
        context['message_count'] += 1
        context['last_messages'].append({
            'content': message.content[:100],  # Limiter pour mémoire
            'timestamp': message.timestamp,
            'type': message.message_type
        })
        
        # Garder seulement les 10 derniers messages en contexte
        if len(context['last_messages']) > 10:
            context['last_messages'] = context['last_messages'][-10:]
    
    def _process_message_status(self, status: Dict):
        """Traite le statut d'un message"""
        try:
            message_id = status.get("id")
            recipient_id = status.get("recipient_id")
            status_type = status.get("status")  # sent, delivered, read, failed
            timestamp = status.get("timestamp")
            
            self.logger.debug(f"📋 Statut message {message_id}: {status_type}")
            
            # Ici vous pouvez mettre à jour le statut dans votre base de données
            # ou déclencher des actions spécifiques
            
        except Exception as e:
            self.logger.error(f"Erreur traitement statut: {e}")
    
    def send_message(self, to: str, message: str, message_type: str = "text") -> Dict:
        """Envoie un message WhatsApp"""
        try:
            url = f"{self.base_url}/messages"
            
            # Normaliser le numéro (retirer le +)
            to_clean = self._normalize_phone_number(to).replace('+', '')
            
            payload = {
                "messaging_product": "whatsapp",
                "to": to_clean,
                "type": message_type
            }
            
            if message_type == "text":
                payload["text"] = {"body": message}
            
            response = requests.post(url, headers=self.headers, json=payload)
            result = response.json()
            
            if response.status_code == 200:
                self.logger.info(f"📤 Message envoyé à {to}")
                
                # Stocker le message sortant
                outgoing_message = WhatsAppMessage(
                    id=result.get("messages", [{}])[0].get("id", ""),
                    phone_number=self._normalize_phone_number(to),
                    content=message,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    sender="me",
                    is_outgoing=True,
                    message_type=message_type
                )
                
                # Stocker de façon asynchrone
                asyncio.create_task(self._store_message_realtime(outgoing_message))
            else:
                self.logger.error(f"❌ Erreur envoi: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur envoi message: {e}")
            return {"error": str(e)}
    
    async def _send_response_async(self, phone_number: str, response: str):
        """Envoie une réponse de façon asynchrone"""
        try:
            # Limiter la longueur des réponses automatiques
            if len(response) > 1000:
                response = response[:997] + "..."
            
            self.send_message(phone_number, response)
            
        except Exception as e:
            self.logger.error(f"Erreur envoi réponse async: {e}")
    
    def enable_auto_responses(self, enabled: bool = True, delay: int = 2):
        """Active ou désactive les réponses automatiques"""
        self.auto_response_enabled = enabled
        self.response_delay = delay
        self.logger.info(f"🤖 Réponses automatiques: {'activées' if enabled else 'désactivées'}")
    
    def get_conversation_history(self, phone_number: str, days_back: int = 30) -> List[Dict]:
        """Récupère l'historique d'une conversation"""
        try:
            from datetime import timedelta
            date_from = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            result = self.embedding_processor.supabase.table('conversations')\
                .select('*')\
                .eq('phone_number', self._normalize_phone_number(phone_number))\
                .gte('timestamp', date_from)\
                .order('timestamp', desc=False)\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            self.logger.error(f"Erreur récupération historique: {e}")
            return []
    
    def add_message_handler(self, handler: Callable[[WhatsAppMessage], None]):
        """Ajoute un gestionnaire de messages personnalisé"""
        self.message_handlers.append(handler)
        self.logger.info(f"Handler ajouté. Total: {len(self.message_handlers)}")
    
    def start_webhook_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Démarre le serveur webhook"""
        def run_server():
            uvicorn.run(self.app, host=host, port=port, log_level="info")
        
        server_thread = Thread(target=run_server, daemon=True)
        server_thread.start()
        self.logger.info(f"🚀 Serveur webhook démarré sur {host}:{port}")
        return server_thread
    
    def setup_webhook_configuration(self, webhook_url: str) -> Dict:
        """Affiche les instructions de configuration du webhook"""
        config_info = {
            "webhook_url": f"{webhook_url}/webhook",
            "verify_token": self.verify_token,
            "webhook_secret": self.webhook_secret,
            "instructions": [
                "1. Allez sur developers.facebook.com",
                "2. Sélectionnez votre app WhatsApp Business",
                "3. Allez dans 'Configuration' > 'Webhooks'",
                f"4. URL du webhook: {webhook_url}/webhook",
                f"5. Token de vérification: {self.verify_token}",
                "6. Abonnez-vous aux événements 'messages' et 'message_deliveries'"
            ]
        }
        
        self.logger.info("📋 Configuration webhook:")
        for instruction in config_info["instructions"]:
            self.logger.info(f"   {instruction}")
        
        return config_info
    
    def get_active_contacts(self) -> Dict:
        """Retourne les contacts actifs avec leur contexte"""
        return self.active_contacts.copy()
    
    def cleanup_inactive_contacts(self, hours_threshold: int = 24):
        """Nettoie les contacts inactifs du cache"""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(hours=hours_threshold)
            
            inactive_contacts = []
            for phone, context in self.active_contacts.items():
                last_time = datetime.fromisoformat(context['last_message_time'].replace('Z', '+00:00'))
                if last_time < cutoff_time:
                    inactive_contacts.append(phone)
            
            for phone in inactive_contacts:
                del self.active_contacts[phone]
            
            if inactive_contacts:
                self.logger.info(f"🧹 Nettoyage: {len(inactive_contacts)} contacts inactifs supprimés")
            
        except Exception as e:
            self.logger.error(f"Erreur nettoyage contacts: {e}")


# Exemple d'utilisation
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configuration
    whatsapp_api = WhatsAppRealtimeAPI(
        access_token=os.getenv('WHATSAPP_ACCESS_TOKEN'),
        phone_number_id=os.getenv('WHATSAPP_PHONE_NUMBER_ID'),
        verify_token=os.getenv('WHATSAPP_VERIFY_TOKEN'),
        webhook_secret=os.getenv('WHATSAPP_WEBHOOK_SECRET'),
        supabase_url=os.getenv('SUPABASE_URL'),
        supabase_key=os.getenv('SUPABASE_ANON_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # Activer les réponses automatiques
    whatsapp_api.enable_auto_responses(enabled=True, delay=3)
    
    # Ajouter un handler personnalisé
    def custom_message_handler(message: WhatsAppMessage):
        print(f"🔔 Nouveau message: {message.phone_number} - {message.content[:30]}...")
    
    whatsapp_api.add_message_handler(custom_message_handler)
    
    # Démarrer le serveur
    server_thread = whatsapp_api.start_webhook_server(port=8000)
    
    # Afficher les instructions de configuration
    whatsapp_api.setup_webhook_configuration("https://votre-domaine.com")
    
    print("🤖 WhatsApp RAG API en cours d'exécution...")
    print("Appuyez sur Ctrl+C pour arrêter")
    
    try:
        while True:
            time.sleep(10)
            # Nettoyage périodique
            whatsapp_api.cleanup_inactive_contacts()
    except KeyboardInterrupt:
        print("\n👋 Arrêt de l'API WhatsApp RAG")
