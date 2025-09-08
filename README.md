# 🔧 WhatsApp RAG - Serveur Webhook

> Serveur Python FastAPI pour réception et traitement automatique des messages WhatsApp avec extraction d'informations montres

## 📋 Description

Ce serveur gère la partie **backend** du système WhatsApp RAG :
- 📨 **Réception webhook** des messages WhatsApp Business API
- 🕰️ **Extraction automatique** d'informations sur les montres
- 🧠 **Génération d'embeddings** avec OpenAI
- 💾 **Stockage enrichi** en base Supabase
- 📊 **Statistiques** en temps réel

## ⚠️ Configuration Sécurisée

### Variables d'environnement (Render)
```env
# WhatsApp Business API
WHATSAPP_VERIFY_TOKEN=your_verify_token
WHATSAPP_ACCESS_TOKEN=your_access_token  
WHATSAPP_PHONE_NUMBER_ID=your_phone_id

# OpenAI pour embeddings
OPENAI_API_KEY=your_openai_key

# Supabase pour stockage
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key
```

### ⚠️ IMPORTANT - Sécurité
- ❌ **Ne jamais commiter** de fichiers `.env`
- ✅ **Utiliser les variables d'environnement** de Render
- 🔒 **Tokens WhatsApp** : Configurer directement sur Render
- 🔐 **Clés API** : Variables d'environnement uniquement

## 🚀 Déploiement Render

### 1. Build Command
```bash
pip install -r requirements.txt
```

### 2. Start Command
```bash
cd whatsappserver && uvicorn app:app --host 0.0.0.0 --port $PORT
```

### 3. Variables d'environnement
Configurer dans l'interface Render (Settings → Environment)

## 🔧 Architecture Serveur

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   WhatsApp      │────│  Webhook Render  │────│    Supabase     │
│ Business API    │    │   (ce serveur)   │    │  (stockage)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │     OpenAI       │
                       │  (embeddings)    │
                       └──────────────────┘
```

## 📂 Structure

```
whatsappserver/
├── app.py                    # Application FastAPI principale
├── requirements.txt          # Dépendances Python
├── .gitignore               # Fichiers à ignorer (avec .env)
├── src/
│   ├── embedding_processor.py    # Traitement embeddings
│   ├── rag_searcher.py           # Recherche sémantique
│   └── watch_info_extractor.py   # Extraction infos montres
└── RENDER_DEPLOYMENT.md     # Guide déploiement
```

## 📨 Endpoints

### `/webhook` (GET)
Vérification du webhook par Meta
```http
GET /webhook?hub.mode=subscribe&hub.verify_token=TOKEN&hub.challenge=CHALLENGE
```

### `/webhook` (POST)  
Réception des messages WhatsApp
```http
POST /webhook
Content-Type: application/json

{
  "object": "whatsapp_business_account",
  "entry": [...]
}
```

### `/health` (GET)
État du serveur et statistiques
```http
GET /health
```

Retourne :
```json
{
  "status": "healthy",
  "components": {
    "webhook": true,
    "whatsapp_api": true,
    "embedding_processor": true,
    "rag_searcher": true,
    "supabase": true,
    "openai": true,
    "watch_extractor": true
  },
  "runtime_stats": {
    "messages_received": 42,
    "messages_processed": 41,
    "errors": 1,
    "watch_messages_detected": 15,
    "sales_detected": 8,
    "wanted_detected": 3,
    "questions_detected": 4,
    "total_value_detected": 85420.50
  },
  "watch_stats": {
    "messages_analyzed": 15,
    "sales_found": 8,
    "wanted_found": 3,
    "questions_found": 4,
    "total_value_eur": "85420.50€"
  }
}
```

## 🕰️ Extraction de Montres

Le serveur analyse automatiquement chaque message pour extraire :

- **Marques** : Rolex, Omega, Seiko, Tudor, Tissot, etc.
- **Modèles** : Submariner, Speedmaster, GMT Master, etc.
- **Prix** : Détection multi-devises (€, $, £, CHF)
- **Conditions** : neuf, excellent, bon, occasion, vintage
- **Caractéristiques** : Taille, année, mouvement, authenticité
- **Type de message** : sale, wanted, question, trade

### Exemple de traitement
```
📨 Message reçu: "Vends Rolex Submariner 116610LN de 2018, excellent état, 8200€"

🔍 Extraction:
  - Marque: Rolex
  - Modèle: Submariner 
  - Référence: 116610LN
  - Prix: 8200€
  - Condition: excellent
  - Année: 2018
  - Type: sale
  - Confiance: 0.92

💾 Stockage:
  - Embedding OpenAI généré
  - Métadonnées enrichies
  - Stockage Supabase avec index vectoriel
```

## 🔄 Flux de Traitement

1. **Réception** : Message WhatsApp via webhook
2. **Extraction** : Analyse automatique des informations montres  
3. **Embedding** : Génération vectorielle OpenAI
4. **Enrichissement** : Métadonnées + informations extraites
5. **Stockage** : Base Supabase avec recherche vectorielle
6. **Statistiques** : Mise à jour des compteurs en temps réel

## 🔗 Configuration WhatsApp Business

### URL Webhook
```
https://your-app-name.onrender.com/webhook
```

### Verify Token
Utiliser la valeur de `WHATSAPP_VERIFY_TOKEN`

### Champs requis
- `messages` : Réception des messages texte
- `message_deliveries` : Statuts de livraison (optionnel)

## 🛠️ Développement Local

### Installation
```bash
cd whatsappserver
pip install -r requirements.txt
```

### Variables d'environnement
```bash
cp .env.example .env
# Éditer .env avec vos clés
```

### Lancement
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Test local avec ngrok
```bash
# Terminal 1
uvicorn app:app --reload

# Terminal 2  
ngrok http 8000
# Utiliser l'URL https://xxx.ngrok.io/webhook
```

## 📊 Monitoring

### Logs Render
- Accès aux logs en temps réel
- Filtrage par niveau (INFO, ERROR)
- Surveillance des erreurs de traitement

### Métriques importantes
- **Messages reçus** vs **messages traités**
- **Erreurs de traitement** 
- **Taux de détection de montres**
- **Performance des embeddings**

## 🔧 Maintenance

### Mise à jour du code
1. Commit les changements
2. Push vers GitHub
3. Redéploiement automatique Render

### Rotation des tokens
1. Renouveler les tokens WhatsApp Business
2. Mettre à jour les variables Render
3. Redémarrer le service

## ⚠️ Limitations Render

- **Build timeout** : 15 minutes max
- **Cold start** : ~30 secondes après inactivité  
- **Memory limit** : Selon le plan choisi
- **Concurrent connections** : Limitées selon le plan

## 🤝 Repository Client

Le bot Telegram et les outils de test sont dans un repository séparé :
`whatsapp-rag-client`

---

🔧 **Serveur optimisé pour la production avec sécurité renforcée**
