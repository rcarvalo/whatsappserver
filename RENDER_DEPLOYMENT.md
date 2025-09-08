# 🚀 Déploiement Render - WhatsApp RAG Server

## 📋 Configuration Render

### **Build Command**
```bash
pip install -r requirements.txt
```

### **Start Command**  
```bash
python app.py
```

### **Environment Variables** (à configurer dans Render)

#### **🔐 Obligatoires pour le webhook**
```env
WHATSAPP_VERIFY_TOKEN=hellotesttoken
```

#### **⚡ Optionnelles pour le RAG complet**
```env
WHATSAPP_ACCESS_TOKEN=EAAc...
WHATSAPP_PHONE_NUMBER_ID=808...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=eyJ...
OPENAI_API_KEY=sk-...
```

#### **🔧 Optionnelles avancées**
```env
WHATSAPP_WEBHOOK_SECRET=your_secret
WHATSAPP_AUTO_RESPONSES=true
```

---

## 🎯 Modes de Fonctionnement

### **Mode Webhook Basique** ✅
- Variables requises : `WHATSAPP_VERIFY_TOKEN`
- Fonctionnalités : Réception messages, logs détaillés
- Idéal pour : Tests, développement

### **Mode RAG Complet** 🚀  
- Variables requises : Toutes les variables ci-dessus
- Fonctionnalités : RAG, Supabase, embeddings, recherche
- Idéal pour : Production

---

## 📊 Endpoints Disponibles

### **🏠 Page d'accueil**
```
GET / 
```

### **💚 Health Check**
```
GET /health
```

### **📈 Statistiques**
```
GET /stats
```

### **🪝 Webhook WhatsApp**
```
GET /webhook    # Vérification Facebook
POST /webhook   # Réception messages
```

### **🔍 Recherche RAG**
```
POST /search
{
  "query": "Comment créer un compte ?",
  "limit": 5
}
```

### **📚 Documentation Interactive**
```
GET /docs       # Swagger UI
GET /redoc      # ReDoc
```

---

## 🔧 Instructions de Déploiement

### **Étape 1 : Créer le Service**
1. Connectez votre repo GitHub sur Render
2. Sélectionnez **Web Service**
3. Repository : `rcarvalot/whatsappserver`

### **Étape 2 : Configuration**
```
Name: whatsapp-rag-python
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: python app.py
```

### **Étape 3 : Variables d'Environnement**
Ajoutez au minimum :
```
WHATSAPP_VERIFY_TOKEN = hellotesttoken
```

### **Étape 4 : Déploiement**
- Render déploiera automatiquement
- URL : `https://whatsapp-rag-python.onrender.com`

### **Étape 5 : Test**
```bash
curl https://votre-url.onrender.com/health
```

---

## 🧪 Tests Post-Déploiement

### **Test 1 : Health Check**
```bash
curl https://votre-url.onrender.com/health
```

### **Test 2 : Webhook Verification**
```bash
curl "https://votre-url.onrender.com/webhook?hub.mode=subscribe&hub.verify_token=hellotesttoken&hub.challenge=test123"
```
**Résultat attendu :** `test123`

### **Test 3 : Simulation Message**
```bash
curl -X POST https://votre-url.onrender.com/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "object": "whatsapp_business_account",
    "entry": [{
      "changes": [{
        "field": "messages",
        "value": {
          "metadata": {"phone_number_id": "test"},
          "messages": [{
            "id": "test_msg",
            "from": "33123456789", 
            "timestamp": "1683896400",
            "type": "text",
            "text": {"body": "Test message"}
          }]
        }
      }]
    }]
  }'
```

---

## 🔍 Monitoring & Logs

### **Logs Render**
- Accédez aux logs via le dashboard Render
- Recherchez les emoji pour identifier rapidement :
  - 🚀 Démarrage
  - ✅ Succès  
  - ❌ Erreurs
  - 📱 Messages WhatsApp

### **Métriques Importantes**
- Messages reçus : `/stats`
- Santé des composants : `/health`
- Erreurs : Logs Render

---

## 🆘 Troubleshooting

### **Problème : Build échoue**
```bash
# Vérifiez requirements.txt
pip install -r requirements.txt
```

### **Problème : App crash au démarrage**
```bash
# Variables d'environnement manquantes
# Le serveur démarre en mode basique même sans toutes les variables
```

### **Problème : Webhook 403**
```bash
# Token de vérification incorrect
# Vérifiez WHATSAPP_VERIFY_TOKEN dans Render
```

---

## 🔄 Mise à Jour

Pour mettre à jour :
1. Push sur GitHub
2. Render redéploie automatiquement  
3. Vérifiez les logs de déploiement
