-- ===========================================
-- Mise à jour du schéma pour métadonnées enrichies
-- ===========================================

-- Ajouter des colonnes pour les métadonnées WhatsApp enrichies
ALTER TABLE watch_conversations 
ADD COLUMN IF NOT EXISTS sender_formatted_name TEXT,
ADD COLUMN IF NOT EXISTS sender_wa_id TEXT,
ADD COLUMN IF NOT EXISTS business_phone_number_id TEXT,
ADD COLUMN IF NOT EXISTS display_phone_number TEXT,
ADD COLUMN IF NOT EXISTS is_group_message BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS context_reply_to TEXT,
ADD COLUMN IF NOT EXISTS context_message_id TEXT,
ADD COLUMN IF NOT EXISTS webhook_entry_id TEXT,
ADD COLUMN IF NOT EXISTS webhook_timestamp BIGINT;

-- Index pour améliorer les performances des requêtes sur les nouvelles colonnes
CREATE INDEX IF NOT EXISTS idx_watch_conversations_sender_wa_id ON watch_conversations(sender_wa_id);
CREATE INDEX IF NOT EXISTS idx_watch_conversations_is_group ON watch_conversations(is_group_message);
CREATE INDEX IF NOT EXISTS idx_watch_conversations_business_phone ON watch_conversations(business_phone_number_id);
CREATE INDEX IF NOT EXISTS idx_watch_conversations_sentiment ON watch_conversations(sentiment_score);
CREATE INDEX IF NOT EXISTS idx_watch_conversations_urgency ON watch_conversations(urgency_level);

-- Index composite pour les recherches par groupe et type de message
CREATE INDEX IF NOT EXISTS idx_watch_conversations_group_type ON watch_conversations(is_group_message, message_type, message_timestamp);

-- Index sur les mots-clés extraits (array)
CREATE INDEX IF NOT EXISTS idx_watch_conversations_keywords ON watch_conversations USING GIN(extracted_keywords);

-- Index JSON pour les métadonnées de recherche enrichies
CREATE INDEX IF NOT EXISTS idx_watch_conversations_search_metadata ON watch_conversations USING GIN(search_metadata);
CREATE INDEX IF NOT EXISTS idx_watch_conversations_detailed_extraction ON watch_conversations USING GIN(detailed_extraction);

-- Fonction pour rechercher par groupes avec métadonnées enrichies
CREATE OR REPLACE FUNCTION search_group_messages(
    group_filter TEXT DEFAULT NULL,
    sender_filter TEXT DEFAULT NULL,
    intent_filter TEXT DEFAULT NULL,
    days_back INTEGER DEFAULT 30,
    limit_results INTEGER DEFAULT 50
)
RETURNS TABLE (
    id BIGINT,
    message_content TEXT,
    watch_brand TEXT,
    watch_model TEXT,
    price_mentioned DECIMAL,
    group_name TEXT,
    sender TEXT,
    sender_formatted_name TEXT,
    is_group_message BOOLEAN,
    message_type TEXT,
    intent_score FLOAT,
    sentiment_score FLOAT,
    urgency_level INTEGER,
    extracted_keywords TEXT[],
    message_timestamp TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.message_content,
        c.watch_brand,
        c.watch_model,
        c.price_mentioned,
        c.group_name,
        c.sender,
        c.sender_formatted_name,
        c.is_group_message,
        c.message_type,
        c.intent_score,
        c.sentiment_score,
        c.urgency_level,
        c.extracted_keywords,
        c.message_timestamp
    FROM watch_conversations c
    WHERE 
        c.message_timestamp > NOW() - INTERVAL '1 day' * days_back
        AND (group_filter IS NULL OR c.group_name ILIKE '%' || group_filter || '%')
        AND (sender_filter IS NULL OR (c.sender ILIKE '%' || sender_filter || '%' OR c.sender_formatted_name ILIKE '%' || sender_filter || '%'))
        AND (intent_filter IS NULL OR c.message_type = intent_filter)
    ORDER BY c.message_timestamp DESC
    LIMIT limit_results;
END;
$$;

-- Fonction pour analyser les patterns de groupes
CREATE OR REPLACE FUNCTION analyze_group_patterns(
    days_back INTEGER DEFAULT 30
)
RETURNS TABLE (
    group_name TEXT,
    message_count BIGINT,
    unique_senders BIGINT,
    avg_intent_score DECIMAL,
    top_watch_brands TEXT[],
    total_value DECIMAL,
    avg_urgency DECIMAL,
    last_activity TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.group_name,
        COUNT(*)::BIGINT as message_count,
        COUNT(DISTINCT c.sender_wa_id)::BIGINT as unique_senders,
        AVG(c.intent_score)::DECIMAL as avg_intent_score,
        ARRAY_AGG(DISTINCT c.watch_brand ORDER BY c.watch_brand)
            FILTER (WHERE c.watch_brand IS NOT NULL) as top_watch_brands,
        COALESCE(SUM(c.price_mentioned), 0)::DECIMAL as total_value,
        AVG(c.urgency_level)::DECIMAL as avg_urgency,
        MAX(c.message_timestamp) as last_activity
    FROM watch_conversations c
    WHERE 
        c.message_timestamp > NOW() - INTERVAL '1 day' * days_back
        AND c.is_group_message = TRUE
        AND c.group_name IS NOT NULL
    GROUP BY c.group_name
    HAVING COUNT(*) >= 3  -- Seulement les groupes avec au moins 3 messages
    ORDER BY message_count DESC, last_activity DESC;
END;
$$;

-- Fonction pour l'analyse de sentiment enrichie
CREATE OR REPLACE FUNCTION analyze_sentiment_trends(
    brand_filter TEXT DEFAULT NULL,
    days_back INTEGER DEFAULT 30
)
RETURNS TABLE (
    time_period DATE,
    avg_sentiment DECIMAL,
    message_count BIGINT,
    positive_messages BIGINT,
    negative_messages BIGINT,
    neutral_messages BIGINT,
    avg_price DECIMAL,
    watch_brand TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        DATE(c.message_timestamp) as time_period,
        AVG(c.sentiment_score)::DECIMAL as avg_sentiment,
        COUNT(*)::BIGINT as message_count,
        SUM(CASE WHEN c.sentiment_score > 0.2 THEN 1 ELSE 0 END)::BIGINT as positive_messages,
        SUM(CASE WHEN c.sentiment_score < -0.2 THEN 1 ELSE 0 END)::BIGINT as negative_messages,
        SUM(CASE WHEN c.sentiment_score BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END)::BIGINT as neutral_messages,
        AVG(c.price_mentioned)::DECIMAL as avg_price,
        c.watch_brand
    FROM watch_conversations c
    WHERE 
        c.message_timestamp > NOW() - INTERVAL '1 day' * days_back
        AND c.sentiment_score IS NOT NULL
        AND (brand_filter IS NULL OR c.watch_brand ILIKE '%' || brand_filter || '%')
        AND c.watch_brand IS NOT NULL
    GROUP BY DATE(c.message_timestamp), c.watch_brand
    HAVING COUNT(*) >= 2
    ORDER BY time_period DESC, avg_sentiment DESC;
END;
$$;

-- Vue pour les statistiques enrichies en temps réel
CREATE OR REPLACE VIEW enriched_watch_analytics AS
SELECT 
    -- Statistiques générales
    COUNT(*) as total_messages,
    COUNT(DISTINCT sender_wa_id) as unique_senders,
    COUNT(DISTINCT group_name) FILTER (WHERE is_group_message = TRUE) as active_groups,
    
    -- Analyse des intentions
    COUNT(*) FILTER (WHERE message_type = 'sale') as sales_messages,
    COUNT(*) FILTER (WHERE message_type = 'wanted') as wanted_messages,
    COUNT(*) FILTER (WHERE message_type = 'question') as question_messages,
    AVG(intent_score) as avg_intent_confidence,
    
    -- Analyse des montres
    COUNT(DISTINCT watch_brand) FILTER (WHERE watch_brand IS NOT NULL) as brands_mentioned,
    COUNT(*) FILTER (WHERE watch_brand IS NOT NULL) as watch_messages,
    AVG(price_mentioned) FILTER (WHERE price_mentioned IS NOT NULL) as avg_price,
    SUM(price_mentioned) FILTER (WHERE message_type = 'sale' AND price_mentioned IS NOT NULL) as total_sales_value,
    
    -- Analyse de sentiment
    AVG(sentiment_score) as avg_sentiment,
    COUNT(*) FILTER (WHERE sentiment_score > 0.2) as positive_sentiment_count,
    COUNT(*) FILTER (WHERE sentiment_score < -0.2) as negative_sentiment_count,
    
    -- Analyse d'urgence
    AVG(urgency_level) as avg_urgency,
    COUNT(*) FILTER (WHERE urgency_level >= 3) as urgent_messages,
    
    -- Analyse temporelle
    COUNT(*) FILTER (WHERE message_timestamp > NOW() - INTERVAL '24 hours') as messages_24h,
    COUNT(*) FILTER (WHERE message_timestamp > NOW() - INTERVAL '7 days') as messages_7d,
    
    -- Métadonnées de mise à jour
    NOW() as computed_at
FROM watch_conversations
WHERE message_timestamp > NOW() - INTERVAL '30 days';

-- Commentaires sur les nouvelles fonctionnalités
COMMENT ON COLUMN watch_conversations.sender_formatted_name IS 'Nom formaté de l''expéditeur depuis WhatsApp';
COMMENT ON COLUMN watch_conversations.sender_wa_id IS 'ID WhatsApp de l''expéditeur';
COMMENT ON COLUMN watch_conversations.is_group_message IS 'Indique si le message provient d''un groupe';
COMMENT ON COLUMN watch_conversations.context_reply_to IS 'ID du message auquel on répond';
COMMENT ON COLUMN watch_conversations.extracted_keywords IS 'Mots-clés enrichis pour la recherche sémantique';
COMMENT ON COLUMN watch_conversations.sentiment_score IS 'Score de sentiment (-1 à 1)';
COMMENT ON COLUMN watch_conversations.search_metadata IS 'Métadonnées enrichies style Azure Search pour optimiser la recherche';

COMMENT ON FUNCTION search_group_messages IS 'Recherche avancée dans les messages de groupes avec filtres enrichis';
COMMENT ON FUNCTION analyze_group_patterns IS 'Analyse les patterns d''activité des groupes WhatsApp';
COMMENT ON FUNCTION analyze_sentiment_trends IS 'Analyse les tendances de sentiment par marque et période';
COMMENT ON VIEW enriched_watch_analytics IS 'Vue analytique enrichie pour les métriques en temps réel';
