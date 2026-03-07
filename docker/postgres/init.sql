-- ============================================================
-- PostgreSQL Init Script
-- Runs automatically on first container startup.
-- ============================================================

-- 1. Enable pgvector extension for semantic similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Enable pg_trgm for fast text/trigram search (useful for fuzzy matching)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 3. Enable uuid-ossp for UUID generation (fallback if app doesn't generate)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Log confirmation
DO $$
BEGIN
    RAISE NOTICE '✅  Extensions enabled: vector, pg_trgm, uuid-ossp';
END $$;
