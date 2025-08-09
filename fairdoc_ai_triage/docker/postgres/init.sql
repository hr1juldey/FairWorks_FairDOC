-- Fairdoc AI Database Initialization with pgvector
-- This file runs when PostgreSQL container starts

-- =============================================================================
-- EXTENSIONS & SCHEMAS
-- =============================================================================

-- Create pgvector extension for embeddings (medical similarity search)
CREATE EXTENSION IF NOT EXISTS vector;

-- Core PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS fairdoc_ai_schema;

-- =============================================================================
-- USER PERMISSIONS
-- =============================================================================

-- Grant comprehensive permissions
GRANT ALL PRIVILEGES ON DATABASE fairdoc_ai TO fairdoc_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO fairdoc_user;
GRANT ALL PRIVILEGES ON SCHEMA fairdoc_ai_schema TO fairdoc_user;
GRANT CREATE ON SCHEMA public TO fairdoc_user;

-- Grant usage on extensions
GRANT USAGE ON SCHEMA public TO fairdoc_user;

-- =============================================================================
-- ENUMS (Create before tables that reference them)
-- =============================================================================

-- Medical outcome enum for triage decisions
CREATE TYPE medical_outcome_enum AS ENUM (
    'emergency_route_to_doctor',
    'routine_doctor_consultation', 
    'self_care_advice',
    'need_more_questions',
    'spam_or_irrelevant'
);

-- Conversation status enum
CREATE TYPE conversation_status_enum AS ENUM (
    'new',
    'in_progress',
    'awaiting_response',
    'completed',
    'escalated',
    'abandoned'
);

-- Stakeholder role enum  
CREATE TYPE stakeholder_role_enum AS ENUM (
    'patient',
    'family_member',
    'doctor',
    'admin',
    'triage_agent'
);

-- Chat provider enum
CREATE TYPE chat_provider_enum AS ENUM (
    'raven',
    'telegram', 
    'whatsapp',
    'api_direct'
);

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- NICE Protocols table (referenced by application)
CREATE TABLE IF NOT EXISTS nice_protocols (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    protocol_code VARCHAR(50) NOT NULL UNIQUE,
    condition_name VARCHAR(200) NOT NULL,
    primary_symptoms JSONB NOT NULL DEFAULT '[]',
    red_flag_symptoms JSONB NOT NULL DEFAULT '[]',  
    initial_questions JSONB NOT NULL DEFAULT '[]',
    follow_up_questions JSONB NOT NULL DEFAULT '[]',
    emergency_criteria TEXT NOT NULL,
    routine_criteria TEXT NOT NULL,
    self_care_criteria TEXT NOT NULL,
    evidence_level VARCHAR(10) NOT NULL DEFAULT 'C',
    fhir_code VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Conversation states for V2 system
CREATE TABLE IF NOT EXISTS conversation_states_v2 (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stakeholder_role stakeholder_role_enum NOT NULL DEFAULT 'patient',
    stakeholder_id VARCHAR(100),
    chat_provider chat_provider_enum NOT NULL DEFAULT 'api_direct', 
    provider_metadata JSONB NOT NULL DEFAULT '{}',
    status conversation_status_enum NOT NULL DEFAULT 'new',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    patient_age INTEGER CHECK (patient_age >= 0 AND patient_age <= 120),
    patient_gender VARCHAR(20),
    final_outcome medical_outcome_enum,
    confidence_score REAL NOT NULL DEFAULT 0.0 CHECK (confidence_score >= 0 AND confidence_score <= 100),
    red_flags_detected JSONB NOT NULL DEFAULT '[]',
    requires_human_review BOOLEAN NOT NULL DEFAULT FALSE,
    is_emergency BOOLEAN NOT NULL DEFAULT FALSE,
    emergency_notified_at TIMESTAMP WITH TIME ZONE,
    turn_count INTEGER NOT NULL DEFAULT 1 CHECK (turn_count >= 1),
    total_processing_time_ms INTEGER,
    relevant_protocols JSONB NOT NULL DEFAULT '[]',
    conversation_turns JSONB NOT NULL DEFAULT '[]',
    model_version VARCHAR(20) NOT NULL DEFAULT 'v2.6-stable'
);

-- Gold standard dialogues for training
CREATE TABLE IF NOT EXISTS gold_standard_dialogues_v2 (
    standard_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    primary_symptom VARCHAR(100) NOT NULL,
    expected_outcome VARCHAR(50) NOT NULL,
    patient_age INTEGER NOT NULL CHECK (patient_age >= 0 AND patient_age <= 120),
    patient_gender VARCHAR(20) NOT NULL,
    expected_red_flags JSONB NOT NULL DEFAULT '[]',
    should_escalate BOOLEAN NOT NULL DEFAULT FALSE,
    conversation_dialogue JSONB NOT NULL,
    relevant_protocols JSONB NOT NULL DEFAULT '[]',
    minimum_confidence_threshold REAL NOT NULL DEFAULT 70.0 CHECK (minimum_confidence_threshold >= 0 AND minimum_confidence_threshold <= 100),
    expected_turn_count INTEGER NOT NULL DEFAULT 3,
    max_acceptable_turns INTEGER NOT NULL DEFAULT 8,
    created_by VARCHAR(100) NOT NULL,
    reviewed_by VARCHAR(100),
    clinical_notes TEXT,
    version VARCHAR(10) NOT NULL DEFAULT '1.0',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    CHECK (expected_turn_count >= 1 AND expected_turn_count <= max_acceptable_turns)
);

-- Medical embeddings table for semantic search
CREATE TABLE IF NOT EXISTS medical_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_type VARCHAR(50) NOT NULL, -- 'symptom', 'protocol', 'dialogue'
    content_id VARCHAR(100) NOT NULL,  -- references to other tables
    content_text TEXT NOT NULL,
    embedding vector(384), -- 384-dimensional embeddings (sentence-transformers)
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(content_type, content_id)
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- NICE Protocols indexes  
CREATE INDEX IF NOT EXISTS idx_nice_protocol_code ON nice_protocols(protocol_code);
CREATE INDEX IF NOT EXISTS idx_nice_condition ON nice_protocols(condition_name);
CREATE INDEX IF NOT EXISTS idx_nice_symptoms ON nice_protocols USING GIN(primary_symptoms);

-- Conversation indexes
CREATE INDEX IF NOT EXISTS idx_conversations_emergency ON conversation_states_v2(is_emergency, created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversation_states_v2(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_conversations_stakeholder ON conversation_states_v2(stakeholder_id, chat_provider);
CREATE INDEX IF NOT EXISTS idx_conversations_review ON conversation_states_v2(requires_human_review, created_at);

-- Gold standards indexes
CREATE INDEX IF NOT EXISTS idx_gold_standards_active ON gold_standard_dialogues_v2(is_active, primary_symptom);
CREATE INDEX IF NOT EXISTS idx_gold_standards_outcome ON gold_standard_dialogues_v2(expected_outcome, is_active);

-- Vector similarity index (HNSW for fast similarity search)
CREATE INDEX IF NOT EXISTS idx_medical_embeddings_vector ON medical_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_medical_embeddings_content ON medical_embeddings(content_type, content_id);

-- =============================================================================
-- FINAL CONFIGURATION
-- =============================================================================

-- Set timezone
SET timezone = 'UTC';

-- Grant table permissions to fairdoc_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fairdoc_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fairdoc_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO fairdoc_user;

-- Grant future permissions
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO fairdoc_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO fairdoc_user;

-- Log successful initialization
SELECT 'Fairdoc AI database with pgvector initialized successfully' AS message;
