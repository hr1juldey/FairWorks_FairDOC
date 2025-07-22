-- Fairdoc AI Database Initialization
-- This file runs when PostgreSQL container starts

-- Create additional schemas if needed
CREATE SCHEMA IF NOT EXISTS fairdoc_ai_schema;

-- Create extensions for better performance
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE fairdoc_ai TO fairdoc_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO fairdoc_user;
GRANT ALL PRIVILEGES ON SCHEMA fairdoc_ai_schema TO fairdoc_user;

-- Set timezone
SET timezone = 'UTC';

-- Log initialization
SELECT 'Fairdoc AI database initialized successfully' AS message;
