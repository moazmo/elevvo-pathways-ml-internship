-- Initialize Walmart Forecasting Database
-- This script creates the necessary tables for the application

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    store_id INTEGER NOT NULL,
    dept_id INTEGER NOT NULL,
    prediction_date DATE NOT NULL,
    predicted_sales DECIMAL(12, 2) NOT NULL,
    confidence_lower DECIMAL(12, 2),
    confidence_upper DECIMAL(12, 2),
    model_used VARCHAR(100),
    input_features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, dept_id, prediction_date, created_at)
);

-- Create user sessions table (for future authentication)
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create system metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(12, 4),
    metric_data JSONB,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create batch jobs table
CREATE TABLE IF NOT EXISTS batch_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    input_data JSONB,
    results JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_predictions_store_dept_date ON predictions(store_id, dept_id, prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON system_metrics(metric_name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status);

-- Insert initial system metrics
INSERT INTO system_metrics (metric_name, metric_value, metric_data) VALUES
('system_initialized', 1, '{"version": "1.0.0", "initialized_at": "' || CURRENT_TIMESTAMP || '"}'),
('models_loaded', 6, '{"model_types": ["weighted_ensemble", "xgboost", "lightgbm", "random_forest", "linear_regression", "prophet"]}'),
('features_available', 89, '{"feature_engineering": "complete", "preprocessing": "ready"}');

-- Create a view for recent predictions
CREATE OR REPLACE VIEW recent_predictions AS
SELECT 
    p.*,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - p.created_at))/3600 as hours_ago
FROM predictions p
WHERE p.created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY p.created_at DESC;

COMMENT ON TABLE predictions IS 'Stores all sales predictions made by the system';
COMMENT ON TABLE user_sessions IS 'Manages user sessions and authentication data';
COMMENT ON TABLE system_metrics IS 'Tracks system performance and health metrics';
COMMENT ON TABLE batch_jobs IS 'Manages batch prediction jobs and their status';