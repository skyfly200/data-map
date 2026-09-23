-- Migration date: 2026-09-21
-- MaxEnt Modeling Suite Schema
-- Version 1.1

-- 1. Model Configurations
-- Stores the 'recipe' for a model run.
CREATE TABLE model_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    
    -- The predictors used for this model (stored as a JSON array of keys)
    predictors JSONB NOT NULL, 
    
    -- Background sampling configuration
    background_count INTEGER NOT NULL DEFAULT 1000,
    effort_weighted BOOLEAN NOT NULL DEFAULT true,
    
    -- Region for projection (GeoJSON or BBox)
    projection_region JSONB,
    
    -- The source dataset used for presence points
    source_dataset_id UUID REFERENCES saved_datasets(id) ON DELETE SET NULL,
    
    visibility TEXT NOT NULL DEFAULT 'private',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. Model Runs
-- Links a configuration to a specific Earth Engine job execution.
CREATE TABLE model_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_id UUID NOT NULL REFERENCES model_configs(id) ON DELETE CASCADE,
    job_id TEXT NOT NULL UNIQUE, -- The GEE Job ID
    
    status TEXT NOT NULL DEFAULT 'pending', -- pending, running, succeeded, failed
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    
    error_message TEXT,
    
    -- Metadata about the run (e.g., actual point counts used)
    run_meta JSONB,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3. Model Results
-- Stores the calculated metrics and paths to the generated assets.
CREATE TABLE model_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    config_id UUID NOT NULL REFERENCES model_configs(id) ON DELETE CASCADE,
    
    -- Metrics
    auc FLOAT,
    auc_sd FLOAT,
    grade TEXT, -- 'excellent', 'good', 'fair', 'weak'
    
    -- Asset paths
    suitability_asset_path TEXT NOT NULL, -- Path to the EE Image asset
    results_json_path TEXT, -- Path to stored JSON (Response curves, etc.)
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(run_id)
);

-- Indexes for performance
CREATE INDEX idx_model_configs_owner ON model_configs(owner_id);
CREATE INDEX idx_model_runs_job_id ON model_runs(job_id);
CREATE INDEX idx_model_results_config ON model_results(config_id);

-- RLS Policies (Standard for the project)
ALTER TABLE model_configs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can manage their own model configs" 
    ON model_configs FOR ALL USING (auth.uid() = owner_id);
CREATE POLICY "Public models are readable by all" 
    ON model_configs FOR SELECT USING (visibility = 'public');

ALTER TABLE model_runs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view runs of their own configs" 
    ON model_runs FOR SELECT USING (
        EXISTS (SELECT 1 FROM model_configs WHERE id = model_runs.config_id AND owner_id = auth.uid())
    );

ALTER TABLE model_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view results of their own configs" 
    ON model_results FOR SELECT USING (
        EXISTS (SELECT 1 FROM model_configs WHERE id = model_results.config_id AND owner_id = auth.uid())
    );
