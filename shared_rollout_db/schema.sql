PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=60000;

CREATE TABLE IF NOT EXISTS scenario(
  scenario_uid TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  code_config_fingerprint TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS state(
  state_uid TEXT PRIMARY KEY,
  scenario_uid TEXT NOT NULL REFERENCES scenario(scenario_uid),
  source_group TEXT,
  content_hash TEXT NOT NULL,
  physical_state_json TEXT,
  h0_json TEXT,
  goals_geometry_json TEXT,
  provenance_json TEXT NOT NULL,
  identity_quality TEXT NOT NULL CHECK(identity_quality IN ('CONTENT_EXACT','CONDITIONING_EXACT','SOURCE_GROUP_STABLE','ALIAS_ONLY')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS state_alias(
  scenario_uid TEXT NOT NULL,
  alias TEXT NOT NULL,
  state_uid TEXT NOT NULL REFERENCES state(state_uid),
  experiment_uid TEXT NOT NULL,
  PRIMARY KEY(scenario_uid,alias,state_uid,experiment_uid)
);

CREATE TABLE IF NOT EXISTS eta(
  eta_uid TEXT PRIMARY KEY,
  eta1 REAL NOT NULL, eta2 REAL NOT NULL, eta3 REAL NOT NULL,
  normalized_eta_json TEXT,
  canonical_hex TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS controller_config(
  controller_uid TEXT PRIMARY KEY,
  scenario_uid TEXT NOT NULL REFERENCES scenario(scenario_uid),
  flow_checkpoint_sha256 TEXT,
  orthoflow3_sha256 TEXT,
  safety_config_hash TEXT,
  horizon TEXT,
  dt TEXT,
  success_semantics_version TEXT,
  conditioning_version TEXT,
  rng_semantics_version TEXT,
  config_json TEXT NOT NULL,
  compatibility_quality TEXT NOT NULL CHECK(compatibility_quality IN ('EXACT_PROFILE','PARTIAL_PROFILE','AMBIGUOUS')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiment(
  experiment_uid TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  path TEXT NOT NULL,
  protocol_hash TEXT,
  code_hash TEXT,
  start_time TEXT,
  end_time TEXT,
  reused_rollout_count INTEGER NOT NULL DEFAULT 0,
  new_rollout_count INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL,
  UNIQUE(path)
);

CREATE TABLE IF NOT EXISTS source_file(
  source_uid TEXT PRIMARY KEY,
  experiment_uid TEXT NOT NULL REFERENCES experiment(experiment_uid),
  path TEXT NOT NULL UNIQUE,
  sha256 TEXT NOT NULL,
  file_type TEXT NOT NULL,
  classification TEXT NOT NULL,
  rows_seen INTEGER NOT NULL DEFAULT 0,
  rows_imported INTEGER NOT NULL DEFAULT 0,
  rows_duplicate INTEGER NOT NULL DEFAULT 0,
  rows_ambiguous INTEGER NOT NULL DEFAULT 0,
  ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rollout(
  rollout_uid TEXT PRIMARY KEY,
  state_uid TEXT NOT NULL REFERENCES state(state_uid),
  eta_uid TEXT NOT NULL REFERENCES eta(eta_uid),
  controller_uid TEXT NOT NULL REFERENCES controller_config(controller_uid),
  seed_key TEXT NOT NULL,
  continuation_seed_json TEXT NOT NULL,
  success INTEGER NOT NULL,
  deadlock INTEGER NOT NULL,
  timeout INTEGER NOT NULL,
  collision INTEGER NOT NULL,
  numerical_failure INTEGER NOT NULL,
  episode_length REAL,
  j_def REAL,
  min_wall_distance REAL,
  min_agent_distance REAL,
  outcome TEXT,
  experiment_uid TEXT NOT NULL REFERENCES experiment(experiment_uid),
  original_source_file TEXT NOT NULL,
  timestamp TEXT,
  compatibility_quality TEXT NOT NULL,
  raw_record_hash TEXT NOT NULL,
  conflict_quarantined INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(state_uid,eta_uid,controller_uid,seed_key)
);
CREATE INDEX IF NOT EXISTS idx_rollout_lookup ON rollout(state_uid,eta_uid,controller_uid,seed_key,conflict_quarantined);
CREATE INDEX IF NOT EXISTS idx_rollout_eta ON rollout(eta_uid,state_uid);
CREATE TABLE IF NOT EXISTS rollout_source(
  rollout_uid TEXT NOT NULL REFERENCES rollout(rollout_uid),
  source_uid TEXT NOT NULL REFERENCES source_file(source_uid),
  source_line INTEGER,
  PRIMARY KEY(rollout_uid,source_uid,source_line)
);

CREATE TABLE IF NOT EXISTS aggregate_evidence(
  aggregate_uid TEXT PRIMARY KEY,
  state_uid TEXT NOT NULL REFERENCES state(state_uid),
  eta_uid TEXT NOT NULL REFERENCES eta(eta_uid),
  controller_uid TEXT NOT NULL REFERENCES controller_config(controller_uid),
  n_trials INTEGER NOT NULL,
  n_success INTEGER NOT NULL,
  n_deadlock INTEGER,
  n_timeout INTEGER,
  n_collision INTEGER,
  seed_identities_known INTEGER NOT NULL,
  evidence_class TEXT NOT NULL,
  certified_b15 INTEGER NOT NULL,
  provenance_json TEXT NOT NULL,
  experiment_uid TEXT NOT NULL REFERENCES experiment(experiment_uid),
  original_source_file TEXT NOT NULL,
  conflict_quarantined INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_aggregate_lookup ON aggregate_evidence(state_uid,eta_uid,controller_uid,conflict_quarantined);
CREATE TABLE IF NOT EXISTS robust_logical_decision(
  decision_uid TEXT PRIMARY KEY,
  state_uid TEXT NOT NULL REFERENCES state(state_uid),
  eta_uid TEXT NOT NULL REFERENCES eta(eta_uid),
  controller_uid TEXT NOT NULL REFERENCES controller_config(controller_uid),
  protocol_name TEXT NOT NULL,
  canonical_seed_order_json TEXT NOT NULL,
  evaluated_seed_count INTEGER NOT NULL,
  n_success INTEGER NOT NULL,
  n_failure INTEGER NOT NULL,
  stop_reason TEXT NOT NULL CHECK(stop_reason IN (
    'ROBUST_IMPOSSIBLE_2_FAILURES',
    'ROBUST_CONFIRMED_15_SUCCESSES',
    'FULL_16_EVALUATED'
  )),
  robust INTEGER NOT NULL,
  stage TEXT NOT NULL,
  experiment_uid TEXT NOT NULL REFERENCES experiment(experiment_uid),
  provenance_json TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(state_uid,eta_uid,controller_uid,protocol_name)
);
CREATE INDEX IF NOT EXISTS idx_robust_logical_lookup
  ON robust_logical_decision(state_uid,eta_uid,controller_uid,protocol_name);
CREATE TABLE IF NOT EXISTS aggregate_source(
  aggregate_uid TEXT NOT NULL REFERENCES aggregate_evidence(aggregate_uid),
  source_uid TEXT NOT NULL REFERENCES source_file(source_uid),
  source_line INTEGER,
  PRIMARY KEY(aggregate_uid,source_uid,source_line)
);

CREATE TABLE IF NOT EXISTS conflict(
  conflict_uid TEXT PRIMARY KEY,
  entity_type TEXT NOT NULL,
  identity_key TEXT NOT NULL,
  existing_json TEXT NOT NULL,
  incoming_json TEXT NOT NULL,
  source_file TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'CONFLICT_QUARANTINED',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE VIEW IF NOT EXISTS state_eta_seed_coverage AS
SELECT r.state_uid,r.eta_uid,r.controller_uid,COUNT(*) n_trials,
 SUM(r.success) n_success,SUM(r.deadlock) n_deadlock,SUM(r.timeout) n_timeout,SUM(r.collision) n_collision
FROM rollout r WHERE r.conflict_quarantined=0 AND r.numerical_failure=0
GROUP BY r.state_uid,r.eta_uid,r.controller_uid;
