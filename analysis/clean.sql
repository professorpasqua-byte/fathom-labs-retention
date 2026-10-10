-- Fathom Labs Researcher Retention — cleaning pipeline
-- Run against the raw export (fathom_labs_researchers_raw.csv loaded into
-- a `researchers_raw` table) to reproduce fathom_labs_researchers_clean.csv.

-- 1. Deduplicate on researcher_id, keeping the most complete row.
WITH deduped AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY researcher_id
               ORDER BY (manager_support IS NOT NULL)
                      + (org_justice_perception IS NOT NULL)
                      + (workload_hours_week IS NOT NULL) DESC
           ) AS rn
    FROM researchers_raw
),

-- 2. Normalize inconsistent casing in categorical fields.
normalized AS (
    SELECT
        researcher_id,
        name,
        INITCAP(role) AS role,
        department,
        tenure_months,
        hire_date,
        moral_injury_index,
        escalation_confidence,
        pace_vs_safety_tension,
        org_justice_perception,
        manager_support,
        workload_hours_week,
        equity_vested_pct,
        left_company
    FROM deduped
    WHERE rn = 1
),

-- 3. Impute missing Likert-scale fields with the role-level median rather
--    than dropping rows — losing a researcher's record because one survey
--    field was missing would bias the sample toward whoever filled out
--    every field, which is its own confound.
imputed AS (
    SELECT
        n.*,
        COALESCE(n.manager_support,
                 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY manager_support)
                     OVER (PARTITION BY n.role)) AS manager_support_filled,
        COALESCE(n.org_justice_perception,
                 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY org_justice_perception)
                     OVER (PARTITION BY n.role)) AS org_justice_filled,
        COALESCE(n.workload_hours_week,
                 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY workload_hours_week)
                     OVER (PARTITION BY n.role)) AS workload_hours_filled
    FROM normalized n
)

SELECT
    researcher_id,
    name,
    role,
    department,
    tenure_months,
    hire_date,
    moral_injury_index,
    escalation_confidence,
    pace_vs_safety_tension,
    org_justice_filled   AS org_justice_perception,
    manager_support_filled AS manager_support,
    workload_hours_filled  AS workload_hours_week,
    equity_vested_pct,
    left_company
FROM imputed
ORDER BY researcher_id;
