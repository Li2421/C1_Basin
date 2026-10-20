# 开发前提议集核查

{
  "utc": "2026-10-18T19:00:00.498448+00:00",
  "evidence": "DEVELOPMENT",
  "model_used": false,
  "summaries": {
    "Flow": {
      "type_equal": {
        "diagonal_oracle_pct": 78.27380952380952,
        "non_diagonal_oracle_pct": 89.13690476190476,
        "diagonal_minus_non_diagonal_pp": -10.863095238095237,
        "unknown_bounds_pp": [
          -11.755952380952381,
          -9.077380952380953
        ],
        "CI95_pp": [
          -17.559523809523807,
          -3.869047619047619
        ],
        "human_step500_reference_pct": 94.04761904761905,
        "evidence": "SAMPLED_ACTIONS_IN_SAMPLE_SEED0"
      }
    },
    "diffusion": {
      "type_equal": {
        "diagonal_oracle_pct": 100.0,
        "non_diagonal_oracle_pct": 100.0,
        "diagonal_minus_non_diagonal_pp": 0.0,
        "unknown_bounds_pp": [
          0.0,
          0.0
        ],
        "CI95_pp": [
          0.0,
          0.0
        ],
        "human_step500_reference_pct": 93.05555555555554,
        "evidence": "SAMPLED_ACTIONS_IN_SAMPLE_SEED0"
      }
    },
    "ring_ippo": {
      "type_equal": {
        "diagonal_oracle_pct": 99.65277777777779,
        "non_diagonal_oracle_pct": 99.65277777777779,
        "diagonal_minus_non_diagonal_pp": 0.0,
        "unknown_bounds_pp": [
          0.0,
          0.0
        ],
        "CI95_pp": [
          0.0,
          0.0
        ],
        "human_step500_reference_pct": 99.30555555555554,
        "evidence": "SAMPLED_ACTIONS_IN_SAMPLE_SEED0"
      }
    },
    "combined": {
      "type_equal": {
        "diagonal_oracle_pct": 92.64219576719577,
        "non_diagonal_oracle_pct": 96.2632275132275,
        "diagonal_minus_non_diagonal_pp": -3.6210317460317456,
        "unknown_bounds_pp": [
          -3.9186507936507935,
          -3.0257936507936507
        ],
        "CI95_pp": [
          -11.111111111111112,
          0.0
        ],
        "human_step500_reference_pct": 95.46957671957671,
        "evidence": "SAMPLED_ACTIONS_IN_SAMPLE_SEED0"
      },
      "controller_equal": {
        "diagonal_oracle_pct": 88.22115384615384,
        "non_diagonal_oracle_pct": 94.07051282051282,
        "diagonal_minus_non_diagonal_pp": -5.849358974358975,
        "unknown_bounds_pp": [
          -6.330128205128204,
          -4.887820512820513
        ],
        "CI95_pp": null,
        "human_step500_reference_pct": 95.03205128205127,
        "evidence": "SAMPLED_ACTIONS_IN_SAMPLE_SEED0"
      }
    }
  },
  "stage_complementarity": {
    "Flow": {
      "total_drawn_schedules": 12096,
      "supported_at200": 11998,
      "fraction_at200": 0.2187031171861977,
      "supported_at_actual_switch": 9275,
      "fraction_at_actual_switch": 0.12291105121293801,
      "limitation": "Conditional on actually observed fixed-action traces and known late outcome; no post-termination trajectory imputation or causal claim"
    },
    "diffusion": {
      "total_drawn_schedules": 5184,
      "supported_at200": 5142,
      "fraction_at200": 0.33741734733566703,
      "supported_at_actual_switch": 3456,
      "fraction_at_actual_switch": 0.2925347222222222,
      "limitation": "Conditional on actually observed fixed-action traces and known late outcome; no post-termination trajectory imputation or causal claim"
    },
    "ring_ippo": {
      "total_drawn_schedules": 5184,
      "supported_at200": 5089,
      "fraction_at200": 0.46787188052662604,
      "supported_at_actual_switch": 1152,
      "fraction_at_actual_switch": 0.3923611111111111,
      "limitation": "Conditional on actually observed fixed-action traces and known late outcome; no post-termination trajectory imputation or causal claim"
    }
  },
  "proposal_distances_to_human_references": {
    "13": {
      "all_proposal_distance_quantiles": [
        0.1188845421932705,
        0.2822320043406236,
        0.31892961574098644,
        0.3717342169092139,
        0.5617110189219489
      ],
      "nearest_per_state_quantiles": [
        0.1188845421932705,
        0.2174759918998045,
        0.23604199979409518,
        0.25566058695646443,
        0.2970178608846688
      ]
    },
    "10": {
      "all_proposal_distance_quantiles": [
        0.2293312070933943,
        0.4943862069150058,
        0.5600180793947314,
        0.6262008242276671,
        0.8605069029038417
      ],
      "nearest_per_state_quantiles": [
        0.2293312070933943,
        0.35653568329640367,
        0.40432390384381434,
        0.47292572525797344,
        0.556165786348147
      ]
    }
  },
  "training_crossfit": "Unavailable: only seed0 for pure generated schedule labels",
  "pilot_does_not_change_subsequent_design": true
}
