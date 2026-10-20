# 开发前提议集核查

{
  "utc": "2026-10-18T19:53:11.285044+00:00",
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
    "chunking": {
      "type_equal": {
        "diagonal_oracle_pct": 95.1388888888889,
        "non_diagonal_oracle_pct": 97.22222222222221,
        "diagonal_minus_non_diagonal_pp": -2.083333333333333,
        "unknown_bounds_pp": [
          -2.083333333333333,
          -2.083333333333333
        ],
        "CI95_pp": [
          -5.208333333333334,
          0.3472222222222222
        ],
        "human_step500_reference_pct": 94.44444444444443,
        "evidence": "SAMPLED_ACTIONS_IN_SAMPLE_SEED0"
      }
    },
    "cvae": {
      "type_equal": {
        "diagonal_oracle_pct": 87.50000000000001,
        "non_diagonal_oracle_pct": 95.13888888888889,
        "diagonal_minus_non_diagonal_pp": -7.638888888888888,
        "unknown_bounds_pp": [
          -7.638888888888888,
          -7.638888888888888
        ],
        "CI95_pp": [
          -13.194444444444445,
          -2.430555555555556
        ],
        "human_step500_reference_pct": 91.66666666666669,
        "evidence": "SAMPLED_ACTIONS_IN_SAMPLE_SEED0"
      }
    },
    "deterministic": {
      "type_equal": {
        "diagonal_oracle_pct": 87.84722222222223,
        "non_diagonal_oracle_pct": 94.44444444444443,
        "diagonal_minus_non_diagonal_pp": -6.597222222222222,
        "unknown_bounds_pp": [
          -6.597222222222222,
          -6.250000000000002
        ],
        "CI95_pp": [
          -11.805555555555555,
          -1.7361111111111112
        ],
        "human_step500_reference_pct": 89.23611111111111,
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
    "gaussian": {
      "type_equal": {
        "diagonal_oracle_pct": 87.15277777777779,
        "non_diagonal_oracle_pct": 94.09722222222221,
        "diagonal_minus_non_diagonal_pp": -6.944444444444444,
        "unknown_bounds_pp": [
          -7.291666666666667,
          -6.25
        ],
        "CI95_pp": [
          -13.88888888888889,
          -1.0416666666666665
        ],
        "human_step500_reference_pct": 95.13888888888889,
        "evidence": "SAMPLED_ACTIONS_IN_SAMPLE_SEED0"
      }
    },
    "mixture": {
      "type_equal": {
        "diagonal_oracle_pct": 90.97222222222221,
        "non_diagonal_oracle_pct": 94.44444444444443,
        "diagonal_minus_non_diagonal_pp": -3.472222222222222,
        "unknown_bounds_pp": [
          -3.8194444444444446,
          -3.125
        ],
        "CI95_pp": [
          -8.333333333333332,
          0.6944444444444444
        ],
        "human_step500_reference_pct": 96.18055555555554,
        "evidence": "SAMPLED_ACTIONS_IN_SAMPLE_SEED0"
      }
    },
    "recurrent": {
      "type_equal": {
        "diagonal_oracle_pct": 92.0138888888889,
        "non_diagonal_oracle_pct": 97.56944444444446,
        "diagonal_minus_non_diagonal_pp": -5.555555555555556,
        "unknown_bounds_pp": [
          -5.555555555555556,
          -5.208333333333334
        ],
        "CI95_pp": [
          -10.76388888888889,
          -1.3888888888888888
        ],
        "human_step500_reference_pct": 96.18055555555554,
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
        "diagonal_oracle_pct": 90.95017636684304,
        "non_diagonal_oracle_pct": 95.74514991181657,
        "diagonal_minus_non_diagonal_pp": -4.794973544973544,
        "unknown_bounds_pp": [
          -4.971340388007055,
          -4.403659611992946
        ],
        "CI95_pp": [
          -7.809882054673722,
          -2.083333333333333
        ],
        "human_step500_reference_pct": 94.3617724867725,
        "evidence": "SAMPLED_ACTIONS_IN_SAMPLE_SEED0"
      },
      "controller_equal": {
        "diagonal_oracle_pct": 89.31451612903226,
        "non_diagonal_oracle_pct": 94.89247311827957,
        "diagonal_minus_non_diagonal_pp": -5.577956989247312,
        "unknown_bounds_pp": [
          -5.846774193548387,
          -5.006720430107527
        ],
        "CI95_pp": null,
        "human_step500_reference_pct": 94.32123655913978,
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
    "chunking": {
      "total_drawn_schedules": 5184,
      "supported_at200": 5145,
      "fraction_at200": 0.3230320699708455,
      "supported_at_actual_switch": 2919,
      "fraction_at_actual_switch": 0.223706748886605,
      "limitation": "Conditional on actually observed fixed-action traces and known late outcome; no post-termination trajectory imputation or causal claim"
    },
    "cvae": {
      "total_drawn_schedules": 5184,
      "supported_at200": 5160,
      "fraction_at200": 0.24011627906976743,
      "supported_at_actual_switch": 3780,
      "fraction_at_actual_switch": 0.17539682539682538,
      "limitation": "Conditional on actually observed fixed-action traces and known late outcome; no post-termination trajectory imputation or causal claim"
    },
    "deterministic": {
      "total_drawn_schedules": 5184,
      "supported_at200": 5134,
      "fraction_at200": 0.2415270744059213,
      "supported_at_actual_switch": 3871,
      "fraction_at_actual_switch": 0.17592353397055024,
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
    "gaussian": {
      "total_drawn_schedules": 5184,
      "supported_at200": 5137,
      "fraction_at200": 0.23087405100253067,
      "supported_at_actual_switch": 4014,
      "fraction_at_actual_switch": 0.17289486796213255,
      "limitation": "Conditional on actually observed fixed-action traces and known late outcome; no post-termination trajectory imputation or causal claim"
    },
    "mixture": {
      "total_drawn_schedules": 5184,
      "supported_at200": 5137,
      "fraction_at200": 0.22737006034650575,
      "supported_at_actual_switch": 4095,
      "fraction_at_actual_switch": 0.17094017094017094,
      "limitation": "Conditional on actually observed fixed-action traces and known late outcome; no post-termination trajectory imputation or causal claim"
    },
    "recurrent": {
      "total_drawn_schedules": 5184,
      "supported_at200": 5154,
      "fraction_at200": 0.2811408614668219,
      "supported_at_actual_switch": 3230,
      "fraction_at_actual_switch": 0.19411764705882353,
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
