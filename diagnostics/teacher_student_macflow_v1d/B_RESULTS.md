# 阶段 B

{
  "selected": {
    "utc": "2026-10-15T13:00:52.760540+00:00",
    "students": {
      "rule": {
        "mode": "progress",
        "steps": 40,
        "speed": 0.25
      },
      "grid_index": 18,
      "tuning_lower": 0.125,
      "tuning_upper": 0.125,
      "selection_ambiguous_with_unknown": false
    },
    "G": {
      "rule": {
        "mode": "always",
        "steps": 0,
        "speed": 0.1
      },
      "grid_index": 0,
      "tuning_lower": 1.0,
      "tuning_upper": 1.0,
      "selection_ambiguous_with_unknown": false
    },
    "main_labels_read": false
  },
  "students_RHR": {
    "mean": 0.11272321428571429,
    "unknown": 0,
    "identification_interval": [
      0.11272321428571429,
      0.11272321428571429
    ],
    "CI95": [
      0.0,
      0.27901785714285715
    ],
    "CI_unknown_envelope": [
      0.0,
      0.27901785714285715
    ]
  },
  "G_RHR": {
    "mean": 1.0,
    "unknown": 0,
    "identification_interval": [
      1.0,
      1.0
    ],
    "CI95": [
      1.0,
      1.0
    ],
    "CI_unknown_envelope": [
      1.0,
      1.0
    ]
  },
  "students_S500_minus_RHR": {
    "mean": 0.8292410714285714,
    "CI95": [
      0.6606863839285714,
      0.9564732142857143
    ],
    "identification_interval": [
      0.8292410714285714,
      0.8292410714285714
    ],
    "CI_unknown_envelope": [
      0.6606863839285714,
      0.9564732142857143
    ]
  },
  "G_S500_minus_RHR": {
    "mean": -0.78125,
    "CI95": [
      -0.90625,
      -0.625
    ],
    "identification_interval": [
      -0.78125,
      -0.78125
    ],
    "CI_unknown_envelope": [
      -0.90625,
      -0.625
    ]
  },
  "students_interpretation": "S500 优于公平调参 RHR"
}
