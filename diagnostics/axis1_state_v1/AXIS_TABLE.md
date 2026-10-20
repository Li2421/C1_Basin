| subitem | evidence | nominal only | safety baseline | F10 | S500 | A3 zero-shot | A0 | G_E | best fixed oracle | full menu oracle | A3 per state | A0 per state | state oracle | F13 | S300 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1a same-distribution new states; seen recipes/unseen checkpoints | CONFIRMATORY specified cohort; State-only isolation GAP | 9.10–9.10 [0.00, 23.72] | 9.21–9.26 [0.00, 24.11] | 61.38–61.94 [48.38, 75.56] | 90.96–91.91 [87.44, 94.81] | 91.13–92.08 [87.56, 94.87] | 90.96–91.91 [87.44, 94.81] | 91.02–91.96 [87.50, 94.70] | 63.73–65.68 [50.28, 78.18] | 90.57–92.35 [85.60, 94.92] | 91.02–91.96 [87.44, 94.75] | 90.85–91.80 [87.28, 94.70] | 82.31–84.26 [77.01, 89.01] | 37.00–38.06 [22.38, 52.57] | 88.78–89.68 [84.21, 93.58] | cross-fitted oracle; unknown intervals retained |
| 1b shifted states; seen recipes/unseen checkpoints | CONFIRMATORY specified cohort; State-only isolation GAP | 0.00–0.00 [0.00, 0.00] | 0.00–0.00 [0.00, 0.00] | 28.24–29.46 [21.76, 36.05] | 60.71–61.16 [42.86, 77.79] | 61.83–62.28 [44.64, 78.24] | 60.71–61.16 [42.86, 77.79] | 61.38–61.83 [43.86, 77.90] | 27.46–30.13 [19.20, 35.71] | 59.82–62.17 [42.63, 77.23] | 61.27–61.72 [44.31, 77.46] | 59.82–60.38 [42.41, 76.12] | 50.67–52.34 [37.05, 65.07] | 15.40–15.74 [10.27, 21.65] | 58.48–58.82 [42.19, 73.44] | cross-fitted oracle; unknown intervals retained |
| 1c state selection | CONFIRMATORY specified cohort; State-only isolation GAP | N/A: comparison endpoint | N/A: comparison endpoint | N/A: comparison endpoint | N/A: comparison endpoint | see 1a/1b rows | see 1a/1b rows | N/A: comparison endpoint | N/A: comparison endpoint | N/A: comparison endpoint | see 1a/1b rows | see 1a/1b rows | see 1a/1b rows | N/A: comparison endpoint | N/A: comparison endpoint | {"state_minus_controller_pp": [-1.0416666666666665, 0.5208333333333333], "CI_pp": [-1.8973214285714284, 1.3392857142857142], "status": "undetected"} |
| historical T2 R26 aligned | DEVELOPMENT | N/A: not in extracted historical panel | N/A: not in extracted historical panel | 57.76–58.65 [43.64, 72.71] | 91.41–92.19 [88.06, 94.92] | 91.63–92.41 [88.23, 95.09] | N/A: not in extracted historical panel | 91.85–92.63 [88.28, 95.42] | N/A: not in extracted historical panel | N/A: not in extracted historical panel | N/A: not in extracted historical panel | N/A: not in extracted historical panel | N/A: not in extracted historical panel | N/A: not in extracted historical panel | N/A: not in extracted historical panel | Existing wider R10 menu and32-state evaluation split; differs from current four-arm protocol |
| historical T2 R16 | DEVELOPMENT | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | N/A: prefer aligned R26 panel | Legacy reported A3 regret0.75pp; obsolete state-input qualification |
| historical R2 per-state selection | DEVELOPMENT | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | N/A: different single-controller/menu/Q512 design | Per-state gain+5.21pp [2.65,7.88] relative best fixed; see source REPORT |

Success % is lower–upper unknown envelope [95% paired nested bootstrap CI]. All raw unknowns are retained.

| cohort | method | collision% | timeout% | unknown% | mean regret pp | max regret pp | catastrophes certain–possible | harm certain–possible |
|---|---|---|---|---|---|---|---|---|
|1a|nominal only|60.21205357142857|30.691964285714285|0.0|[81.47321428571429, 83.25892857142857]|[100.0, 100.0]|13–13|0–0|
|1a|safety baseline|0.0|90.73660714285714|0.055803571428571425|[81.30580357142857, 83.14732142857143]|[100.0, 100.0]|13–13|0–0|
|1a|F10|0.0|38.058035714285715|0.5580357142857143|[28.627232142857146, 30.970982142857146]|[57.8125, 60.15625]|0–0|1–1|
|1a|S500|0.0|8.091517857142858|0.9486607142857142|[-1.3392857142857142, 1.3950892857142858]|[3.90625, 4.6875]|0–0|0–0|
|1a|F13|0.0|61.94196428571429|1.0602678571428572|[52.51116071428571, 55.35714285714286]|[85.15625, 87.5]|10–11|0–0|
|1a|S300|0.0|10.323660714285714|0.8928571428571428|[0.8928571428571428, 3.571428571428571]|[7.8125, 10.9375]|0–0|0–0|
|1a|A3 zero-shot|0.0|7.924107142857142|0.9486607142857142|[-1.5066964285714286, 1.2276785714285714]|[3.90625, 4.6875]|0–0|0–0|
|1a|A3 per state|0.0|8.035714285714286|0.9486607142857142|[-1.3950892857142858, 1.3392857142857142]|[3.90625, 4.6875]|0–0|0–0|
|1a|A0|0.0|8.091517857142858|0.9486607142857142|[-1.3392857142857142, 1.3950892857142858]|[3.90625, 4.6875]|0–0|0–0|
|1a|A0 per state|0.0|8.203125|0.9486607142857142|[-1.2276785714285714, 1.5066964285714286]|[3.90625, 4.6875]|0–0|0–0|
|1a|G_E|0.0|8.035714285714286|0.9486607142857142|[-1.3950892857142858, 1.3392857142857142]|[3.90625, 4.6875]|0–0|0–0|
|1a|best fixed oracle|0.0|34.82142857142857|0.6138392857142857|[24.888392857142858, 28.627232142857146]|[57.8125, 64.84375]|0–1|0–0|
|1a|full menu oracle|0.0|8.147321428571429|0.8928571428571428|[-1.7857142857142856, 1.7857142857142856]|[0.0, 10.9375]|0–0|0–0|
|1a|state oracle|0.0|16.294642857142858|0.8928571428571428|[6.305803571428571, 10.044642857142858]|[15.625, 25.0]|0–0|0–0|
|1b|nominal only|66.29464285714286|33.705357142857146|0.0|[59.82142857142857, 62.16517857142857]|[87.5, 93.75]|13–13|0–0|
|1b|safety baseline|0.0|100.0|0.0|[59.82142857142857, 62.16517857142857]|[87.5, 93.75]|13–13|0–0|
|1b|F10|0.0|70.53571428571429|1.2276785714285714|[30.357142857142854, 33.92857142857143]|[60.9375, 71.875]|0–0|0–0|
|1b|S500|0.0|38.839285714285715|0.4464285714285714|[-1.3392857142857142, 1.4508928571428572]|[6.25, 6.25]|0–0|0–0|
|1b|F13|0.0|84.26339285714286|0.33482142857142855|[44.08482142857143, 46.763392857142854]|[73.4375, 75.0]|8–9|0–0|
|1b|S300|0.0|41.183035714285715|0.33482142857142855|[1.0044642857142858, 3.6830357142857144]|[12.5, 14.0625]|0–0|0–0|
|1b|A3 zero-shot|0.0|37.723214285714285|0.4464285714285714|[-2.455357142857143, 0.33482142857142855]|[6.25, 6.25]|0–0|0–0|
|1b|A3 per state|0.0|38.28125|0.4464285714285714|[-1.8973214285714284, 0.8928571428571428]|[4.6875, 4.6875]|0–0|0–0|
|1b|A0|0.0|38.839285714285715|0.4464285714285714|[-1.3392857142857142, 1.4508928571428572]|[6.25, 6.25]|0–0|0–0|
|1b|A0 per state|0.0|39.620535714285715|0.5580357142857143|[-0.5580357142857143, 2.34375]|[6.25, 6.25]|0–0|0–0|
|1b|G_E|0.0|38.169642857142854|0.4464285714285714|[-2.0089285714285716, 0.78125]|[3.125, 6.25]|0–0|0–0|
|1b|best fixed oracle|0.0|69.97767857142857|1.0044642857142858|[29.6875, 34.70982142857143]|[60.9375, 71.875]|0–1|0–0|
|1b|full menu oracle|0.0|39.0625|0.4464285714285714|[-2.34375, 2.34375]|[0.0, 9.375]|0–0|0–0|
|1b|state oracle|0.0|48.10267857142857|0.78125|[7.477678571428571, 11.495535714285714]|[26.5625, 26.5625]|0–0|0–0|
