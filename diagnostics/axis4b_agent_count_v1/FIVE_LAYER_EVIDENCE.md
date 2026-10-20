| N | H_N | geometry valid | single-robot path | joint expert Q1 | controller capability | correction | certified deadlock resolution |
|---|---|---|---|---|---|---|---|
| 2 | 2000 | 32/32 | 32/32 | 32/32 | Flow:FAIL_CONTROLLER_CAPABILITY; P2:FAIL_CONTROLLER_CAPABILITY | GAP or source-only | NOT_ESTABLISHED |
| 10 | 8000 | 32/32 | 32/32 | 32/32 | Flow:FAIL_CONTROLLER_CAPABILITY; P2:FAIL_CONTROLLER_CAPABILITY | GAP or source-only | NOT_ESTABLISHED |
| 20 | 16000 | 32/32 | 32/32 | 32/32 | Flow:FAIL_CONTROLLER_CAPABILITY; P2:INCONCLUSIVE | GAP or source-only | NOT_ESTABLISHED |
| 50 | 40000 | 32/32 | 32/32 | 30/32 | Flow:INCONCLUSIVE; P2:FAIL_CONTROLLER_CAPABILITY | GAP or source-only | NOT_ESTABLISHED |

专家失败只表示没有找到本规划器的联合证书，不能证明不可解。正常排队、stalled_groups、timeout都不是正式deadlock。最后一列与一般修正成功率严格分开。
