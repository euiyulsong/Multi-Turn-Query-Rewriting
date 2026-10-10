## CoQA: 멀티턴 이력과 Query Rewriting 비교

### 실험 설정

- **데이터:** CoQA dev의 서로 다른 대화 100개에서 두 번째 턴 이후 질문 하나씩 선택
- **모델:** `gpt-6-luna`, reasoning effort `none`
- **조건:** 검색 없이 모든 방식에 동일한 gold passage 제공
- **이력:** 현재 질문 이전의 gold Q/A 사용
- **평가:** 최종 답변의 Answer EM/F1
- **성공:** 세 방식 모두 100개 완료, 오류 0건
- **지표 집계:** 정규화 후 여러 gold 답변 중 최대 점수 사용. 공식 CoQA leaderboard 집계와 다름

| 방식 | 답변 생성 입력 | LLM 호출 |
|---|---|---:|
| `multiturn` | Gold passage + 이전 Q/A + 마지막 질문 | 1 |
| `single_raw` | Gold passage + 마지막 질문 | 1 |
| `single_rewrite` | 이전 Q/A로 standalone 질문 생성 → gold passage + rewrite | 2 |

### 결과

토큰 수와 지연은 예제당 평균이며, `single_rewrite`는 rewriting과 답변 생성 비용을 합산한다.

| 방식 | Answer EM | Answer F1 | 평균 지연 | p95 지연 | 입력 토큰 | 출력 토큰 |
|---|---:|---:|---:|---:|---:|---:|
| `multiturn` | 46.0% | 62.21% | 1.464초 | 2.196초 | 598.33 | 9.92 |
| `single_raw` | 42.0% | 56.40% | **1.376초** | **1.930초** | **422.08** | **8.97** |
| `single_rewrite` | **58.0%** | **76.77%** | 2.844초 | 3.695초 | 680.91 | 26.53 |

### Multiturn 대비 Paired 비교

동일한 100개 예제에서 비교했다. 승/패는 해당 지표가 multiturn보다 상승/하락한 예제 수이며, 나머지는 동점이다.

| 방식 | 지표 | 점수 차이 | Bootstrap 95% CI | 승 / 패 / 동점 |
|---|---|---:|---:|---:|
| `single_raw` | EM | −4.00%p | [−10.00, +2.00]%p | 3 / 7 / 90 |
| `single_raw` | F1 | −5.81%p | [−11.43, −0.22]%p | 9 / 19 / 72 |
| `single_rewrite` | EM | **+12.00%p** | **[+4.00, +21.00]%p** | 17 / 5 / 78 |
| `single_rewrite` | F1 | **+14.56%p** | **[+6.39, +23.16]%p** | 29 / 15 / 56 |

### 해석

**이력으로 질문을 독립형으로 rewriting한 뒤 싱글턴으로 답변한 방식이 가장 높은 성능을 보였다.**

- `single_rewrite`는 multiturn 대비 EM **+12.00%p**, F1 **+14.56%p** 향상했다. 두 지표의 bootstrap 95% CI 모두 0보다 높다.
- `single_raw`는 EM −4.00%p, F1 −5.81%p 하락했다. EM의 CI는 0을 포함하며, F1의 CI는 0보다 낮다.
- 이력을 그대로 제거하면 문맥 정보가 손실될 수 있다. 이력 정보를 standalone 질문에 반영하면 답변 생성 단계는 싱글턴으로 처리할 수 있다는 가능성을 보여준다.
- Rewriting이 지시 대상과 생략된 조건을 명시해 답변에 도움을 줬을 가능성이 있다. 구체적인 개선 원인은 개별 rewrite와 답변을 확인해야 한다.

### 성능과 비용

`single_rewrite`의 multiturn 대비 비용 변화:

| 항목 | 변화 |
|---|---:|
| 평균 지연 | +1.380초, 약 1.94배 |
| p95 지연 | +1.500초 |
| 평균 입력 토큰 | +13.8% |
| 평균 출력 토큰 | +167.4% |
| LLM 호출 | 1회 → 2회 |

`single_raw`는 입력 토큰을 약 29.5% 줄였지만, 평균 지연 감소는 약 0.088초였다.

### 적용 판단과 한계

이번 조건에서는 **이력 기반 rewriting → 싱글턴 답변 생성**을 우선 후보로 볼 수 있다. 이력을 완전히 제거한 `single_raw`는 성능 손실을 고려해야 한다.

다만 이 결과는 gold passage를 제공한 100개 표본의 답변 생성 실험이다. 검색 성능과 실제 서비스의 이전 답변 오류 누적은 측정하지 않았다. 실서비스 적용 전에는 실제 검색 context와 대화 이력으로 재평가해야 한다.

### 실행 명령

~~~bash
python3 bench.py \
  --samples samples_100.json \
  --n 100 \
  --model gpt-6-luna \
  --effort none \
  --concurrency 4 \
  --out results_luna
~~~
