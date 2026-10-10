# Gold passage 멀티턴 답변 실험

검색 없이 동일한 gold passage에서 gpt-6-luna로 답변한다.

| Variant | 답변 생성 입력 | LLM 호출 |
|---|---|---|
| multiturn | passage + 이전 Q/A + 마지막 질문 | 1 |
| single_raw | passage + 마지막 질문 | 1 |
| single_rewrite | 이전 Q/A로 standalone 질문 생성 → passage + rewrite | 2 |

`single_rewrite`는 답변 생성 단계가 싱글턴이다. 파이프라인은 rewriting 때 이력을 사용한다.
`single_raw`는 이력을 완전히 제거한다. 원래 멀티턴 질문의 대명사/생략도 그대로 둔다.

## 실행

Python 3.10 이상. 외부 패키지 설치 불필요.

```bash
unzip coqa_gold_bench.zip
cd coqa_gold_bench
export OPENAI_API_KEY='YOUR_KEY'
python3 bench.py --samples samples_100.json --n 100 \
  --model gpt-6-luna --effort none --concurrency 4 --out results_luna
```

첨부한 samples_100.json은 공식 CoQA dev에서 seed=42로 뽑은 실제 데이터다.
처음 실행은 400 LLM calls (100 + 100 + 200), API 재시도 제외.
모델 접근 권한이 필요하다. 모델 이름을 자동 변경하지 않는다.
Responses API 지원 endpoint가 필요하며 `--base-url`로 변경할 수 있다.
endpoint가 reasoning 필드를 지원하지 않으면 새 출력 폴더와 `--effort omit`을 사용한다.

직접 데이터 다운로드 및 다른 표본 준비:

```bash
python3 bench.py --prepare-only --n 100 --seed 42 --out prepared
python3 bench.py --samples prepared/samples.json --out results_new
```

이력은 기본 전체 이전 턴. `--history-turns 3`으로 마지막 세 Q/A만 쓸 수 있다.
빠른 API 확인은 `--samples samples_100.json --n 3 --out smoke`.
중단/오류 후 동일 명령을 재실행하면 완료한 variant는 건너뛴다.
실패한 rewrite variant는 rewrite부터 다시 시작하므로 추가 호출이 발생한다.
프롬프트/모델/표본/이력 조건이 바뀌면 새 --out을 사용해야 한다.

## 데이터와 누출 방지

- 서로 다른 대화 100개에서 각각 turn_id > 1인 질문 하나를 균등 선택한다.
- passage는 전체 story이며 current gold answer span만 잘라 주지 않는다.
- history는 이전 질문과 원래 gold 답변이다. Teacher-forced 조건으로 평가하며 실제 서비스의 누적 답변 오류는 측정하지 않는다.
- 현재 정답, 추가 annotator 정답, 이후 질문/답변, 정답 rationale span은 모델 입력에 넣지 않는다.
- rewriting에는 이전 Q/A와 마지막 질문만 주고 passage도 제외한다.
- 세 경로 모두 동일한 answer system prompt를 사용하며 답변 설명 없이 짧은 정답만 요청한다.
- 전체 도메인에서 무작위 표본이며 도메인별 균등 할당/문맥 의존성 수동 라벨링은 하지 않는다.

## 출력/해석

- `summary.json`: Answer EM/F1, 오류 수, 유효 요청의 latency/tokens, paired differences 및 bootstrap 95% CI.
- `predictions.jsonl`: 개별 질문, rewrite, 답변, 정답, API 사용량, 오류.
- `samples.json`, `config.json`: 표본과 실험 fingerprint.
- EM/F1은 소문자, ASCII punctuation, 영어 관사, 공백 정규화 후 여러 gold 답변 중 최대 점수. **공식 CoQA leaderboard 평가의 leave-one-annotator-out 집계와 다르다.** 비교용 max-over-references 지표다.
- 오류는 전체 EM/F1에서 0점이며 valid-only 점수도 별도 기록한다. paired 비교에는 세 경로가 모두 성공한 예제만 사용한다.
- delta = variant − multiturn. 100개는 탐색 실험이며 CI가 넓을 수 있다. paraphrase에도 EM은 0이 될 수 있어 F1과 실제 출력도 함께 검토한다.
- single_rewrite가 multiturn에 근접하면 standalone 질문으로 답변하는 설계가 유망하다. single_raw가 낮고 single_rewrite가 높으면 이력 정보를 rewrite에 보존하는 게 도움이 된다.
- latency는 동시 실행/네트워크 영향을 포함하며 rewrite 경로는 두 호출 시간을 모두 포함한다.

## 출처와 라이선스

CoQA: https://stanfordnlp.github.io/coqa/
공식 dev: https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json
논문: https://arxiv.org/abs/1808.07042
CoQA passage는 도메인별 라이선스가 다르다: Wikipedia/Literature CC BY-SA 4.0,
MCTest MSR-LA, RACE 자체 라이선스, CNN Apache. 재사용/재배포 조건은 공식 페이지와
각 원본 라이선스를 확인한다. samples 파일의 source, conversation_id로 출처를 유지한다.

이 패키지에는 실제 API 성능 결과가 포함되어 있지 않다. 데이터 준비와 mock API 실행 흐름을 검증했다.
