---
date: '2023-01-01'
title: "OOM을 방지하는 세션화 전략: 왜 Window Function인가?"
subtitle: "collect_list + UDF의 메모리 집약적 구조를 Window Function으로 전환한 이유"
categories: data-engineering
tags: [Spark, 성능최적화]
---

사용자 행동 로그를 분석하다 보면 세션화(Sessionization)나 전환 경로(Conversion Path) 분석처럼 유저별로 이벤트를 시간순으로 정렬해 처리해야 하는 경우가 많습니다. 이때 가장 먼저 떠오르는 직관적인 방법은 `groupBy`로 유저를 묶고 `collect_list`로 이벤트들을 리스트화하여 UDF에서 처리하는 방식입니다.

하지만 데이터 규모가 커지고, 특정 유저에게 로그가 쏠리는 <strong>데이터 스큐(Skew)</strong> 상황이 발생하면 이 방식은 여지없이 <strong>Executor OOM(Out Of Memory)</strong>을 일으킵니다. 왜 그럴까요? 그리고 왜 <strong>Window Function</strong>이 그 대안이 될 수 있을까요?

---

## `groupBy` + `collect_list`가 위험한 이유: 메모리 집약적 구조

Spark에서 `collect_list`는 한 유저(Key)에 해당하는 모든 데이터를 하나의 거대한 Java List 객체로 묶습니다. 이러한 방식은 아래와 같은 문제를 야기할 수 있습니다.
- <strong>객체화 오버헤드:</strong> 수만 개의 이벤트가 하나의 리스트 객체로 변환되는 순간, JVM의 Executor Memory를 순식간에 점유합니다.
- <strong>직렬화(Serialization) 병목:</strong> 데이터를 UDF로 넘기기 위해 직렬화/역직렬화하는 과정에서 CPU 부하와 메모리 복사 비용이 기하급수적으로 증가합니다.
- <strong>원자적 처리의 한계:</strong> 리스트는 한꺼번에 메모리에 올라와야 하므로, 메모리보다 큰 데이터를 나누어서 처리할 수 없습니다.

결국 어뷰징 봇이나 헤비 유저처럼 수만~수십만 건의 이벤트를 발생시키는 단 한 명의 유저가 전체 파이프라인을 다운시킬 수 있는 구조적 취약점을 안고 있는 것입니다.

---

## Window Function의 기술적 우위

Window Function은 데이터를 처리하는 메커니즘 자체가 다릅니다.

### 스트리밍 방식의 데이터 처리

Window Function은 모든 데이터를 리스트 객체로 만들지 않습니다. 대신 파티션 내에서 정렬된 데이터를 <strong>한 로우씩 읽어 내려가며 연산</strong>합니다.

현재 로우와 설정된 윈도우 프레임(예: `ROWS BETWEEN 1 PRECEDING AND CURRENT ROW`)에 해당하는 데이터만 참조하므로, 전체 데이터를 메모리에 상주시킬 필요가 없습니다. `LAG`, `LEAD`, `SUM OVER` 같은 연산은 현재 로우 기준으로 앞뒤 몇 건만 참조하면 되기 때문에, 유저의 이벤트 수가 아무리 많아도 메모리 사용량이 일정하게 유지됩니다.

### 정렬 기반의 Spill to Disk 메커니즘

`groupBy` + `collect_list`는 메모리가 부족하면 OOM으로 작업이 실패합니다. 반면 Window Function은 Spark의 <strong>External Sorter</strong>를 적극 활용합니다.

파티션 내 데이터를 정렬하는 과정에서 메모리가 부족해지면, Spark는 이를 디스크로 <strong>Spill</strong>시켜 작업을 끝까지 완주합니다. 속도는 조금 느려질지언정 파이프라인이 중단되지는 않습니다. 이것이 운영 환경에서 Window Function이 훨씬 안정적인 이유입니다.

### Tungsten 엔진 최적화

Window Function은 Spark의 Catalyst Optimizer와 Tungsten 실행 엔진의 이점을 그대로 활용합니다. Whole-Stage CodeGen을 통해 JVM 바이트코드로 직접 컴파일되고, off-heap 메모리 관리를 통해 GC 부하를 최소화합니다. 반면 `collect_list` + UDF 조합은 이 최적화 경로를 완전히 우회하게 됩니다.

### 데이터 스큐(Skew) 상황에서의 방어력

어뷰징 트래픽이나 헤비 유저로 인해 특정 Key에 수백만 건의 데이터가 몰리는 스큐 상황에서 차이가 극명해집니다.

`collect_list`는 수백만 건을 한꺼번에 메모리에 올리려다 OOM이 발생하지만, Window Function은 정렬된 상태에서 순차적으로 연산을 수행하므로 메모리 부하를 상수로 제어할 수 있습니다.

| 항목 | <strong>collect_list + UDF</strong> | <strong>Window Function</strong> |
|------|---------|---------|
| <strong>데이터 처리 방식</strong> | 전체 리스트를 메모리에 적재 | 정렬된 파티션을 스트리밍 처리 |
| <strong>메모리 사용</strong> | Key당 이벤트 수에 비례 (O(n)) | 윈도우 프레임 크기에 비례 (O(1)) |
| <strong>메모리 부족 시</strong> | OOM → 작업 실패 | Spill to Disk → 작업 완주 |
| <strong>엔진 최적화</strong> | Tungsten/CodeGen 우회 | Tungsten/CodeGen 적용 |
| <strong>데이터 스큐 방어</strong> | 헤비 유저 1명이 파이프라인 다운 | 메모리 부하 상수 제어 |

