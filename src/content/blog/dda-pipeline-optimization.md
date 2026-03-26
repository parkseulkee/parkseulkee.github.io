---
date: '2024-10-01'
title: DDA 파이프라인 최적화 — Shapley Value 연산을 분산 처리로 전환한 이야기
subtitle: Broadcast Variable + mapPartitions로 Driver 메모리 75% 절감
categories: data-engineering
tags: [Spark, 성능최적화]
---

대규모 쇼핑 이벤트 데이터 기반의 DDA(Data-Driven Attribution) 파이프라인에서, Shapley Value 기여도 모델 생성 단계의 성능 병목을 해결한 과정을 정리합니다. 단순히 "빨라졌다"는 결과보다, Spark의 분산 아키텍처를 어떻게 활용하여 병목을 해결했는지에 초점을 맞추었습니다.

## Shapley Value 기여도 산출 원리

DDA 모델은 Google Fractribution을 기반으로, 협력 게임 이론의 <strong>Shapley Value</strong>를 마케팅 채널 기여도 측정에 적용한 모델입니다. 핵심 연산은 다음과 같습니다:

1. 하나의 전환 경로(Path)에서 각 채널을 하나씩 제외한 <strong>Counterfactual Path</strong>를 만든다
2. 원래 경로의 전환 확률과 Counterfactual Path의 전환 확률 차이를 구한다 — 이것이 <strong>한계 기여도(Marginal Contribution)</strong>
3. 각 채널의 한계 기여도를 정규화(Normalize)하여 합이 1이 되도록 분배한다
4. <strong>모든 경로에 대해 반복</strong>한다

아래는 Marginal Contribution을 구하는 과정입니다. Path Summary Table에서 원래 경로와 채널을 하나씩 제외한 Counterfactual Path의 전환 확률 차이를 계산합니다.

![Marginal Contribution 계산 과정 — Google Fractribution Slides](/images/blog/fractribution-shapley-calculation.png)

여기서 핵심 문제는, Counterfactual Path의 전환 확률을 구하려면 <strong>전체 Path Summary Table(경로별 전환/비전환 집계)</strong>을 참조해야 한다는 것입니다. 즉, 모든 경로의 기여도 연산이 하나의 거대한 참조 테이블에 의존하는 구조입니다.

---

## 최적화 전: Driver 중심의 순차 처리

기존 구현은 Google Fractribution 레퍼런스와 동일하게, Path Summary Table을 하나의 Map 객체로 Driver 메모리에 올린 뒤, <strong>단일 프로세스에서 모든 경로를 순차적으로 순회</strong>하며 기여도를 계산하는 방식이었습니다.

```scala
// [Before] Sequential processing on Driver's single Task
// pathSummaryMap: Map[Path, (totalConversions, totalNull)] — all path summaries

pathSummaryMap.map { case (path, probability) =>
  // Create counterfactual paths by removing each channel one by one
  // → requires lookup against the entire pathSummaryMap
  val marginalContributions = path.indices.map { i =>
    val counterfactualPath = path.patch(i, Nil, 1)  // remove i-th channel
    val baseProb = probability
    val counterfactualProb = pathSummaryMap.getOrElse(counterfactualPath, 0.0)
    baseProb - counterfactualProb  // marginal contribution
  }

  val total = marginalContributions.sum
  if (total > 0.0) {
    // normalize: fractional attribution per channel
    path.zipWithIndex.map { case (channel, idx) =>
      (path.mkString(">"), channel, marginalContributions(idx) / total)
    }
  } else Nil
}.flatten.toList
```

분산 환경인 Spark를 사용함에도 불구하고 실제 핵심 연산은 Driver의 단일 Task에서 순차적으로 수행되는 구조였습니다. 이는 사실상 <strong>분산 처리를 하지 않는 것과 동일</strong>합니다.

<strong>문제점:</strong>
- <strong>Driver Memory 병목</strong> — Path Summary Table과 모든 연산 결과가 Driver 메모리에 적재되어, 데이터 규모가 커질수록 `java.lang.OutOfMemoryError: GC overhead limit exceeded` 발생 (8GB 할당 필요)
- <strong>Single Thread 연산</strong> — Executor 클러스터가 있어도 CPU 자원 활용도가 극도로 낮음
- <strong>확장성 부재</strong> — 데이터량 증가 시 driver-memory를 계속 늘려야 하며, 언젠가 한계에 도달하는 구조적 문제

---

## 최적화 후: Broadcast Variable + mapPartitions 기반 분산 설계

핵심 아이디어는 두 가지입니다:

### ① Broadcast Variable — 참조 테이블을 전역 공유

모든 경로의 기여도 연산에 Path Summary Table 전체가 필요합니다. 이 테이블을 `Broadcast Variable`로 선언하면, <strong>각 Worker Node에 serialized 형태로 한 번만 전송되어 캐싱</strong>됩니다. 개별 Task가 매번 데이터를 참조하는 네트워크 오버헤드를 최소화하면서, Driver가 아닌 Executor에서 연산할 수 있는 기반을 마련합니다.

### ② mapPartitions — 연산을 Executor로 분산

`mapPartitions`를 사용하면 각 파티션의 데이터가 Executor에서 병렬로 처리됩니다. `map` 대신 `mapPartitions`를 사용한 이유는, 파티션 내 데이터를 배치(Batch) 단위로 처리하여 객체 생성 오버헤드를 줄이고 Broadcast 변수의 deserialization을 파티션당 한 번만 수행하기 위함입니다.

```scala
// [After] Fully Distributed via Broadcast + mapPartitions

// 1. Share Path Summary Table as Broadcast Variable
//    → cached on each Worker Node, network transfer once per node
val pathSummaryMap: Map[List[String], Double] =
  pathSummaryRdd.collectAsMap().toMap
val broadcastPathSummary = spark.sparkContext.broadcast(pathSummaryMap)

// 2. Build DataFrame from path data
val pathsDf = pathSummaryRdd.toDF("path", "probability")

// 3. Parallel computation on Executors via mapPartitions
val resultDf = pathsDf.mapPartitions { partition =>
  // Deserialize broadcast variable once per partition
  val summaryTable = broadcastPathSummary.value

  partition.flatMap { row =>
    val path = row.getAs[String]("path")
    val channels = path.split(">").toList
    val baseProb = row.getAs[Double]("probability")

    // create counterfactual paths and lookup probabilities from summaryTable
    val marginalContributions = channels.indices.map { i =>
      val counterfactualPath = channels.patch(i, Nil, 1)
      val counterfactualProb = summaryTable.getOrElse(counterfactualPath, 0.0)
      baseProb - counterfactualProb
    }

    val total = marginalContributions.sum
    if (total > 0.0) {
      channels.zipWithIndex.map { case (channel, idx) =>
        (path, channel, marginalContributions(idx) / total)
      }
    } else Nil
  }
}.toDF("path", "channel", "attribution")
```

<strong>No-Collect 구조:</strong> 데이터를 Driver로 모으지 않고, Worker Node에서 직접 결과값을 산출하여 즉시 다음 레이어로 전달하는 <strong>Fully Distributed Pipeline</strong>을 완성했습니다.

---

## 결과

동일한 Spark 클러스터 설정에서 수정 전/후를 비교한 결과입니다.

```
--conf spark.dynamicAllocation.enabled=true
--conf spark.dynamicAllocation.maxExecutors=200
--conf spark.dynamicAllocation.minExecutors=1
--executor-memory 4g
--executor-cores 3
```

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| <strong>처리 방식</strong> | Driver 단일 Task 순차 처리 | Executor N개 Task 분산 처리 |
| <strong>Driver Memory</strong> | 8GB (OOM 위험) | 2GB (안정) |
| <strong>메모리 절감</strong> | — | <strong>75% 절감</strong> |
| <strong>처리 시간</strong> | 불안정 (OOM 발생) | <strong>90초 이내</strong> |
| <strong>확장성</strong> | driver-memory 수동 증설 필요 | 노드 추가로 선형 확장 |

<strong>메모리 효율성:</strong> Driver 메모리 의존성을 제거하여, 한정된 클러스터 자원 내에서 더 많은 배치 파이프라인을 동시에 실행할 수 있는 운영 효율로 이어졌습니다.

<strong>운영 안정성:</strong> 데이터 스케일 아웃 시에도 클러스터 노드만 추가하면 성능이 선형적으로 확장되는 구조를 확보하여, 향후 트래픽 증가에 유연하게 대응할 수 있게 되었습니다.
