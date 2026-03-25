---
date: '2024-10-01'
title: DDA 파이프라인 최적화 — Shapley Value 연산을 분산 처리로 전환한 이야기
subtitle: Broadcast Variable + mapPartitions로 Driver 메모리 75% 절감
categories: data-engineering
tags: [Spark, 성능최적화]
---

대규모 쇼핑 이벤트 데이터 기반의 DDA(Data-Driven Attribution) 파이프라인에서, Shapley Value 기여도 모델 생성 단계의 성능 병목을 해결한 과정을 정리합니다.

## 배경

광고 유입부터 결제까지의 유저 터치포인트를 분석하여 채널별 기여도(Shapley Value)를 산출하는 배치 파이프라인입니다. 전환 경로(Path)별로 모든 하위 채널 조합에 대한 한계 기여도(Marginal Contribution)를 구해야 하며, 이 과정에서 path summary 테이블 전체를 참조하는 연산이 필요합니다.

---

## 문제: 단일 Task에서의 순차 처리

기존 구현은 Google Fractribution 레퍼런스와 동일하게, path summary 데이터를 하나의 Map 객체로 만들어 <strong>Driver에서 단일 Task로 순차 처리</strong>하는 방식이었습니다.

```scala
// 기존: 단일 객체를 순회하며 순차 처리
pathSummaryMap.map(x => {
  val path = x._1
  val probability = x._2
  val marginal_contributions =
    get_counterfactual_marginal_contributions(path, probability, pathSummaryMap)
  val sum_marginal_contributions = marginal_contributions.sum

  if (sum_marginal_contributions > 0.0) {
    path.zipWithIndex.map(x =>
      Fractional_Attribution_Value(
        path.mkString(LIST_SEQ),
        x._1,
        marginal_contributions(x._2) / sum_marginal_contributions))
  } else {
    null
  }
}).filter(_ != null).flatten.toList
```

<strong>문제점:</strong>
- <strong>분산 처리 불가</strong> — Spark를 쓰고 있지만 실제 연산은 Driver의 단일 Task에서 수행
- <strong>Driver 메모리 의존</strong> — path summary 크기가 커지면 `java.lang.OutOfMemoryError: GC overhead limit exceeded` 발생
- driver-memory를 늘려서 우회할 수는 있지만, input 크기에 따라 계속 예측해야 하므로 언젠가 다시 터질 수밖에 없는 구조

---

## 해결: Broadcast Variable + mapPartitions

두 가지 Spark primitive를 조합하여 분산 처리 구조로 재설계했습니다.

### ① Broadcast Variable — 참조 데이터를 전역 공유

모든 Task에서 path summary Map을 참조해야 합니다. 이 데이터를 `Broadcast Variable`로 선언하면, 각 Worker Node에 serialized 형태로 캐싱됩니다. Task가 사용할 때 deserialization을 통해 제공받으므로 Driver가 모든 연산을 수행할 필요가 없어집니다.

### ② mapPartitions — 연산을 파티션 단위로 분산

실제 기여도 연산을 `mapPartitions`로 감싸면, Dynamic Allocation 설정에 따라 필요한 만큼 N개의 Task로 자동 분산됩니다.

```scala
// 수정: Broadcast + mapPartitions 분산 처리
val broadcastedMap = spark.sparkContext.broadcast(
  pathSummaryRdd.collectAsMap().toMap
)
val map = broadcastedMap.value

val df = pathSummaryRdd.toDF("paths", "prob")

df.mapPartitions(iter => {
  iter.flatMap(row => {
    val path = row.getAs[String]("paths")
    val pathList = get_path_list(path)
    val probability = row.getAs[Double]("prob")

    val marginal_contributions =
      get_counterfactual_marginal_contributions(pathList, probability, map)
    val sum_marginal_contributions = marginal_contributions.sum

    if (sum_marginal_contributions > 0.0) {
      pathList.zipWithIndex.map(x =>
        (path, x._1, marginal_contributions(x._2) / sum_marginal_contributions))
    } else {
      List(("", "", 0.0))
    }
  })
}).toDF("paths", "channel", "s")
```

---

## 결과

동일한 Spark 설정에서 수정 전/후를 비교한 결과입니다.

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
| <strong>처리 시간</strong> | 불안정 (OOM 발생) | <strong>1분 30초 내외</strong> |

핵심은 <strong>"Driver에 집중된 연산을 Spark 엔진이 분산 처리할 수 있는 구조로 전환"</strong>한 것입니다. Broadcast로 참조 데이터를 공유하고, mapPartitions로 연산을 분산시켜 Driver의 부하를 Executor들로 고르게 분배했습니다.
