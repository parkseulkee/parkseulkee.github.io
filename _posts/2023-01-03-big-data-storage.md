---
layout: post
title: Storage architectures for big data
subtitle: Data Warehouse vs. Data Lake vs. Data Lakehouse
categories: bigdata
tags: [bigdata]
---
본 글은 [Data Warehouse vs. Data Lake vs. Data Lakehouse: An Overview of Three Cloud Data Storage Patterns](https://www.striim.com/blog/data-warehouse-vs-data-lake-vs-data-lakehouse-an-overview/) 을 번역 및 정리하였습니다.

`Data Warehouse` 와 `Data Lake`는 빅 데이터에 가장 널리 사용되는 스토리지 아키텍처이다.
**Data Lakehouse는 Data Lake 의 유연성과 Data Warehouse의 데이터 관리를 결합한 새로운 데이터 스토리지 아키텍처라고 이해할 수 있다.**
회사의 요구 사항에 따라 다양한 빅 데이터 스토리지 기술을 이해하는 것은 BI, ML 등 워크로드를 위한 강력한 데이터 스토리지 파이프라인을 개발하는 데 도움이 되므로 본 글에서는 각 3가지 아키텍처에 대해 간단하게 이해하고 비교해보도록 한다.

## What is a Data Warehouse?

![](https://velog.velcdn.com/images/srk/post/08b89859-39a6-4664-be10-47f170651ed6/image.png)

### The benefits of a data warehouse
- **데이터 표준화, 품질 및 일관성** : 일관되고 표중화된 형식으로 단일 소스 데이터 제공, 비즈니스 요구 사항에 적합한 데이터로 의존성
- **데이터 분석 및 BI 워크로드의 성능과 속도 향상** : 데이터 준비 및 분석에 필요한 시간을 단축시킴, Data Warehouse 의 데이터는 일관되고 정확하기 때문에 데이터 분석 및 BI 도구에 쉽게 연결 가능, 또한 데이터 수집에 필요한 시간을 단축하고 팀이 데이터를 활용할 수 있는 권한을 제공

### The disadvantages of a data warehouse
- **데이터의 유연성 부족** : Data Warehouse는 structured data에서 잘 작동하지만 로그 분석, 스트리밍 및 소셜 미디어 데이터와 같은 semi-structured 와 unstructured data 에서 어려움을 겪을 수 있음, 이로 인해 ML 및 AI use case 에 대해 권장하기 힘듬
- **높은 구현 및 유지 관리 비용** : 정기적인 유지 관리 비용, 스토리비지 비용

## What is a Data Lake?

![](https://velog.velcdn.com/images/srk/post/cd59e72e-ddd8-4b7a-af66-a5caff56b5eb/image.png)

**Data Lake 는 대량의 structured data 와 unstructured data 를 raw, original, unformatted 형식으로 저장하는 매우 유연한 중앙 집중식 스토리지 저장소이다.** 이미 'cleaned' 관계형 데이터를 저장하는 Data Warehouse 와 달리 Data Lake는 플랫 아키텍처와 객체 스토리지를 사용하여 raw 형식으로 데이터를 저장한다. Data Lake는 유연하고 내구성이 있으며 비용 효율적이며 Data Warehouse 와 달리 unstructured data 에서 통찰력을 얻을 수 있다.

### The benefits of a data lake
- **데이터 통합** : tructured data 와 unstructured data를 모두 저장할 수 있으므로 서로 다른 환경에서 두 데이터 형식을 모두 저장할 필요가 없음. 모든 유형의 조직 데이터를 저장하는 중앙 저장소를 제공.
- **데이터 유연성** : 가장 중요한 이점. 미리 정의된 스키마가 없어도 데이터를 모든 형식이나 매체로 저장할 수 있음.
- **비용 절약** : 기존 Data Warehouse보다 저렴한 편. 객체 스토리지와 같은 저비용 상용 하드웨어에 저장하도록 설계.
- **다양한 data science 와 ML의 use case 지원** : raw 데이터 형식으로 저장되므로 다양한 머신 러닝 또는 딥 러닝 알고리즘을 적용하여 데이터를 처리해 의미 있는 인사이트를 생성하기 더 쉬움.

### The disadvantages of a data lake

- **BI 및 데이터 분석의 성능 저하** : 적절하게 관리되지 않으면 Data Lake가 무질서해져 BI 및 분석 도구와 연결하기 어려울 수 있음. 또한 일관된 데이터 구조 및 ACID(원자성, 일관성, 격리, 및 내구성) 트랜잭션 지원이 부족하여 쿼리 성능이 최적화되지 않을 수 있음.
- **데이터 신뢰성 및 보안 부족** : 모든 데이터 형식을 수용할 수 있어 민감한 데이터 유형을 수용하기 위해 적절한 데이터 보안 및 거버넌스 정책을 구현하는 것이 어려움.

위 단점을 보완하기 위해 많은 경우 Data Lake + Data Mart / 또는 다른 DB 등 용도에 맞는 특화된 시스템을 함께 운영하게 된다. 이에 운영 복잡도가 증가하며 동일 데이터를 여러 시스템에 중복 적재하게 된다.

## What Is a Data Lakehouse? A Combined Approach

![](https://velog.velcdn.com/images/srk/post/8e04c61e-4818-49fe-83b2-9fe5496daef8/image.png)

Data Lakehouse 는 일반적으로 모든 데이터 유형을 포함하는 Data Lake로 시작한다. 그런 다음 데이터는 [Delta Lake](https://docs.delta.io/latest/delta-intro.html) 형식 (Data Lake에 안정성을 제공하는 오픈 소스 스토리지 계층)으로 변환된다. Delta Lake는 기존 Data Warehouse 에서의 ACID 트랜잭션 프로세스를 Data Lake 위에서 지원한다.

> **Delta Lake is an open source project that enables building a Lakehouse architecture on top of data lakes.** Delta Lake provides ACID transactions, scalable metadata handling, and unifies streaming and batch data processing on top of existing data lakes, such as S3, ADLS, GCS, and HDFS.

> Delta Lake 와 비슷한 needs로 Apache Iceberg, Apache Hudi, Delta Lake, Hive ACID table 등의 동일한 포지션의 프로젝트들이 인기를 얻고 있다.


### The benefits of a data lakehouse

**Data Lakehouse 의 아키텍처는 Data Warehouse의 데이터 구조 및 관리 기능을 Data Lake의 저비용 스토리지 및 유연성과 결합했다.** 엄청난 이점이 있으며 다음을 포함한다.

- **데이터 중복 감소** : 모든 비즈니스 데이터 요구 사항을 충족하는 단일 다목적 데이터 스토리지 플랫폼을 제공하여 데이터 중복을 줄임. Data Warehouse와 Data Lake의 각각의 장점 때문에 대부분의 기업은 하이브리드 솔루션을 선택했지만 이 접근 방식은 비용이 많이 드는 데이터 복제로 이어졌지만 Data Lakehouse 는 이를 해결할 수 있음.
- **비용 효율성**
- **다양한 워크로드 제공** : Data Lakehouse는 가장 널리 사용되는 일부 BI 툴(Tableau, PowerBI) 에 대한 직접 액세스를 제공하여 고급 분석을 가능하게 함. 또한 python/R 을 포함한 API 및 ML 라이브러리와 함께 개방형 데이터 형식(예. parquet)를 사용하므로 data scientist 와 ML 엔지니어가 데이터를 쉽게 활용할 수 있음.
- **데이터 버전 관리, 거버넌스 및 보안의 용이성** : 스키마 및 데이터 무결성을 강화하여 강력한 데이터 보안 및 거버넌스 메커니즘을 보다 쉽게 구현할 수 있음.

### The disadvantages of a data lakehouse

아직 Data Warehouse와 Data Lake 에 비해 미성숙하다.