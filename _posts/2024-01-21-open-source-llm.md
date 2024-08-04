---
layout: post
title: Open Source LLM
subtitle: llama, alpaca, vicuna
categories: LLM
tags: [llm, open-source-llm]
---

ChatGPT 를 시작으로 많은 LLM 들이 탄생하고 있다.
그 중에서도 오늘은 Open source LLM 을 다뤄보려한다.

## LLaMa(Large Language Model Meta AI)

`LLaMa` 는 Meta AI 에서 공개한 LLM 이다.
- `알파카(Alpaca)`, `비큐냐(Vicuna)` 등의 파생형 모델들의 탄생들에 기여
- 폐쇄형 소스 모델의 적절한 대체재

![](https://velog.velcdn.com/images/srk/post/0af1695a-93f7-49ea-919b-2e675ce5d184/image.png)

최근 구글의 최대의 경쟁자는 OpenAI가 아닌 오픈소스 AI라고 주장하는 구글 내부 문서가 공개 유출되었다. 문서 중에 일부를 발췌했는데, 아래와 같다고 합니다. 비슷한 이유로 관련 종사자들이 `Open source LLM` 에 열광하는 이유가 아닐까 생각이 듭니다.

![](https://velog.velcdn.com/images/srk/post/6f315ae3-5b4b-4634-a769-96394dac60a4/image.png)
> 이와 같은 주장의 배경에는 Meta의 LLaMA가 한정적인 researcher들에게 공개된 후, 마치 캄브리아기 대폭발을 연상시키듯 LLaMA를 기반한 오픈소스 AI 모델들이 탄생하고 있기 때문이다. 유출된 구글 내부 문건은 Vicuna-13B의 결과를 인용하면서 LLaMA-13B 기반 Vicuna-13B가 구글의 Bard와 거의 차이가 없는 성능 결과를 보이며, 더 작은 비용과 파라미터로 이를 달성하고 있다는 점을 지적하고 있다.

2023년 7월에 `LLaMa 2` 가 공개되었다.
기존에 연구용 목적으로 한정적으로 오픈했던 LLaMa 와 달리 연구/상업적 용도로 모두 무료이다.
- [Meta and Microsoft Introduce the Next Generation of Llama](https://ai.meta.com/blog/llama-2/)
- [https://labs.perplexity.ai/](https://labs.perplexity.ai/) 에서 Llama 2 를 Chat 으로 사용해볼 수 있다.
- 특징
    - 공개된 데이터셋에 대해서만 학습
    - llama 2 는 연구 또는 상업적 용도로 무료 (오픈소스)
    - 하이퍼 파라미터 규모에 따른 3가지 모델 제공
    - Chat Llama / Code Llama / [Llama Guard](https://github.com/facebookresearch/PurpleLlama/tree/main/Llama-Guard)
- 사용방법
    - https://ai.meta.com/llama/ 에 접속하여 모델 Access 신청
    - [llama github](https://github.com/facebookresearch/llama) 에서 직접 모델을 다운로드 하거나 [Hugging Face](https://huggingface.co/meta-llama) 를 통해서 접근이 가능하다.
        - Hugging Face 를 통한 접근은 Meta 에서 모델 Access 를 신청할 때, Hugging Face 와 동일한 이메일로 신청을 해야한다.
        - [Llama 2 is here - get it on Hugging Face](https://huggingface.co/blog/llama2)

## Alpaca

- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- 스탠퍼드 대학의 CRFM (Center for Research Foundation Models)는 학술 연구 목적으로 Meta의 LLaMA-7B 모델을 finetuning한 Alpaca라는 모델을 공개했다.


![](https://velog.velcdn.com/images/srk/post/1b60f88f-279e-4db0-abcd-3cfe7293cc88/image.png)
- pretrained language model : LLaMa 7B
- high-quality instruction-following data
    - [self-instruct](https://github.com/yizhongw/self-instruct) 를 시드로 GPT-3.5(text-davinci-003) 에게 ICL 로 전달하여 instructions 을 생성하도록 함 (self-instruction)

기존 Alpaca 의 경우, fine tuning 을 하기위해 요구되는 리소스 문제로 해당 문제를 우회하고자 `LoRA` 를 사용하는 개발이 진행되고 있다. 저자는 아래 글을 참고했다.
- https://github.com/tloen/alpaca-lora
- [alpaca-lora: 집에서 만든 대형 언어 모델 실험](https://hackernoon.com/ko/%EC%A7%91%EC%97%90%EC%84%9C-%EB%A7%8C%EB%93%A0-%EB%8C%80%ED%98%95-%EC%96%B8%EC%96%B4-%EB%AA%A8%EB%8D%B8%EC%9D%84-%EC%8B%A4%ED%97%98%ED%95%98%EB%8A%94-%EC%95%8C%ED%8C%8C%EC%B9%B4-%EB%A1%9C%EB%9D%BC)

## Vicuna

Google 이 언급한 Vicuna 도 LLaMa 에서 파생된 LLM 이다.
- `Vicuna-13B` 는 Meta의 LLaMA와 Stanford의 Alpaca에 영감을 받아 UC Berkeley, UCSD, CMU, MBZUAI(Mohamed Bin Zayed Univ. of AI)가 공동으로 개발한 오픈소스 챗봇으로 ShardGPT로 부터 수집된 사용자들의 대화로 **LLaMA를 fine-tuning한 모델**
    - [ShardGPT](https://sharegpt.com/) : 사용자 프롬프트와 ChatGPT의 해당 답변 결과를 서로 공유할 수 있는 웹사이트

![](https://velog.velcdn.com/images/srk/post/ebc3caae-4c00-416c-84af-64f4e3399e5a/image.png)

- pretrained language model : LLaMa 2
- high-quality instruction-following data : ShareGPT

[FastChat](https://github.com/lm-sys/FastChat) 에서 오픈소스로 사용할 수 있으며 Hugging Face 에서도 접근이 가능하다.

## 기타

Open source LLM 관련 벤치마크를 확인하기위해 아래를 참고할 수 있다.
- LLM을 위한 벤치마크 플랫폼인 Chatbot Arena
    - [https://arena.lmsys.org/](https://arena.lmsys.org/) 에서 확인 가능
    - [https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) 투표 결과는 여기서 확인 가능
- [https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [https://llm-leaderboard.streamlit.app/](https://llm-leaderboard.streamlit.app/)