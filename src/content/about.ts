export type AboutHighlight = { label: string; value: string; desc: string };
export type AboutStrength = { key: string; title: string; desc: string };
export type AboutTalk = {
  date: string;
  title: string;
  venue: string;
  detail?: string;
  link?: string;
  image?: string;
  isBook?: boolean;
};
export type AboutSkillRow = { cat: string; items: string[] };

export const ABOUT = {
  name: "박슬기",
  role: "DATA ENGINEER · AI TOOLS BUILDER",
  email: "qud9787@gmail.com",
  github: "https://github.com/parkseulkee",
  githubLabel: "github.com/parkseulkee",
  linkedin: "https://www.linkedin.com/in/seulkeepark1709/",
  linkedinLabel: "linkedin/seulkeepark1709",
  intro:
    "네이버에서 8년간 커머스 데이터를 기반으로 Data Product를 설계하고 운영해 왔습니다. 최근에는 AI 기술을 실제 업무 도구로 만들어 팀의 생산성을 높이는 데 집중하고 있습니다.",
  highlights: [
    { label: "경력", value: "8년+", desc: "네이버 데이터 엔지니어" },
    { label: "저서", value: "1권", desc: "LLMOps 기술서 출간" },
    { label: "발표", value: "5회+", desc: "사내외 기술 발표" },
    { label: "전문", value: "Data+AI", desc: "Product · Agent · LLMOps" },
  ] as AboutHighlight[],
  strengths: [
    {
      key: "Data Product",
      title: "데이터를 제품으로 만드는 설계력",
      desc: "복잡한 이커머스 데이터(클릭·결제·광고·노출)를 판매/유입 축으로 정리해 End-to-End Data Product를 설계합니다.",
    },
    {
      key: "AI · 실행력",
      title: "AI를 도구로 만드는 실행력",
      desc: "LLM 리포트, MCP 분석 Agent, 프롬프트 엔지니어링 도구까지 — 새 기술을 빠르게 업무 도구로 만들어 팀 생산성을 높입니다.",
    },
    {
      key: "공유",
      title: "공유와 커뮤니케이션",
      desc: "팀 내 도입 → 사내 Tech Meetup → 외부 세미나 → 기술서 출간까지, 실무를 체계화해 나눕니다.",
    },
  ] as AboutStrength[],
  talks: [
    {
      date: "2025.07",
      title: "AI 생산성 향상 외부 세미나",
      venue: "대웅제약 IDSTrust · 연사",
    },
    {
      date: "2025.06",
      title: "아이언맨 비긴즈 2 — MCP와 Agent 구축",
      venue: "Tech Meetup · 사내",
    },
    {
      date: "2025.04",
      title: "LLMOps를 활용한 LLM 엔지니어링",
      venue: "위키북스",
      link: "https://www.yes24.com/product/goods/145341599",
      isBook: true,
    },
    {
      date: "2025.04",
      title: "아이언맨 비긴즈 — MCP로 일도 정보도 더 스마트하게",
      venue: "Tech Meetup · 사내",
    },
    {
      date: "2024.08",
      title: "LLMOps를 위한 프롬프트 엔지니어링 도구",
      venue: "D2 & NE",
      link: "https://d2.naver.com/helloworld/3344073",
    },
    {
      date: "2023.11",
      title: "Data Quality로 data downtime 줄이기",
      venue: "D2 & NE",
      link: "https://d2.naver.com/helloworld/5766317",
    },
  ] as AboutTalk[],
  skills: [
    { cat: "Languages", items: ["Scala", "Python"] },
    { cat: "Data Eng", items: ["Spark", "Airflow", "HDFS", "Elasticsearch"] },
    { cat: "Data Quality", items: ["Great Expectations", "Datahub"] },
    { cat: "AI · LLM", items: ["LangGraph", "LangChain", "MCP", "프롬프트"] },
    { cat: "Infra", items: ["Spring", "FastAPI", "Kubernetes", "S3"] },
  ] as AboutSkillRow[],
};
