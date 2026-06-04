# Mamba: The Quant Compression

> **연속-상태 시퀀스 모델(Mamba)의 중요도 인지 용량 배분** — KV-캐시 압축 프론티어를 SSM으로 일반화한 성균관대 2026-1 AI 시스템 팀 프로젝트 홈페이지.

## 🔗 배포 링크 (GitHub Pages)

**https://anyejun.github.io/mamba-quant-compression/**

- 저장소: https://github.com/AnYejun/mamba-quant-compression

## 👥 팀

성균관대학교 2026-1 AI 시스템

| 이름 | 역할 |
|------|------|
| 안예준 | Lead · Theory & Framing (통합 축·용량 바운드·논문 프레이밍) |
| 정찬희 | Experiments & Data Harness (priority-coupled MQAR·학습·통제) |
| 김수윤 | Mechanism & Evaluation (Δ-gate probe·Belady-MIN·평가) |

## ✅ 구현한 기능 체크리스트

**Git / GitHub 사용**
- [x] 커밋 3회 이상 (`init` → `feat: about` → `style` → `docs`)
- [x] 브랜치 생성 (`feature/about`) 후 작업
- [x] Pull Request 생성 ([#1](https://github.com/AnYejun/mamba-quant-compression/pull/1)) 후 Merge

**페이지 구성 (정적 HTML/CSS/JS)**
- [x] 소개(About) — 팀원 소개 + 역할 분담 카드
- [x] 프로젝트(Project) — 기말 프로젝트 소개 (문제·방법·가설·로드맵)
- [x] 연락(Contact) — 이메일 / GitHub 링크
- [x] 상단 내비게이션 — About / Project / Contact 앵커 링크 3개 이상
- [x] 스크롤 시 active 섹션 하이라이트, 반응형 모바일 메뉴, reveal 애니메이션

**배포**
- [x] GitHub Pages로 배포

## 🛠 기술 스택

순수 HTML + CSS + Vanilla JS (빌드 과정 없음). 다크 미니멀 테마, `IntersectionObserver` 기반 스크롤 인터랙션.

```
.
├── index.html        # 단일 페이지 (nav · hero · about · project · contact)
├── css/style.css     # 다크 미니멀 테마
├── js/main.js        # 스크롤 스파이 · 모바일 메뉴 · reveal
└── README.md
```

## 💡 어려웠던 점 / 배운 점

- **Git 협업 흐름**: 브랜치 → PR → Merge 라는 실제 협업 워크플로우를 처음 끝까지 돌려보며, 단순 커밋이 아니라 변경을 리뷰 단위로 분리하는 감각을 익혔다.
- **연구 내용의 시각적 번역**: 통합 축·용량 임계점 같은 추상적 개념을 카드/수식 블록으로 압축해 한눈에 들어오게 만드는 정보 설계가 생각보다 어려웠다.
- **프레임워크 없는 정적 사이트**: 빌드 도구 없이 `IntersectionObserver`만으로 스크롤 하이라이트와 등장 애니메이션을 구현하며, 가벼운 사이트에는 바닐라가 충분하다는 걸 체감했다.
