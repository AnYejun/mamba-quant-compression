# Priority-Conditioned Retention (PCR)

#### 성균관대학교 2026-1 AI시스템 기말 프로젝트 · [프로젝트 페이지](https://anyejun.github.io/mamba-quant-compression/site/)
Pre-trained Mamba 안의 제어 가능한 중간층 기억 회로


## 한 줄 요약

Mamba의 recall 실패는 저장 용량만의 문제가 아니라 제어의 문제이기도 하다. 재학습 없이
타깃 이후 위치의 Δ 게이트만 줄이면(`g<1`) decay가 느려져 capacity 압박 하의
recall이 회복된다. 또, 이 제어는 48개 Layer 중 **L19–22 4개 Layer**에 몰려 있다.


## 구조

```
notebooks/  실험 노트북
paper/      논문
docs/       마크다운 논문 + 기술 보고서
slides/     발표 자료
site/       랜딩 페이지 (GitHub Pages)
```

## 실행

Colab L4 GPU 기준 약 30분

```bash
pip install -r requirements.txt
```

`notebooks/PCR_from_scratch.ipynb`를 열어 전체 셀 실행
`mamba-ssm` / `causal-conv1d`는 설치하지 말 것!


## 라이선스

MIT - [LICENSE](LICENSE)
