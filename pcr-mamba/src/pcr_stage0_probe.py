# =============================================================
# Fork 2 · Stage 0 — storage-vs-selectivity PROBE  (gate)
# PCR_from_scratch.ipynb 뒤에 붙여 실행.
# 의존: model, _WORDS, _enc, build_mqar, _recall_once, base_recall,
#       DEVICE, NL, random, torch  (전부 그 노트북에 정의돼 있음)
#
# 질문: recall이 터질 때 state는 '꽉 찬' 것(storage)인가 '슬랙'(control)인가?
#  - effective rank가 낮은데 recall 실패 → control/간섭 한계(고칠 수 있음) = thesis A ✓
#  - effective rank가 포화하며 recall 실패 → storage 한계 = thesis A 기각 → Fork 1
# =============================================================
import torch, random

# context(쌍들)만 인코딩 — 질의 토큰 없이 '저장된 상태'를 본다
def build_context(N, rng):
    ws = rng.sample(_WORDS, 2 * N)
    pairs = [(ws[2*j], ws[2*j+1]) for j in range(N)]
    seq = []
    for (k, v) in pairs:
        seq += _enc(k); seq += _enc(v)
    return torch.tensor([seq], device=DEVICE), pairs

def _get_ssm_states(cache):
    ss = cache.ssm_states          # 보통 (NL,B,d_inner,d_state) 텐서 or list/dict
    return ss                      # ss[layer] 로 인덱싱

# --- 진단: 캐시 구조/shape 먼저 확인 (API 버전차 대비) ---
ctx, _ = build_context(8, random.Random(0))
with torch.no_grad():
    out = model(ctx, use_cache=True)
ss = _get_ssm_states(out.cache_params)
try:
    print("ssm_states type:", type(ss).__name__,
          "| layer0 shape:", tuple(ss[0].shape))
except Exception as e:
    print("구조 다름 — 확인 필요:", e)

# --- (0a) effective rank (participation ratio) vs N ---
# PR = (Σσ)² / Σσ²  → 실제로 쓰는 차원 수. d_state(=16) 대비 얼마나 차나.
def state_effrank(N, layers, n=20, seed0=0):
    acc = {l: 0.0 for l in layers}
    for t in range(n):
        ctx, _ = build_context(N, random.Random(seed0 + t))
        with torch.no_grad():
            out = model(ctx, use_cache=True)
        ss = _get_ssm_states(out.cache_params)
        for l in layers:
            M = ss[l][0].float()                     # (d_inner, d_state)
            s = torch.linalg.svdvals(M)
            pr = (s.sum() ** 2 / (s.square().sum() + 1e-9)).item()
            acc[l] += pr
    return {l: acc[l] / n for l in layers}

probe_layers = [4, 12, 20, 28, 40]                   # L20 = 회로 한가운데
d_state = _get_ssm_states(out.cache_params)[0].shape[-1]
print(f"\nd_state = {d_state}  (effective rank 상한)")
print("[effective rank vs N]  (recall과 함께)")
print("  N |  recall |  " + " ".join(f"L{l:>2}" for l in probe_layers))
for N in [8, 16, 24, 32, 48, 64]:
    er = state_effrank(N, probe_layers, n=20)
    rec = base_recall(N, n=20)
    print(f"  {N:>2} |  {rec:.2f}   | " + " ".join(f"{er[l]:4.1f}" for l in probe_layers))

print("""
[해석]
- recall이 떨어지는 N에서 effective rank가 아직 d_state보다 한참 낮다
    → state는 '슬랙' → 저장 한계 아님 → control/간섭 한계 (thesis A ✓, Fork 2 진행)
- effective rank가 d_state 근처로 포화하는 지점과 recall 붕괴가 일치
    → state가 꽉 참 → storage 한계 (thesis A 기각 → Fork 1로 후퇴)
""")

# --- (0b, 선택·강버전) state 양자화 b*: 저장된 state를 k-bit로 → recall ---
# context를 캐시로 만든 뒤 state를 양자화하고 질의를 이어서 읽는다.
# cache continuation API가 버전따라 까다로우면 (0a)만으로도 게이트 판정 충분.
def _q_state(M, bits):
    if bits >= 16: return M
    qmax = (2 ** bits) - 1
    lo = M.amin(-1, keepdim=True); hi = M.amax(-1, keepdim=True)
    scale = (hi - lo).clamp(min=1e-8) / qmax
    return torch.round((M - lo) / scale) * scale + lo

def state_quant_recall(bits, N, n=30, seed0=0):
    hit = 0
    for t in range(n):
        rng = random.Random(seed0 + t)
        ctx, pairs = build_context(N, rng)
        with torch.no_grad():
            out = model(ctx, use_cache=True)
        cache = out.cache_params
        ss = _get_ssm_states(cache)
        for l in range(NL):                          # 모든 레이어 state 양자화
            ss[l][:] = _q_state(ss[l].float(), bits).to(ss[l].dtype)
        qk = pairs[0][0]; qv = pairs[0][1]
        q_ids = torch.tensor([_enc(" ?".strip()) if False else
                              (tokenizer(" ?", add_special_tokens=False).input_ids + _enc(qk))],
                             device=DEVICE)
        pos = torch.arange(ctx.shape[1], ctx.shape[1] + q_ids.shape[1], device=DEVICE)
        try:
            with torch.no_grad():
                o2 = model(q_ids, use_cache=True, cache_params=cache, cache_position=pos)
            pred = o2.logits[0, -1].argmax().item()
            hit += int(pred == _enc(qv)[0])
        except Exception as e:
            print("continuation API 이슈 — (0a)로 판정하셈:", e); return None
    return hit / n

print("[state quant b*]  (저장 state를 k-bit로 → target recall)")
for b in [16, 8, 4, 2, 1]:
    r = state_quant_recall(b, 32, n=30)
    if r is None: break
    print(f"  {b:>2}-bit: {r:.2f}")
print("  낮은 b*에도 recall 유지 → state 정보량 적음(슬랙) → control 한계 ✓")
