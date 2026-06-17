# =============================================================
# PCR — write-precision quantization lever (test-time, 790m)
# 두 번째 레버: 생존(Δ)이 아니라 "write 시점 정밀도"를 낮춤.
# conv1d 출력(토큰이 state로 써지는 표현)을 선택 위치에서 k-bit 양자화.
#
# 의존: model (MambaForCausalLM, slow path), DEVICE, tokenizer(tok/tokenizer)
# 기존 dt_proj 훅과 독립. MQAR는 self-contained라 base 수치는
# 기존 하베스트와 약간 다를 수 있음 — 보는 건 "곡선의 모양"(graceful?
# allocation?)이지 절대값이 아님.
# =============================================================
import torch, random

# tokenizer 이름 자동 감지
try:
    tokenizer
except NameError:
    tokenizer = tok  # noqa

# 기존 dt 훅이 살아있다면 gain 중립화 (양자화 레버만 격리)
try:
    for k in list(_CTRL.keys()):
        if k == 'gain_vec':
            _CTRL['gain_vec'] = None
    _CTRL['layers'] = None
    print("[info] _CTRL gain 중립화 (양자화 레버 격리)")
except NameError:
    pass

# ---- 1. fake-quant: 마지막 차원(채널 벡터)을 k-bit uniform 양자화 ----
def fake_quant_vec(v, bits):
    if bits >= 16:
        return v
    qmax = (2 ** bits) - 1
    lo = v.amin(dim=-1, keepdim=True)
    hi = v.amax(dim=-1, keepdim=True)
    scale = (hi - lo).clamp(min=1e-8) / qmax
    return torch.round((v - lo) / scale) * scale + lo

# ---- 2. 양자화 훅: conv1d 출력의 선택 위치를 k-bit로 ----
_QMASK = {'pos': None, 'bits': 16, 'fired': 0}

def _qhook(mod, inp, out):
    if _QMASK['bits'] >= 16 or not _QMASK['pos']:
        return out
    L = out.shape[-1]
    pos = [p for p in _QMASK['pos'] if 0 <= p < L]
    if not pos:
        return out
    o = out.clone()
    b = _QMASK['bits']
    for p in pos:
        o[:, :, p] = fake_quant_vec(o[:, :, p], b)   # (B, d_inner) 채널벡터 양자화
    _QMASK['fired'] += 1
    return o

_qhandles = []
def install_quant_hooks():
    global _qhandles
    for h in _qhandles:
        h.remove()
    _qhandles = []
    for lyr in model.backbone.layers:
        _qhandles.append(lyr.mixer.conv1d.register_forward_hook(_qhook))
    print(f"[ok] quant hooks on {len(_qhandles)} conv1d")

def verify_quant_hooks():
    _QMASK['pos'] = [1]; _QMASK['bits'] = 2; _QMASK['fired'] = 0
    ids = tokenizer(" the cat sat on the mat", return_tensors='pt').input_ids.to(DEVICE)
    with torch.no_grad():
        model(ids)
    _QMASK['pos'] = None; _QMASK['bits'] = 16
    print(f"[verify] hook fired {_QMASK['fired']}x (기대 {len(model.backbone.layers)})")
    return _QMASK['fired'] == len(model.backbone.layers)

# ---- 3. single-token 단어 풀 ----
def single_token_words(n, seed=0):
    rng = random.Random(seed)
    ids = list(range(1000, 45000)); rng.shuffle(ids)
    out = []
    for i in ids:
        s = tokenizer.decode([i]).strip()
        if s.isascii() and s.isalpha() and 3 <= len(s) <= 8:
            if len(tokenizer(" " + s, add_special_tokens=False).input_ids) == 1:
                out.append(s)
        if len(out) >= n:
            break
    return out

_WORDS = single_token_words(4000)
print(f"[ok] single-token 단어 {len(_WORDS)}개")

# ---- 4. MQAR 1회: 첫 쌍(target) 질의. 각 토큰의 위치 기록 ----
def build_mqar(N, rng):
    ws = rng.sample(_WORDS, 2 * N)
    pairs = [(ws[2 * j], ws[2 * j + 1]) for j in range(N)]
    toks, pos_of_pair = [], []
    for (k, v) in pairs:
        kp = len(toks); toks.append(" " + k)
        vp = len(toks); toks.append(" " + v)
        pos_of_pair.append((kp, vp))
    qk = pairs[0][0]; qv = pairs[0][1]
    toks.append(" ?"); toks.append(" " + qk)
    prompt = "".join(toks)
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
    # 위치 보정: tokenizer가 1토큰/단어라는 가정 하에 위치=토큰인덱스
    tgt_id = tokenizer(" " + qv, add_special_tokens=False).input_ids[0]
    return ids, pos_of_pair, tgt_id

def _recall_once(ids, tgt_id):
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    return int(logits.argmax().item() == tgt_id)

# ---- 5. 실험 A: target을 k-bit로 → recall (정밀도→충실도 곡선) ----
def rd_target(bits, N, n=30, seed0=0):
    hit = 0
    for t in range(n):
        rng = random.Random(seed0 + t)
        ids, pp, tgt = build_mqar(N, rng)
        _QMASK['pos'] = list(pp[0]); _QMASK['bits'] = bits   # 첫 쌍(target) 위치
        hit += _recall_once(ids, tgt)
    _QMASK['pos'] = None; _QMASK['bits'] = 16
    return hit / n

# ---- 6. 실험 B: 비-target(저우선)을 k-bit로 → target recall (중요도 배분) ----
def rd_distractor(bits, N, n=30, seed0=0):
    hit = 0
    for t in range(n):
        rng = random.Random(seed0 + t)
        ids, pp, tgt = build_mqar(N, rng)
        dpos = [p for pr in pp[1:] for p in pr]    # 나머지 쌍 전부
        _QMASK['pos'] = dpos; _QMASK['bits'] = bits
        hit += _recall_once(ids, tgt)
    _QMASK['pos'] = None; _QMASK['bits'] = 16
    return hit / n

# =============================================================
# 실행
# =============================================================
install_quant_hooks()
print("verify:", verify_quant_hooks())

# base N* 잡기 (양자화 off)
print("\n[base vs N]")
for N in [4, 8, 16, 24, 32]:
    print(f"  N={N}: base={rd_target(16, N, n=20):.2f}")

N = 16   # base가 ~0.4~0.6 되는 값으로 조정
print(f"\n[A] target을 k-bit 양자화 (N={N}) — 정밀도→recall 곡선")
for b in [16, 8, 4, 2, 1]:
    print(f"  {b:>2}-bit: target_recall={rd_target(b, N, n=30):.2f}")

print(f"\n[B] 저우선(나머지 쌍) k-bit 양자화 (N={N}) — 중요도 배분")
print(f"  off(16): target_recall={rd_distractor(16, N, n=30):.2f}")
for b in [4, 2, 1]:
    print(f"  {b:>2}-bit: target_recall={rd_distractor(b, N, n=30):.2f}")
