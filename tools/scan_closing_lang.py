import json, os, subprocess, time, sys

CFGDIR = r'E:\van\projects\kevcpp\config'
BASE = {
  "markers": {"state":"<|fim_prefix|>","question":"<|fim_middle|>","option_start":"<|box_start|>","option_end":"<|box_end|>","decide":"<|fim_suffix|>"},
  "temperature": 1.0,
  "logits": {"enable": True, "prompt_template": "", "aggregation": "combo", "logits_type": ""}
}

# 受控对照: 分离「语言(中/英)」与「形态(长度/标点/措辞)」两个因素
CASES = [
  # --- 组1: 等长, 仅语言不同 ---
  ("CN 答案：(2字+：)",        "{instructions}\n[ {options} ]\n{state}\n答案："),
  ("EN Ans：(3字母+:)",        "{instructions}\n[ {options} ]\n{state}\nAns:"),
  ("CN 结论：(2字+：)",        "{instructions}\n[ {options} ]\n{state}\n结论："),
  ("EN Yes：(3字母+:)",        "{instructions}\n[ {options} ]\n{state}\nYes:"),
  # --- 组2: 同语言, 不同长度/措辞 ---
  ("CN 是：(1字+：)",          "{instructions}\n[ {options} ]\n{state}\n是："),
  ("CN 答案为：(3字+：)",      "{instructions}\n[ {options} ]\n{state}\n答案为："),
  ("EN Answer：(6字母+:)",     "{instructions}\n[ {options} ]\n{state}\nAnswer:"),
  ("EN Result：(6字母+:)",     "{instructions}\n[ {options} ]\n{state}\nResult:"),
  # --- 组3: 同一语义的不同语言表达 ---
  ("CN 答案是：",              "{instructions}\n[ {options} ]\n{state}\n答案是："),
  ("EN Answer is:",           "{instructions}\n[ {options} ]\n{state}\nAnswer is:"),
  ("EN The answer is:",       "{instructions}\n[ {options} ]\n{state}\nThe answer is:"),
]

model = sys.argv[1] if len(sys.argv)>1 else r'D:\share-models\qwen3.5-4b-q6\Qwen3.5-4B-Q6_K.gguf'
port = int(sys.argv[2]) if len(sys.argv)>2 else 18400
label = sys.argv[3] if len(sys.argv)>3 else '4B'
os.chdir(r'E:\van\projects\kevcpp')
results = []
SKIP = {0, 1, 2, 3, 4, 5}   # 已完成 ctl-0..5, 只补跑 6..10(英文长句对照)
for i,(name,tpl) in enumerate(CASES):
    if i in SKIP: continue
    cfg = json.loads(json.dumps(BASE)); cfg['logits']['prompt_template'] = tpl
    cfg['_comment'] = f'收尾词中英对照: {name}'
    cfgp = os.path.join(CFGDIR, f'_ctl_{label}_{i}.json')
    with open(cfgp,'w',encoding='utf-8') as f: json.dump(cfg,f,ensure_ascii=False,indent=2)
    p = port + i
    env = dict(os.environ)
    for k in ('KEV_DECIDE','KEV_CALIB','KEV_TEMPLATE','KEV_ANSWERLABEL','KEV_LBL','KEV_ROTATE','KEV_HEAD','KEV_FORMULA','KEV_PMI','KEV_ENSEMBLE_TPL'):
        env.pop(k, None)
    proc = subprocess.Popen([r'build\bin\Release\kevcpp.exe','--model',model,'--backend','cpu','--port',str(p),
                             '--ctx','2048','--quiet','--decide-config',cfgp], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(22)
        out = os.path.join('build', f'correctness_{label}-ctl-{i}.json')
        subprocess.run(['python','build/correctness_harness.py','--port',str(p),'--model',f'{label}-ctl-{i}'],
                       capture_output=True, text=True, timeout=900)
        acc='?'; br={}
        if os.path.exists(out):
            d=json.load(open(out,encoding='utf-8')); acc=d['summary']['accuracy']
            for k in ('noul','choice','score'):
                if k in d['summary'].get('by_qtype',{}): br[k]=d['summary']['by_qtype'][k]['acc']
        results.append((name, acc, br)); print(f'{name:22s} acc={acc}  {br}', flush=True)
    finally:
        proc.terminate()
        try: proc.wait(timeout=20)
        except Exception: proc.kill()
        time.sleep(2)
print('\n=== sorted ===')
for name, acc, br in sorted(results, key=lambda x: -(x[1] if isinstance(x[1],float) else 0)):
    print(f'{name:22s} {acc}  {br}')
