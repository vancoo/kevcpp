import json, os, subprocess, time, sys

CFGDIR = r'E:\van\projects\kevcpp\config'
BASE = {
  "markers": {"state":"<|fim_prefix|>","question":"<|fim_middle|>","option_start":"<|box_start|>","option_end":"<|box_end|>","decide":"<|fim_suffix|>"},
  "temperature": 1.0,
  "logits": {"enable": True, "prompt_template": "", "aggregation": "combo", "logits_type": ""}
}

# 第二条细扫: 围绕最优"答案："做更短/更贴的变体
CASES = [
  ("答案：(全角, 基线)",  "{instructions}\n[ {options} ]\n{state}\n答案："),
  ("答案(无冒号)",        "{instructions}\n[ {options} ]\n{state}\n答案"),
  ("答案:(半角冒号)",      "{instructions}\n[ {options} ]\n{state}\n答案:"),
  ("正确答案：(4字)",      "{instructions}\n[ {options} ]\n{state}\n正确答案："),
  ("答案：\\n(换行)",     "{instructions}\n[ {options} ]\n{state}\n答案：\n"),
  ("正确选项：(4字)",      "{instructions}\n[ {options} ]\n{state}\n正确选项："),
  ("答：(1字)",           "{instructions}\n[ {options} ]\n{state}\n答："),
]

model = sys.argv[1]
port = int(sys.argv[2])
label = sys.argv[3]
os.chdir(r'E:\van\projects\kevcpp')
results = []
for i,(name,tpl) in enumerate(CASES):
    cfg = json.loads(json.dumps(BASE)); cfg['logits']['prompt_template'] = tpl
    cfg['_comment'] = f'细扫: {name}'
    cfgp = os.path.join(CFGDIR, f'_fine_{label}_{i}.json')
    with open(cfgp,'w',encoding='utf-8') as f: json.dump(cfg,f,ensure_ascii=False,indent=2)
    p = port + i
    env = dict(os.environ)
    for k in ('KEV_DECIDE','KEV_CALIB','KEV_TEMPLATE','KEV_ANSWERLABEL','KEV_LBL','KEV_ROTATE','KEV_HEAD','KEV_FORMULA','KEV_PMI','KEV_ENSEMBLE_TPL'):
        env.pop(k, None)
    proc = subprocess.Popen([r'build\bin\Release\kevcpp.exe','--model',model,'--backend','cpu','--port',str(p),
                             '--ctx','2048','--quiet','--decide-config',cfgp], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(60)
        out = os.path.join('build', f'correctness_{label}-fine-{i}.json')
        subprocess.run(['python','build/correctness_harness.py','--port',str(p),'--model',f'{label}-fine-{i}'],
                       capture_output=True, text=True, timeout=2400)
        acc='?'; br={}
        if os.path.exists(out):
            d=json.load(open(out,encoding='utf-8')); s=d['summary']; acc=s['accuracy']; br=s.get('by_qtype',{})
        results.append((name, acc, {k:round(v['acc'],3) for k,v in br.items()}))
        print(f'{name:20s} acc={acc}  {results[-1][2]}', flush=True)
    finally:
        proc.terminate()
        try: proc.wait(timeout=30)
        except Exception: proc.kill()
        time.sleep(3)

print('\n=== sorted ===')
for name, acc, br in sorted(results, key=lambda x: -(x[1] if isinstance(x[1],float) else 0)):
    print(f'{name:20s} {acc}  {br}')
