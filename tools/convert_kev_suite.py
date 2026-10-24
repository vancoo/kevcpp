"""把 kev 官方评测集 (evals/<suite>/*.jsonl, {state, questions:{key:{type,instructions,criteria,label}}})
转换成 kevcpp harness 的输入格式 ({id, category, state, questions, expected={key:label}})。

用法:
  python build/convert_kev_suite.py <kev_repo> <suite> <split> <out.jsonl> [--limit N]
例:
  python build/convert_kev_suite.py E:\van\projects\kev hard-v1 test build/kev-hard-v1-test.jsonl
"""
import json, sys, os

def main():
    repo, suite, split, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    limit = None
    if '--limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit') + 1])

    src = os.path.join(repo, 'evals', suite, f'{split}.jsonl')
    if not os.path.exists(src):
        print(f'ERROR: not found: {src}', file=sys.stderr); sys.exit(2)

    n_rec = n_q = 0
    skipped = []
    with open(src, encoding='utf-8') as fin, open(out, 'w', encoding='utf-8') as fout:
        for ln in fin:
            if not ln.strip():
                continue
            rec = json.loads(ln)
            meta = rec.get('_meta') or {}
            questions, expected = {}, {}
            for qkey, q in (rec.get('questions') or {}).items():
                qtype = q.get('type')
                crit = q.get('criteria')
                label = q.get('label')
                # harness 支持 score(int) / choice(str key) / noul(bool)。
                # kev 的 choice label 为 criteria 的键; score label 为整数; noul 无 criteria, label 为 bool。
                if qtype == 'choice':
                    if not isinstance(crit, dict) or label not in crit:
                        skipped.append((meta.get('id'), qkey, 'choice:label-not-in-criteria')); continue
                elif qtype == 'score':
                    if not isinstance(crit, list) or not isinstance(label, int):
                        skipped.append((meta.get('id'), qkey, 'score:bad-criteria-or-label')); continue
                elif qtype == 'noul':
                    if not isinstance(label, bool):
                        # 有些 noul 的 label 可能是 0/1
                        if isinstance(label, int) and label in (0, 1):
                            label = bool(label)
                        else:
                            skipped.append((meta.get('id'), qkey, 'noul:bad-label')); continue
                else:
                    skipped.append((meta.get('id'), qkey, f'unknown-type:{qtype}')); continue
                qq = {'type': qtype, 'instructions': q.get('instructions', '')}
                if crit is not None:
                    if qtype == 'choice' and isinstance(crit, dict):
                        # kev 有些数据集的 criteria 是 {key: null}, 表示"选项描述就是 key 本身"。
                        # harness/服务器需要非空字符串描述, 故把 null 替换为键名(与 kev 判定语义一致:
                        # label 是键, 描述缺省时用键当作描述)。
                        crit = {k: (v if isinstance(v, str) else k) for k, v in crit.items()}
                    qq['criteria'] = crit
                questions[qkey] = qq
                expected[qkey] = label
            if not questions:
                continue
            # 保持 harness 的确定性顺序(同一条记录内按 questions 出现顺序)
            out_rec = {
                'id': meta.get('id') or f'{suite}-{n_rec}',
                'category': meta.get('family') or meta.get('source') or suite,
                'state': rec.get('state', ''),
                'questions': questions,
                'expected': expected,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + '\n')
            n_rec += 1
            n_q += len(questions)
            if limit and n_rec >= limit:
                break

    print(f'wrote {out}: {n_rec} records, {n_q} questions')
    if skipped:
        print(f'skipped {len(skipped)} questions; first 5:')
        for s in skipped[:5]:
            print('  ', s)

if __name__ == '__main__':
    main()
