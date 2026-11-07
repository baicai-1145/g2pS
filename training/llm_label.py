"""LLM polyphone labeler: deepseek-v4.1-flash via OpenAI-compatible endpoint.

Protocol (per spec):
- system prompt: hard constraints (mainland standard, literal reading, candidates only, JSON only)
- user prompt: JSON {sentence, targets:[{position, char, candidates}]}
- output: forced JSON, streaming, reasoning effort low
"""
import asyncio
import json
import os
import re
import sys

from openai import AsyncOpenAI

BASE_URL = 'https://newapi.baicai1145.com/v1'
API_KEY = os.environ.get('LLM_API_KEY', 'sk-4bZPpK7WXOCF54eZqcVpxLDEl9X2ZVxxXn2M9u5eL3JIBRup')
MODEL = 'deepseek-v4.1-flash'

SYSTEM_PROMPT = """你是中文多音字读音判定器。任务：给定句子和其中若干多音字的位置，为每个字选出正确拼音。

硬性约束：
1. 以中国大陆《现代汉语词典》现行规范读音为准。
2. 输出"字面音"（词典读音），不做口语变调：三声变调不标、"一/不"按本调（一 yi1、不 bu4）、轻声按词典标注（候选里有轻声才可选）。
3. 只能从每个字给出的候选读音中选择，禁止输出候选之外的拼音。若实在无法确定，选择语境中最可能的候选。
4. 仅输出 JSON，格式：{"results":[{"position":<int>,"char":"<字>","pinyin":"<拼音>"}]}，不要任何解释、markdown 或多余文本。"""


def build_user_prompt(sentence, targets):
    """targets: list of (position:int, char:str, candidates:list[str])"""
    obj = {
        'sentence': sentence,
        'targets': [
            {'position': p, 'char': c, 'candidates': cands}
            for p, c, cands in targets
        ],
    }
    return json.dumps(obj, ensure_ascii=False)


JSON_RE = re.compile(r'\{.*\}', re.S)


def parse_reply(text, targets):
    """validate + normalize LLM output; returns {position: pinyin} or None"""
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    results = data.get('results')
    if not isinstance(results, list):
        return None
    valid = {}
    cand_by_pos = {p: set(cands) for p, c, cands in targets}
    for r in results:
        try:
            pos = int(r.get('position'))
            py = str(r.get('pinyin', '')).strip()
        except (TypeError, ValueError):
            continue
        if pos in cand_by_pos and py in cand_by_pos[pos]:
            valid[pos] = py
    return valid if valid else None


class LLMLabeler:
    def __init__(self, concurrency=96, model=MODEL):
        self.client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120)
        self.sem = asyncio.Semaphore(concurrency)
        self.model = model
        self.n_call = self.n_ok = self.n_parse_fail = self.n_error = 0

    async def _call(self, sentence, targets, max_retry=3):
        extra = {'reasoning_effort': 'low'}
        for attempt in range(max_retry):
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': build_user_prompt(sentence, targets)},
                    ],
                    temperature=0.0,
                    stream=True,
                    extra_body=dict(
                        reasoning_effort='low',
                        response_format={'type': 'json_object'},
                    ),
                )
                chunks = []
                async for ev in stream:
                    if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                        chunks.append(ev.choices[0].delta.content)
                return ''.join(chunks)
            except Exception as e:
                if attempt == max_retry - 1:
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))

    async def label_sentence(self, sentence, targets):
        """returns {position: pinyin} | {} (parse fail) | None (call error)"""
        async with self.sem:
            self.n_call += 1
            try:
                text = await self._call(sentence, targets)
            except Exception as e:
                self.n_error += 1
                return None
        parsed = parse_reply(text, targets)
        if parsed is None:
            self.n_parse_fail += 1
            return {}
        self.n_ok += 1
        return parsed


async def _smoke():
    lab = LLMLabeler(concurrency=4)
    s = '他在银行工作，行踪不定。'
    targets = [
        (2, '行', ['hang2', 'xing2']),
        (6, '行', ['hang2', 'xing2']),
    ]
    out = await lab.label_sentence(s, targets)
    print('smoke result:', out, '| stats:', lab.n_call, lab.n_ok, lab.n_parse_fail, lab.n_error)


if __name__ == '__main__':
    asyncio.run(_smoke())
