"""Full-sentence pinyin with g2pS (numpy-only) + pypinyin fallback for monophonic chars."""
from g2ps import G2pS
import pypinyin

g = G2pS('models/nano')

def full_pinyin(text):
    used, pairs = g.convert(text)
    return [p or pypinyin.pinyin(c, style=pypinyin.Style.TONE3, heteronym=False)[0][0]
            for c, p in pairs]

print(full_pinyin('他在银行工作，行踪不定。'))
