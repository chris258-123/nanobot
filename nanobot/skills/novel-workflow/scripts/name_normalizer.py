"""Name normalizer for novel entity names.

Cleans dirty names from LLM extraction: splits parenthetical aliases,
filters noise entities, normalizes Unicode.
"""

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class NormalizedName:
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    is_valid: bool = True
    filter_reason: str = ""


# --- Configurable filter rules ---

# Generic role words (not real character names)
GENERIC_ROLES = {
    "母亲", "父亲", "妈妈", "爸爸", "爷爷", "奶奶", "外公", "外婆",
    "哥哥", "姐姐", "弟弟", "妹妹", "叔叔", "阿姨", "舅舅",
    "老师", "师父", "师傅", "徒弟", "弟子",
    "青梅竹马", "初恋", "前任", "恋人",
    "出租车师傅", "司机", "服务员", "老板", "店主",
    "路人", "行人", "旁观者", "围观者",
    "士兵", "卫兵", "守卫", "侍卫", "手下",
    "村民", "村人", "老人", "科研人员", "修女们",
    "女孩", "小女孩", "男孩", "小男孩", "年轻人", "少年", "少女",
    "怪人", "陌生人", "神秘人", "黑衣人", "蒙面人",
    "俊秀青年", "卷发歌手",
}

# Patterns that indicate noise entities
NOISE_PATTERNS = [
    re.compile(r'^[∈∉∋∌⊂⊃⊄⊅⊆⊇√∞≈≠≤≥±×÷∑∏∫∂∇]+'),  # math symbols
    re.compile(r'^[^\w\u4e00-\u9fff]+$'),  # pure non-word chars
    re.compile(r'未具名|未知|不明|匿名|无名'),  # unnamed
    re.compile(r'群体[）\)]?$'),  # group entities
    re.compile(r'^.{0,1}$'),  # too short (0-1 char)
]

# Bracket pairs for alias extraction
BRACKET_PAIRS = [
    ('（', '）'), ('(', ')'),
    ('【', '】'), ('「', '」'),
    ('《', '》'), ('〈', '〉'),
]

# Relationship type keyword mapping
RELATION_TYPE_MAP = {
    "ALLY": ["好友", "朋友", "信任", "伙伴", "盟友", "同盟", "搭档", "战友", "知己", "挚友"],
    "ENEMY": ["敌对", "仇人", "敌人", "宿敌", "死敌", "对立", "敌方", "仇敌"],
    "MENTOR": ["师父", "师傅", "弟子", "徒弟", "老师", "学生", "导师", "指导"],
    "FAMILY": ["父", "母", "兄", "弟", "姐", "妹", "家人", "亲人", "血亲",
               "儿子", "女儿", "丈夫", "妻子", "夫妻", "婚", "同胞"],
    "ROMANTIC": ["恋人", "爱人", "情侣", "暧昧", "喜欢", "爱慕", "倾心", "心仪"],
    "HIERARCHY": ["上级", "下属", "部下", "手下", "主人", "仆人", "奴隶",
                  "首领", "头目", "统领", "司令", "将军", "上司", "领导"],
    "COLLEAGUE": ["同事", "同僚", "同门", "同学", "同伴", "队友"],
    "RIVAL": ["竞争", "对手", "较量", "比试", "争夺"],
}

# All valid structured relationship types
VALID_RELATION_TYPES = set(RELATION_TYPE_MAP.keys()) | {"CO_PARTICIPANT", "ASSOCIATE"}


def normalize_unicode(text: str) -> str:
    """NFKC normalize + strip invisible chars."""
    text = unicodedata.normalize("NFKC", text)
    # Strip zero-width and BOM chars
    text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]', '', text)
    return text.strip()


def split_brackets(name: str) -> tuple[str, list[str]]:
    """Split parenthetical aliases from name.

    '荆璜（玄虹）' → ('荆璜', ['玄虹'])
    '拉戈维坦（星辰女王/银尾辉龙）' → ('拉戈维坦', ['星辰女王', '银尾辉龙'])
    """
    aliases = []
    canonical = name

    for open_b, close_b in BRACKET_PAIRS:
        pattern = re.compile(re.escape(open_b) + r'([^' + re.escape(close_b) + r']+)' + re.escape(close_b))
        matches = pattern.findall(canonical)
        for match in matches:
            # Split by / or 、 inside brackets
            parts = re.split(r'[/、，,]', match)
            aliases.extend(p.strip() for p in parts if p.strip())
        canonical = pattern.sub('', canonical).strip()

    return canonical, aliases


def split_slashes(name: str) -> tuple[str, list[str]]:
    """Split slash-separated names: '村民/老人' → ('村民', ['老人'])"""
    if '/' in name and '（' not in name and '(' not in name:
        parts = [p.strip() for p in name.split('/') if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1:]
    return name, []


def is_noise_entity(name: str) -> tuple[bool, str]:
    """Check if name is a noise entity that should be filtered."""
    # Check noise patterns
    for pattern in NOISE_PATTERNS:
        if pattern.search(name):
            return True, f"matches noise pattern: {pattern.pattern}"

    # Check generic roles
    if name in GENERIC_ROLES:
        return True, f"generic role: {name}"

    # Check if name is purely numeric
    if name.isdigit():
        return True, "purely numeric"

    return False, ""


def normalize_name(raw_name: str) -> NormalizedName:
    """Full normalization pipeline for a single name.

    Returns NormalizedName with canonical_name, aliases, validity.
    """
    if not raw_name or not raw_name.strip():
        return NormalizedName("", [], False, "empty name")

    # Step 1: Unicode normalize
    name = normalize_unicode(raw_name)

    # Step 2: Split brackets → extract aliases
    name, bracket_aliases = split_brackets(name)

    # Step 3: Split slashes
    name, slash_aliases = split_slashes(name)

    # Combine aliases, deduplicate
    all_aliases = []
    seen = {name}
    for a in bracket_aliases + slash_aliases:
        a = normalize_unicode(a)
        if a and a not in seen:
            all_aliases.append(a)
            seen.add(a)

    # Step 4: Check if noise (check both cleaned name and original)
    is_noise, reason = is_noise_entity(name)
    if is_noise:
        return NormalizedName(name, all_aliases, False, reason)

    # Also check original raw name for patterns like 科研人员（未具名）
    orig_normalized = normalize_unicode(raw_name)
    is_noise_orig, reason_orig = is_noise_entity(orig_normalized)
    if is_noise_orig:
        return NormalizedName(name, all_aliases, False, reason_orig)

    # Also filter aliases that are noise
    clean_aliases = []
    for a in all_aliases:
        noise, _ = is_noise_entity(a)
        if not noise:
            clean_aliases.append(a)

    return NormalizedName(name, clean_aliases, True, "")


def classify_relation_type(free_text: str) -> str:
    """Map free-text Chinese relationship description to structured type.

    '多年好友，关系密切' → 'ALLY'
    '敌对且被其击败的对手' → 'ENEMY'
    """
    text = free_text.lower()
    for rel_type, keywords in RELATION_TYPE_MAP.items():
        for kw in keywords:
            if kw in text:
                return rel_type
    return "ASSOCIATE"


# --- Batch operations ---

def normalize_entity_list(entities: list[dict]) -> tuple[list[dict], list[dict]]:
    """Normalize a list of entity dicts. Returns (valid, filtered).

    Each entity dict must have 'name' key. Will be enriched with
    'canonical_name', 'aliases', and original 'raw_name'.
    """
    valid = []
    filtered = []

    # Track canonical names to detect duplicates within batch
    seen_canonical: dict[str, int] = {}  # canonical_name → index in valid list

    for entity in entities:
        raw_name = entity.get("name", "")
        result = normalize_name(raw_name)

        if not result.is_valid:
            filtered.append({**entity, "raw_name": raw_name, "filter_reason": result.filter_reason})
            continue

        # Check for duplicate canonical name in this batch
        if result.canonical_name in seen_canonical:
            # Merge aliases into existing entity
            idx = seen_canonical[result.canonical_name]
            existing_aliases = set(valid[idx].get("aliases", []))
            existing_aliases.update(result.aliases)
            # Also add the raw_name as alias if different
            if raw_name != result.canonical_name and raw_name not in existing_aliases:
                existing_aliases.add(raw_name)
            valid[idx]["aliases"] = list(existing_aliases)
            continue

        # Also check if any alias matches an existing canonical
        merged = False
        for alias in result.aliases:
            if alias in seen_canonical:
                idx = seen_canonical[alias]
                existing_aliases = set(valid[idx].get("aliases", []))
                existing_aliases.update(result.aliases)
                existing_aliases.add(result.canonical_name)
                existing_aliases.discard(valid[idx]["name"])
                valid[idx]["aliases"] = list(existing_aliases)
                merged = True
                break

        if merged:
            continue

        entity_out = {
            **entity,
            "name": result.canonical_name,
            "raw_name": raw_name,
            "aliases": result.aliases,
        }
        seen_canonical[result.canonical_name] = len(valid)
        # Also index aliases
        for a in result.aliases:
            seen_canonical[a] = len(valid)
        valid.append(entity_out)

    return valid, filtered


if __name__ == "__main__":
    # Test with real dirty names from the dataset
    test_names = [
        "荆璜（玄虹）",
        "玄虹（荆璜）",
        "∈先生",
        "∈",
        "拉戈维坦（星辰女王/银尾辉龙）",
        "村民/老人",
        "科研人员（未具名）",
        "修女们（群体）",
        "蓝发女孩",
        "出租车师傅",
        "母亲",
        "罗彬瀚",
        "周雨",
        "弥娅",
    ]

    print("=== Name Normalization Test ===\n")
    for name in test_names:
        result = normalize_name(name)
        status = "✓" if result.is_valid else "✗"
        print(f"{status} '{name}' → canonical='{result.canonical_name}' aliases={result.aliases}"
              + (f" FILTERED: {result.filter_reason}" if not result.is_valid else ""))

    print("\n=== Relationship Type Classification ===\n")
    test_rels = [
        "多年好友，关系密切",
        "敌对且被其击败的对手",
        "律师，在国外，常要求他帮忙",
        "同母异父，麻烦不断",
        "信任的好友",
        "已失踪，关系密切",
        "绑架并审问他的敌对势力",
        "师父，传授武艺",
    ]
    for rel in test_rels:
        print(f"  '{rel}' → {classify_relation_type(rel)}")
