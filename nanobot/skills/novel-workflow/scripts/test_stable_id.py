#!/usr/bin/env python3
"""测试 UUIDv5 ID 生成的稳定性"""

import uuid

# 使用与 embedder_parallel.py 相同的命名空间
ASSET_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "nanobot.novel.assets")


def stable_point_id(book_id: str, chapter: str, asset_type: str, asset_key: str) -> str:
    """Generate stable, deterministic point ID using UUIDv5."""
    composite_key = f"{book_id}|{chapter}|{asset_type}|{asset_key}"
    return uuid.uuid5(ASSET_NAMESPACE, composite_key).hex


# 测试：同样的输入应该产生同样的 ID
print("=" * 60)
print("测试 UUIDv5 ID 稳定性")
print("=" * 60)

test_cases = [
    ("novel_04", "第1章 梨海市宗教文化交流节（上）", "plot_beat", "0"),
    ("novel_04", "第1章 梨海市宗教文化交流节（上）", "plot_beat", "1"),
    ("novel_04", "第1章 梨海市宗教文化交流节（上）", "character_card", "0"),
    ("novel_04", "第2章 梨海市宗教文化交流节（中）", "plot_beat", "0"),
]

print("\n第一次运行：")
ids_run1 = []
for book_id, chapter, asset_type, asset_key in test_cases:
    point_id = stable_point_id(book_id, chapter, asset_type, asset_key)
    ids_run1.append(point_id)
    print(f"  {asset_type}[{asset_key}]: {point_id}")

print("\n第二次运行（应该完全相同）：")
ids_run2 = []
for book_id, chapter, asset_type, asset_key in test_cases:
    point_id = stable_point_id(book_id, chapter, asset_type, asset_key)
    ids_run2.append(point_id)
    print(f"  {asset_type}[{asset_key}]: {point_id}")

print("\n验证结果：")
if ids_run1 == ids_run2:
    print("✅ 成功！两次运行产生完全相同的 ID")
else:
    print("❌ 失败！ID 不稳定")

print("\n验证唯一性：")
if len(set(ids_run1)) == len(ids_run1):
    print("✅ 成功！所有 ID 都是唯一的")
else:
    print("❌ 失败！存在 ID 碰撞")

print("\n" + "=" * 60)
print("对比旧方法（Python hash）的问题：")
print("=" * 60)

# 演示 Python hash 的不稳定性
import os
import subprocess

print("\n使用 Python hash() 的问题演示：")
print("（注意：由于 hash 随机化，这个演示可能在某些环境下看不到差异）")

test_str = "novel_04_第1章_plot_beat_0"
hash_val = abs(hash(test_str)) % (10**15)
print(f"当前进程的 hash 值: {hash_val}")

# 在子进程中计算 hash（可能不同）
code = f"""
import sys
test_str = "{test_str}"
hash_val = abs(hash(test_str)) % (10**15)
print(hash_val)
"""

result = subprocess.run(
    ["python", "-c", code],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    subprocess_hash = int(result.stdout.strip())
    print(f"子进程的 hash 值: {subprocess_hash}")

    if hash_val == subprocess_hash:
        print("⚠️  在这个环境中 hash 值相同（可能禁用了随机化）")
    else:
        print("❌ hash 值不同！这会导致 ID 不稳定")
else:
    print("⚠️  无法运行子进程测试")

print("\n结论：")
print("- UUIDv5: 稳定、确定性、跨进程/跨机器一致")
print("- Python hash(): 不稳定、可能随机化、不适合持久化 ID")
