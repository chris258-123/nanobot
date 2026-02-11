---
name: novel-crawler
description: 小说网站爬取技能。当用户需要从小说网站爬取章节内容时使用这个技能。技能包含：使用浏览器工具提取章节列表，使用 Playwright 爬虫批量爬取章节内容，以及自动整理去重和排序章节文件。
---

# 小说爬虫技能

这个技能用于从小说网站完整爬取小说章节，并自动整理为有序的 Markdown 文件。

## 使用场景

- 用户需要从小说网站下载完整小说时
- 用户需要批量爬取多个章节时
- 用户需要整理已爬取的章节文件时

## 工作流程

### 第一步：提取章节列表

使用 `agent-browser` 工具从小说网站的目录页提取所有章节链接。

#### 单个目录页提取

```bash
# 打开目录页
agent-browser open "https://www.example.com/book/12345/"
sleep 2

# 提取章节链接
agent-browser snapshot | \
    grep -A 2 "link.*第.*章" | \
    grep "url:" | \
    sed 's/.*url: //' | \
    sort -u > /tmp/chapters.txt

# 关闭浏览器
agent-browser close

# 查看提取结果
echo "找到 $(wc -l < /tmp/chapters.txt) 个章节"
```

#### 多个目录页批量提取

使用 [`extract_chapters.sh`](./extract_chapters.sh) 脚本：

```bash
# 用法: bash extract_chapters.sh <base_url> <start_page> <end_page>
bash extract_chapters.sh "https://www.22biqu.com/biqu7034" 2 9

# 查看提取结果
wc -l /tmp/all_chapters.txt
```

### 第二步：批量爬取章节

使用 [`crawl_novel.sh`](./crawl_novel.sh) 批量爬取脚本：

```bash
# 确保章节列表已准备好
ls -lh /tmp/all_chapters.txt

# 开始爬取
bash crawl_novel.sh

# 查看进度
tmux attach -t novel_crawler

# 分离会话: Ctrl+B 然后 D
```

**脚本特点：**
- 在 tmux 后台运行，可随时查看进度
- 自动重命名章节文件（使用章节标题）
- 失败重试和错误处理
- 实时进度监控

**核心爬虫：** [`crawl-chapter-playwright.js`](./crawl-chapter-playwright.js)
- 60 秒超时，适应慢速网站
- 自动设置 User-Agent 避免被识别为爬虫
- 多选择器策略，适配不同网站结构
- 自动清理广告和乱码文本
- 输出标准 Markdown 格式

### 第三步：整理章节文件

使用 [`organize_chapters.py`](./organize_chapters.py) 整理脚本：

```bash
# 整理章节：去重、排序、重命名
python3 organize_chapters.py
```

**整理功能：**
- 删除重复章节（基于标题）
- 按章节号排序
- 番外章节自动放在最后
- 重命名为 `0001_`, `0002_` 格式
- 清理文件名中的非法字符

## 爬虫参数说明

### Playwright 配置

| 参数 | 说明 | 默认值 |
|---|---|---|
| `headless` | 无头模式 | `true` |
| `timeout` | 页面加载超时 | `60000ms` |
| `waitUntil` | 等待策略 | `domcontentloaded` |
| `User-Agent` | 浏览器标识 | Chrome 120 |

### 批量爬取配置

| 参数 | 说明 | 默认值 |
|---|---|---|
| `OUTPUT_DIR` | 输出目录 | `~/novel_data` |
| `TMUX_SESSION` | tmux 会话名 | `novel_crawler` |
| `sleep` | 请求间隔（秒） | `1` |

## 使用示例

### 完整流程示例

```bash
# 1. 安装依赖
npm install playwright

# 2. 提取章节列表
bash extract_chapters.sh "https://www.22biqu.com/biqu7034" 2 9

# 3. 开始爬取
bash crawl_novel.sh

# 4. 查看进度
tmux attach -t novel_crawler

# 5. 等待完成后整理
python3 organize_chapters.py
```

### 单页爬取示例

```bash
# 提取单页章节
agent-browser open "https://www.22biqu.com/biqu7034/"
sleep 2
agent-browser snapshot | grep -A 2 "link.*第.*章" | grep "url:" | sed 's/.*url: //' > /tmp/all_chapters.txt
agent-browser close

# 开始爬取
bash crawl_novel.sh
```

## 技能资源

### 脚本文件
- [`crawl-chapter-playwright.js`](./crawl-chapter-playwright.js) - Playwright 单章节爬虫
- [`crawl_novel.sh`](./crawl_novel.sh) - 批量爬取脚本
- [`extract_chapters.sh`](./extract_chapters.sh) - 多页章节提取脚本
- [`organize_chapters.py`](./organize_chapters.py) - 章节整理脚本

### 输出文件
- `~/novel_data/*.md` - 爬取的章节文件
- `/tmp/all_chapters.txt` - 章节链接列表
- `/tmp/crawl_task.sh` - 临时爬取任务脚本

## 注意事项

1. **速度控制**：脚本中使用 `sleep 1` 避免请求过快被封，可根据网站情况调整
2. **User-Agent**：已设置浏览器标识避免被识别为爬虫，必要时可更换
3. **超时设置**：默认 60 秒超时，慢速网站可能需要增加
4. **选择器适配**：不同网站的 HTML 结构不同，需要调整 CSS 选择器
5. **内容清理**：脚本会自动清理常见广告文本，特殊情况需手动添加规则
6. **文件命名**：章节文件名会自动清理非法字符，保持跨平台兼容
7. **去重逻辑**：基于章节标题去重，保留文件名最短的版本
8. **番外处理**：番外章节自动识别并放在正文之后
9. **tmux 使用**：爬取任务在 tmux 后台运行，可随时查看进度或分离会话
10. **依赖检查**：确保已安装 Node.js、Playwright、agent-browser 和 Python 3

## 常见问题

**Q: 爬取失败率高怎么办？**

A:
- 增加超时时间（修改 `crawl-chapter-playwright.js` 中的 `page.setDefaultTimeout()`）
- 检查 CSS 选择器是否正确
- 确认网站是否有反爬措施（验证码、IP 限制等）
- 增加请求间隔（修改 `crawl_novel.sh` 中的 `sleep` 时间）

**Q: 如何处理动态加载的内容？**

A:
- 在 `crawl-chapter-playwright.js` 中使用 `page.waitForSelector('.content')` 等待特定元素
- 增加 `waitForTimeout` 时间
- 改用 `waitUntil: 'networkidle'` 策略

**Q: 章节顺序混乱怎么办？**

A: 运行整理脚本 `python3 organize_chapters.py`，它会按章节号重新排序

**Q: 如何爬取其他小说网站？**

A:
1. 修改章节提取的 grep 模式（如 `link.*第.*章`）
2. 修改 `crawl-chapter-playwright.js` 中的 CSS 选择器
3. 调整内容清理规则（广告文本）
4. 修改 `crawl_novel.sh` 中的 URL 拼接逻辑

**Q: 如何查看爬取进度？**

A:
- 使用 `tmux attach -t novel_crawler` 查看实时日志
- 使用 `find ~/novel_data -name "*.md" | wc -l` 统计已爬取章节数
- 按 `Ctrl+B` 然后 `D` 分离 tmux 会话

**Q: 爬取中断了怎么办？**

A:
- 脚本会跳过已存在的章节文件
- 重新运行 `bash crawl_novel.sh` 即可继续爬取
- 或手动编辑 `/tmp/all_chapters.txt` 删除已爬取的链接

## 自定义配置

### 修改输出目录

编辑 `crawl_novel.sh`：
```bash
OUTPUT_DIR="/your/custom/path"
```

### 修改选择器

编辑 `crawl-chapter-playwright.js` 中的选择器数组：
```javascript
const titleSelectors = ['.your-title-selector', ...];
const contentSelectors = ['.your-content-selector', ...];
```

### 修改清理规则

编辑 `crawl-chapter-playwright.js` 中的清理逻辑：
```javascript
let cleanContent = data.content
    .replace(/your-ad-text/g, '')
    .replace(/another-pattern/g, '')
    .trim();
```
