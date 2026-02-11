# Crawlee Scripts for nanobot

这个目录包含用于网页爬取的Crawlee脚本，可以通过nanobot的`exec`工具调用。

## 安装

```bash
cd scripts/crawlee
npm install
```

## 脚本列表

### 1. crawl-url.js - 爬取单个URL

爬取单个网页并提取详细信息。

**用法:**
```bash
node crawl-url.js <url> [output_file]
```

**示例:**
```bash
node crawl-url.js https://example.com
node crawl-url.js https://github.com output.json
```

**输出:**
- 页面标题
- 完整文本内容（前5000字符）
- 所有链接（前50个）
- 所有图片（前20张）
- Meta信息（description, keywords）

---

### 2. crawl-urls.js - 批量爬取URL列表

批量爬取多个URL。

**用法:**
```bash
node crawl-urls.js <url1> <url2> ... [--output output.json]
```

**示例:**
```bash
node crawl-urls.js https://example.com https://github.com
node crawl-urls.js https://site1.com https://site2.com --output results.json
```

**输出:**
- 每个URL的基本信息
- 链接和图片数量统计
- 成功/失败统计

---

### 3. crawl-site.js - 爬取整个网站

递归爬取网站的多个页面（同域名）。

**用法:**
```bash
node crawl-site.js <url> [--max-pages 10] [--output output.json]
```

**示例:**
```bash
node crawl-site.js https://example.com
node crawl-site.js https://docs.example.com --max-pages 50 --output docs.json
```

**输出:**
- 网站结构信息
- 每个页面的标题和标题层级
- 链接统计

---

## 在nanobot中使用

### 方法1: 直接通过exec工具

```python
# 爬取单个URL
exec(command="node scripts/crawlee/crawl-url.js https://example.com output.json")

# 批量爬取
exec(command="node scripts/crawlee/crawl-urls.js https://site1.com https://site2.com")

# 爬取网站
exec(command="node scripts/crawlee/crawl-site.js https://example.com --max-pages 20")
```

### 方法2: 通过nanobot agent

```bash
nanobot agent -m "使用crawlee爬取 https://example.com 并保存结果"
```

LLM会自动调用相应的脚本。

---

## 输出格式

所有脚本都输出JSON格式的结果文件，包含：

```json
{
  "title": "页面标题",
  "url": "https://example.com",
  "text": "页面文本内容...",
  "links": [
    {"text": "链接文本", "href": "https://..."}
  ],
  "images": [
    {"src": "https://...", "alt": "图片描述"}
  ],
  "meta": {
    "description": "页面描述",
    "keywords": "关键词"
  }
}
```

---

## 注意事项

1. **速率限制**: Crawlee会自动处理速率限制，避免被封禁
2. **内存使用**: 爬取大量页面时注意内存使用
3. **超时设置**: 默认超时60秒，可以通过环境变量调整
4. **代理支持**: 可以通过Crawlee配置使用代理

---

## 高级配置

如需更复杂的爬取逻辑，可以修改脚本或创建新脚本。参考：
- [Crawlee文档](https://crawlee.dev/)
- [Playwright文档](https://playwright.dev/)

---

## 故障排除

**问题: "playwright not found"**
```bash
npx playwright install
```

**问题: 爬取失败**
- 检查网络连接
- 检查目标网站是否可访问
- 增加超时时间
- 使用代理

**问题: 内存不足**
- 减少 --max-pages 参数
- 分批爬取
