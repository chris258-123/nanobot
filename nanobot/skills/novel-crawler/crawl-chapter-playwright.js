#!/usr/bin/env node
/**
 * Playwright 章节爬虫
 * 用法: node crawl-chapter-playwright.js <url> <output_file>
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function crawlChapter(url, outputFile) {
    const browser = await chromium.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    try {
        const page = await browser.newPage();
        page.setDefaultTimeout(60000);

        // 设置 User-Agent 避免被识别为爬虫
        await page.setExtraHTTPHeaders({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        });

        // 访问页面（使用 domcontentloaded 策略加快速度）
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
        await page.waitForTimeout(2000);

        // 提取章节标题和内容
        const data = await page.evaluate(() => {
            // 提取标题
            let title = '';
            const titleSelectors = [
                '.bookname h1',
                '.content h1',
                '#chaptername',
                '.chapter-title',
                'h1',
                '.title'
            ];

            for (const selector of titleSelectors) {
                const el = document.querySelector(selector);
                if (el && el.textContent.trim()) {
                    const text = el.textContent.trim();
                    // 跳过网站名称
                    if (text !== '笔趣阁' && text !== '22biqu' && !text.includes('www.')) {
                        title = text;
                        break;
                    }
                }
            }

            // 如果还没找到标题，尝试从内容中提取第一行
            if (!title) {
                const contentEl = document.querySelector('#content, .content');
                if (contentEl) {
                    const firstLine = contentEl.textContent.trim().split('\n')[0];
                    if (firstLine && firstLine.includes('第') && firstLine.includes('章')) {
                        title = firstLine.trim();
                    }
                }
            }

            // 提取内容
            let content = '';
            const contentSelectors = [
                '#content',
                '.content',
                '#chaptercontent',
                '.chapter-content',
                '.text-content',
                'article'
            ];

            for (const selector of contentSelectors) {
                const el = document.querySelector(selector);
                if (el) {
                    // 移除脚本和样式标签
                    const scripts = el.querySelectorAll('script, style');
                    scripts.forEach(s => s.remove());

                    const text = el.innerText || el.textContent;
                    if (text && text.trim().length > 100) {
                        content = text.trim();
                        break;
                    }
                }
            }

            return { title, content };
        });

        // 验证提取的数据
        if (!data.title || data.title === '未找到标题') {
            data.title = '未找到标题';
        }

        if (!data.content || data.content.length < 50) {
            throw new Error('内容提取失败或内容过短');
        }

        // 清理内容中的广告和乱码
        let cleanContent = data.content
            // 清理常见广告标识
            .replace(/ghxs9☆cc/g, '')
            .replace(/blbiji♜cc/g, '')
            .replace(/22biqu\.com/g, '')
            .replace(/笔趣阁/g, '')
            .replace(/www\.\w+\.com/g, '')
            .replace(/请收藏本站：.*/g, '')
            .replace(/最快更新.*/g, '')
            .replace(/无弹窗.*/g, '')
            // 清理中文之间的英文字母（乱码）
            // 匹配模式：中文字符 + 英文字母 + 中文字符
            .replace(/([\u4e00-\u9fa5])[a-zA-Z]+([\u4e00-\u9fa5])/g, '$1$2')
            // 清理中文之间的单个字母
            .replace(/([\u4e00-\u9fa5])[a-zA-Z]([\u4e00-\u9fa5])/g, '$1$2')
            .trim();

        // 生成 Markdown 格式
        const markdown = `# ${data.title}

${cleanContent}

---

**元数据：**
- 来源：${url}
- 字数：${cleanContent.length} 字符
- 爬取时间：${new Date().toLocaleString('zh-CN')}
`;

        // 确保输出目录存在
        const dir = path.dirname(outputFile);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }

        // 写入文件
        fs.writeFileSync(outputFile, markdown, 'utf-8');

        console.log(`✓ 成功爬取: ${data.title}`);
        console.log(`  字数: ${cleanContent.length}`);
        console.log(`  保存: ${outputFile}`);

        return { success: true, title: data.title };

    } catch (error) {
        console.error(`✗ 爬取失败: ${error.message}`);
        return { success: false, error: error.message };
    } finally {
        await browser.close();
    }
}

async function main() {
    const args = process.argv.slice(2);

    if (args.length < 2) {
        console.error('用法: node crawl-chapter-playwright.js <url> <output_file>');
        process.exit(1);
    }

    const [url, outputFile] = args;
    const result = await crawlChapter(url, outputFile);

    if (!result.success) {
        process.exit(1);
    }
}

main();
