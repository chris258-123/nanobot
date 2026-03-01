#!/usr/bin/env node
/**
 * Playwright 章节爬虫（支持同章分页拼接）
 * 用法: node crawl-chapter-playwright.js <url> <output_file>
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const PAGE_TIMEOUT_MS = 60000;
const MAX_PAGE_PARTS = 20;
const MIN_CONTENT_LENGTH = 50;

function normalizeText(value) {
    return String(value || '').replace(/\s+/g, '').trim();
}

function extractChapterKey(rawUrl) {
    try {
        const parsed = new URL(rawUrl);
        const match = parsed.pathname.match(/\/(\d+)(?:_(\d+))?\.html$/);
        if (!match) {
            return null;
        }
        return {
            chapterId: match[1],
            pageNo: match[2] ? parseInt(match[2], 10) : 1,
        };
    } catch {
        return null;
    }
}

function findNextPageUrl(currentUrl, anchors) {
    const currentKey = extractChapterKey(currentUrl);
    if (!currentKey) {
        return null;
    }

    let best = null;

    for (const anchor of anchors || []) {
        const href = String(anchor.href || '').trim();
        if (!href) {
            continue;
        }

        let resolved;
        try {
            resolved = new URL(href, currentUrl).href;
        } catch {
            continue;
        }

        if (resolved === currentUrl) {
            continue;
        }

        const key = extractChapterKey(resolved);
        if (!key || key.chapterId !== currentKey.chapterId) {
            continue;
        }
        if (key.pageNo <= currentKey.pageNo) {
            continue;
        }

        const text = String(anchor.text || '').trim();
        if (text.includes('下一章')) {
            continue;
        }

        if (!best || key.pageNo < best.pageNo) {
            best = { url: resolved, pageNo: key.pageNo };
        }
    }

    return best ? best.url : null;
}

function cleanPageContent(rawContent, title) {
    let cleanContent = String(rawContent || '')
        .replace(/ghxs9☆cc/g, '')
        .replace(/blbiji♜cc/g, '')
        .replace(/22biqu\.com/g, '')
        .replace(/笔趣阁/g, '')
        .replace(/www\.\w+\.com/g, '')
        .replace(/请收藏本站：.*/g, '')
        .replace(/最快更新.*/g, '')
        .replace(/无弹窗.*/g, '')
        .replace(/([\u4e00-\u9fa5])[a-zA-Z]+([\u4e00-\u9fa5])/g, '$1$2')
        .replace(/([\u4e00-\u9fa5])[a-zA-Z]([\u4e00-\u9fa5])/g, '$1$2')
        .trim();

    const lines = cleanContent
        .split(/\n+/)
        .map((line) => line.trim())
        .filter(Boolean);

    if (lines.length > 0 && normalizeText(lines[0]) === normalizeText(title)) {
        lines.shift();
    }

    return lines.join('\n\n').trim();
}

function mergeContentParts(parts) {
    if (!parts.length) {
        return '';
    }

    let merged = parts[0];

    for (let i = 1; i < parts.length; i += 1) {
        const next = parts[i];
        if (!next) {
            continue;
        }
        if (normalizeText(next) === normalizeText(merged)) {
            continue;
        }

        const prevLines = merged.split('\n').map((line) => line.trim()).filter(Boolean);
        const nextLines = next.split('\n').map((line) => line.trim()).filter(Boolean);

        let overlap = 0;
        const maxOverlap = Math.min(20, prevLines.length, nextLines.length);
        for (let size = maxOverlap; size >= 1; size -= 1) {
            const left = normalizeText(prevLines.slice(-size).join(''));
            const right = normalizeText(nextLines.slice(0, size).join(''));
            if (left && left === right) {
                overlap = size;
                break;
            }
        }

        const dedupedNext = overlap > 0 ? nextLines.slice(overlap).join('\n\n') : next;
        if (!dedupedNext.trim()) {
            continue;
        }

        merged = `${merged}\n\n${dedupedNext.trim()}`.trim();
    }

    return merged;
}

async function crawlChapter(url, outputFile) {
    const browser = await chromium.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });

    try {
        const page = await browser.newPage();
        page.setDefaultTimeout(PAGE_TIMEOUT_MS);

        await page.setExtraHTTPHeaders({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        });

        let currentUrl = url;
        const visited = new Set();
        const contentParts = [];
        const sourceUrls = [];
        let chapterTitle = '';

        while (currentUrl && contentParts.length < MAX_PAGE_PARTS) {
            if (visited.has(currentUrl)) {
                break;
            }
            visited.add(currentUrl);
            sourceUrls.push(currentUrl);

            await page.goto(currentUrl, { waitUntil: 'domcontentloaded', timeout: PAGE_TIMEOUT_MS });
            await page.waitForTimeout(1500);

            const data = await page.evaluate(() => {
                let title = '';
                const titleSelectors = ['.bookname h1', '.content h1', '#chaptername', '.chapter-title', 'h1', '.title'];

                for (const selector of titleSelectors) {
                    const el = document.querySelector(selector);
                    if (el && el.textContent.trim()) {
                        const text = el.textContent.trim();
                        if (text !== '笔趣阁' && text !== '22biqu' && !text.includes('www.')) {
                            title = text;
                            break;
                        }
                    }
                }

                if (!title) {
                    const contentEl = document.querySelector('#content, .content');
                    if (contentEl) {
                        const firstLine = contentEl.textContent.trim().split('\n')[0];
                        if (firstLine && firstLine.includes('第') && firstLine.includes('章')) {
                            title = firstLine.trim();
                        }
                    }
                }

                let content = '';
                const contentSelectors = ['#content', '.content', '#chaptercontent', '.chapter-content', '.text-content', 'article'];

                for (const selector of contentSelectors) {
                    const el = document.querySelector(selector);
                    if (!el) {
                        continue;
                    }
                    const scripts = el.querySelectorAll('script, style');
                    scripts.forEach((node) => node.remove());

                    const text = el.innerText || el.textContent;
                    if (text && text.trim().length > 60) {
                        content = text.trim();
                        break;
                    }
                }

                const navSelectors = ['.bottem2', '.bottem1', '.page_chapter', '.page1', '.page2', '.pager', '.bookname'];
                let anchorNodes = [];

                for (const selector of navSelectors) {
                    const root = document.querySelector(selector);
                    if (root) {
                        anchorNodes = anchorNodes.concat(Array.from(root.querySelectorAll('a[href]')));
                    }
                }

                if (!anchorNodes.length) {
                    anchorNodes = Array.from(document.querySelectorAll('a[href]'));
                }

                const anchors = anchorNodes.map((a) => ({
                    text: (a.textContent || '').trim(),
                    href: (a.getAttribute('href') || '').trim(),
                }));

                return { title, content, anchors };
            });

            if (!chapterTitle) {
                chapterTitle = data.title || '未找到标题';
            }

            const cleanedPart = cleanPageContent(data.content, chapterTitle);
            if (contentParts.length === 0 && cleanedPart.length < MIN_CONTENT_LENGTH) {
                throw new Error('内容提取失败或内容过短');
            }
            if (cleanedPart) {
                contentParts.push(cleanedPart);
            }

            const nextPageUrl = findNextPageUrl(currentUrl, data.anchors || []);
            if (!nextPageUrl) {
                break;
            }
            currentUrl = nextPageUrl;
        }

        if (!chapterTitle) {
            chapterTitle = '未找到标题';
        }

        const mergedContent = mergeContentParts(contentParts);
        if (!mergedContent || mergedContent.length < MIN_CONTENT_LENGTH) {
            throw new Error('内容提取失败或内容过短');
        }

        const sourceLines = sourceUrls.map((item) => `  - ${item}`).join('\n');
        const markdown = `# ${chapterTitle}\n\n${mergedContent}\n\n---\n\n**元数据：**\n- 来源：${url}\n- 分页数：${sourceUrls.length}\n- 分页链接：\n${sourceLines}\n- 字数：${mergedContent.length} 字符\n- 爬取时间：${new Date().toLocaleString('zh-CN')}\n`;

        const dir = path.dirname(outputFile);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }

        fs.writeFileSync(outputFile, markdown, 'utf-8');

        console.log(`✓ 成功爬取: ${chapterTitle}`);
        console.log(`  分页: ${sourceUrls.length} 页`);
        console.log(`  字数: ${mergedContent.length}`);
        console.log(`  保存: ${outputFile}`);

        return { success: true, title: chapterTitle, pages: sourceUrls.length };
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
