#!/usr/bin/env node
/**
 * Crawlee Script: 爬取小说章节
 *
 * 用法: node crawl-novel-chapter.js <chapter_url> [output_file] [--format=json|markdown]
 * 示例:
 *   node crawl-novel-chapter.js https://www.bqg518.cc/#/book/341/1 chapter1.json
 *   node crawl-novel-chapter.js https://www.bqg518.cc/#/book/341/1 chapter1.md --format=markdown
 */

const { PlaywrightCrawler } = require('crawlee');
const fs = require('fs');

// 解析命令行参数
function parseArgs() {
    const args = process.argv.slice(2);
    let url = null;
    let outputFile = null;
    let format = 'json'; // 默认格式

    for (const arg of args) {
        if (arg.startsWith('--format=')) {
            format = arg.split('=')[1];
        } else if (!url) {
            url = arg;
        } else if (!outputFile) {
            outputFile = arg;
        }
    }

    // 根据格式设置默认输出文件
    if (!outputFile) {
        outputFile = format === 'markdown' ? 'chapter.md' : 'chapter.json';
    }

    return { url, outputFile, format };
}

// 生成Markdown格式输出
function formatAsMarkdown(data) {
    const lines = [
        `# ${data.title}`,
        '',
        data.content,
        '',
        '---',
        '',
        '**元数据：**',
        `- 来源：${data.url}`,
        `- 字数：${data.contentLength} 字符`,
        `- 爬取时间：${new Date(data.timestamp).toLocaleString('zh-CN')}`,
        ''
    ];
    return lines.join('\n');
}

async function main() {
    const { url, outputFile, format } = parseArgs();

    if (!url) {
        console.error('错误: 请提供章节URL');
        console.error('用法: node crawl-novel-chapter.js <chapter_url> [output_file] [--format=json|markdown]');
        console.error('示例:');
        console.error('  node crawl-novel-chapter.js https://example.com/chapter1 chapter1.json');
        console.error('  node crawl-novel-chapter.js https://example.com/chapter1 chapter1.md --format=markdown');
        process.exit(1);
    }

    if (format !== 'json' && format !== 'markdown') {
        console.error('错误: format参数只支持 json 或 markdown');
        process.exit(1);
    }

    const results = [];

    const crawler = new PlaywrightCrawler({
        maxRequestsPerCrawl: 1,
        async requestHandler({ request, page, log }) {
            log.info(`正在爬取章节: ${request.url}`);

            // 等待页面加载
            await page.waitForLoadState('networkidle');

            // 等待内容加载（针对SPA应用）
            await page.waitForTimeout(2000);

            // 提取章节信息
            const data = await page.evaluate(() => {
                // 尝试多种选择器来提取章节内容
                const chapterTitle =
                    document.querySelector('.chapter-title')?.innerText ||
                    document.querySelector('h1')?.innerText ||
                    document.querySelector('.title')?.innerText ||
                    '未找到标题';

                const chapterContent =
                    document.querySelector('.chapter-content')?.innerText ||
                    document.querySelector('.content')?.innerText ||
                    document.querySelector('#content')?.innerText ||
                    document.querySelector('article')?.innerText ||
                    document.body.innerText;

                return {
                    title: chapterTitle.trim(),
                    url: window.location.href,
                    content: chapterContent.trim(),
                    contentLength: chapterContent.length,
                    timestamp: new Date().toISOString()
                };
            });

            results.push(data);
            log.info(`✓ 章节爬取完成: ${data.title}`);
            log.info(`  内容长度: ${data.contentLength} 字符`);
        },
        failedRequestHandler({ request, log }) {
            log.error(`✗ 爬取失败: ${request.url}`);
        },
    });

    try {
        await crawler.run([url]);

        // 根据格式保存结果
        let output;
        if (format === 'markdown') {
            output = formatAsMarkdown(results[0]);
        } else {
            output = JSON.stringify(results[0], null, 2);
        }

        fs.writeFileSync(outputFile, output);
        console.log(`\n✓ 结果已保存到: ${outputFile}`);
        console.log(`✓ 输出格式: ${format}`);
        console.log(`\n章节信息:`);
        console.log(`  标题: ${results[0].title}`);
        console.log(`  内容长度: ${results[0].contentLength} 字符`);

    } catch (error) {
        console.error('爬取错误:', error.message);
        process.exit(1);
    }
}

main();
