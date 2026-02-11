#!/usr/bin/env node
/**
 * Crawlee Script: 爬取网站（递归爬取多个页面）
 *
 * 用法: node crawl-site.js <url> [--max-pages 10] [--output output.json]
 * 示例: node crawl-site.js https://example.com --max-pages 20 --output site-data.json
 */

const { PlaywrightCrawler } = require('crawlee');
const fs = require('fs');

async function main() {
    const args = process.argv.slice(2);

    // 解析参数
    let startUrl = null;
    let maxPages = 10;
    let outputFile = 'site-crawl.json';

    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--max-pages' || args[i] === '-m') {
            maxPages = parseInt(args[i + 1]);
            i++;
        } else if (args[i] === '--output' || args[i] === '-o') {
            outputFile = args[i + 1];
            i++;
        } else if (args[i].startsWith('http')) {
            startUrl = args[i];
        }
    }

    if (!startUrl) {
        console.error('错误: 请提供起始URL');
        console.error('用法: node crawl-site.js <url> [--max-pages 10] [--output output.json]');
        process.exit(1);
    }

    console.log(`开始爬取网站: ${startUrl}`);
    console.log(`最大页面数: ${maxPages}`);

    const results = [];
    const baseUrl = new URL(startUrl).origin;

    const crawler = new PlaywrightCrawler({
        maxRequestsPerCrawl: maxPages,
        async requestHandler({ request, page, enqueueLinks, log }) {
            log.info(`[${results.length + 1}/${maxPages}] ${request.url}`);

            await page.waitForLoadState('networkidle');

            // 提取页面数据
            const data = await page.evaluate(() => {
                return {
                    title: document.title,
                    url: window.location.href,
                    h1: Array.from(document.querySelectorAll('h1')).map(h => h.innerText.trim()),
                    h2: Array.from(document.querySelectorAll('h2')).map(h => h.innerText.trim()).slice(0, 10),
                    linkCount: document.querySelectorAll('a').length,
                    imageCount: document.querySelectorAll('img').length,
                };
            });

            results.push(data);

            // 只爬取同域名下的链接
            await enqueueLinks({
                globs: [`${baseUrl}/**`],
                label: 'page',
            });

            log.info(`✓ 完成，发现 ${data.linkCount} 个链接`);
        },
        failedRequestHandler({ request, log }) {
            log.error(`✗ 失败: ${request.url}`);
        },
    });

    try {
        await crawler.run([startUrl]);

        // 保存结果
        fs.writeFileSync(outputFile, JSON.stringify(results, null, 2));
        console.log(`\n✓ 爬取完成！`);
        console.log(`  爬取页面数: ${results.length}`);
        console.log(`  结果保存到: ${outputFile}`);

    } catch (error) {
        console.error('爬取错误:', error.message);
        process.exit(1);
    }
}

main();
