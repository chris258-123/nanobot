#!/usr/bin/env node
/**
 * Crawlee Script: 批量爬取URL列表
 *
 * 用法: node crawl-urls.js <url1> <url2> ... [--output output.json]
 * 示例: node crawl-urls.js https://example.com https://github.com --output results.json
 */

const { PlaywrightCrawler } = require('crawlee');
const fs = require('fs');

async function main() {
    const args = process.argv.slice(2);

    // 解析参数
    let urls = [];
    let outputFile = 'crawl-results.json';

    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--output' || args[i] === '-o') {
            outputFile = args[i + 1];
            i++;
        } else if (args[i].startsWith('http')) {
            urls.push(args[i]);
        }
    }

    if (urls.length === 0) {
        console.error('错误: 请提供至少一个URL');
        console.error('用法: node crawl-urls.js <url1> <url2> ... [--output output.json]');
        process.exit(1);
    }

    console.log(`准备爬取 ${urls.length} 个URL...`);
    const results = [];

    const crawler = new PlaywrightCrawler({
        maxRequestsPerCrawl: urls.length,
        async requestHandler({ request, page, log }) {
            log.info(`[${results.length + 1}/${urls.length}] 正在爬取: ${request.url}`);

            await page.waitForLoadState('networkidle');

            const data = await page.evaluate(() => {
                return {
                    title: document.title,
                    url: window.location.href,
                    text: document.body.innerText.substring(0, 3000),
                    linkCount: document.querySelectorAll('a').length,
                    imageCount: document.querySelectorAll('img').length,
                    meta: {
                        description: document.querySelector('meta[name="description"]')?.content || '',
                    }
                };
            });

            results.push(data);
            log.info(`✓ 完成: ${request.url}`);
        },
        failedRequestHandler({ request, log }) {
            log.error(`✗ 失败: ${request.url}`);
            results.push({
                url: request.url,
                error: '爬取失败'
            });
        },
    });

    try {
        await crawler.run(urls);

        // 保存结果
        fs.writeFileSync(outputFile, JSON.stringify(results, null, 2));
        console.log(`\n✓ 爬取完成！结果已保存到: ${outputFile}`);
        console.log(`\n统计:`);
        console.log(`  成功: ${results.filter(r => !r.error).length}`);
        console.log(`  失败: ${results.filter(r => r.error).length}`);

    } catch (error) {
        console.error('爬取错误:', error.message);
        process.exit(1);
    }
}

main();
