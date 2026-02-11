#!/usr/bin/env node
/**
 * Crawlee Script: 爬取单个URL
 *
 * 用法: node crawl-url.js <url> [output_file]
 * 示例: node crawl-url.js https://example.com output.json
 */

const { PlaywrightCrawler } = require('crawlee');
const fs = require('fs');

async function main() {
    const url = process.argv[2];
    const outputFile = process.argv[3] || 'crawl-result.json';

    if (!url) {
        console.error('错误: 请提供URL');
        console.error('用法: node crawl-url.js <url> [output_file]');
        process.exit(1);
    }

    const results = [];

    const crawler = new PlaywrightCrawler({
        maxRequestsPerCrawl: 1,
        async requestHandler({ request, page, enqueueLinks, log }) {
            log.info(`正在爬取: ${request.url}`);

            // 等待页面加载
            await page.waitForLoadState('networkidle');

            // 提取页面信息
            const data = await page.evaluate(() => {
                return {
                    title: document.title,
                    url: window.location.href,
                    text: document.body.innerText.substring(0, 5000), // 前5000字符
                    links: Array.from(document.querySelectorAll('a')).map(a => ({
                        text: a.innerText.trim(),
                        href: a.href
                    })).filter(link => link.href && link.text).slice(0, 50), // 前50个链接
                    images: Array.from(document.querySelectorAll('img')).map(img => ({
                        src: img.src,
                        alt: img.alt
                    })).slice(0, 20), // 前20张图片
                    meta: {
                        description: document.querySelector('meta[name="description"]')?.content || '',
                        keywords: document.querySelector('meta[name="keywords"]')?.content || '',
                    }
                };
            });

            results.push(data);
            log.info(`✓ 爬取完成: ${request.url}`);
        },
        failedRequestHandler({ request, log }) {
            log.error(`✗ 爬取失败: ${request.url}`);
        },
    });

    try {
        await crawler.run([url]);

        // 保存结果
        fs.writeFileSync(outputFile, JSON.stringify(results[0], null, 2));
        console.log(`\n✓ 结果已保存到: ${outputFile}`);
        console.log(`\n摘要:`);
        console.log(`  标题: ${results[0].title}`);
        console.log(`  链接数: ${results[0].links.length}`);
        console.log(`  图片数: ${results[0].images.length}`);

    } catch (error) {
        console.error('爬取错误:', error.message);
        process.exit(1);
    }
}

main();
