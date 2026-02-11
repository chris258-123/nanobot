#!/bin/bash

# Nanobot 代理启动脚本
# 使用 Clash Verge 代理访问 API

# 设置代理环境变量
export HTTP_PROXY="http://127.0.0.1:7897"
export HTTPS_PROXY="http://127.0.0.1:7897"
export ALL_PROXY="http://127.0.0.1:7897"

# 可选：不代理本地地址
export NO_PROXY="localhost,127.0.0.1,::1"

echo "🌐 使用代理: http://127.0.0.1:7897"
echo "📡 启动 nanobot..."
echo ""

# 运行 nanobot，传递所有参数
nanobot "$@"
