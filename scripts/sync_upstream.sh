#!/usr/bin/env bash
# 上游同步辅助脚本（my_fund_support 分支新增）
#
# 作用：
#   1) 从 upstream/main 拉取最新代码
#   2) 切到 my_fund_support，rebase 到最新 upstream/main
#   3) 提示冲突文件（基本都集中在已知的黄区/雷区文件）
#   4) 不自动 push —— 用户检查后再手动 push
#
# 设计原则（详见 memory project_git_workflow.md）：
#   - 永远 rebase 而非 merge，保持单线历史
#   - 冲突预期集中在 4 个文件：
#       * tradingagents/agents/utils/agent_utils.py (Toolkit append)
#       * tradingagents/utils/stock_validator.py (基金代码识别)
#       * tradingagents/tools/unified_news_tool.py (基金代码识别)
#       * tradingagents/graph/setup.py (asset_type 路由)
#       * tradingagents/graph/signal_processing.py (基金语义)
#   - 所有改动都被 `# === start/end edit by my_fund_support ===` 标记包围，
#     冲突时只需保留两端标记内的内容
#
# 用法：
#   bash scripts/sync_upstream.sh          # 标准同步
#   bash scripts/sync_upstream.sh --check  # 仅检查 upstream 是否有新提交，不动分支

set -euo pipefail

UPSTREAM_REMOTE="upstream"
UPSTREAM_BRANCH="main"
WORK_BRANCH="my_fund_support"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[sync]${NC} $*"; }
warn() { echo -e "${YELLOW}[sync]${NC} $*"; }
err()  { echo -e "${RED}[sync]${NC} $*" >&2; }

# 0. 前置检查
if ! git remote | grep -q "^${UPSTREAM_REMOTE}$"; then
    err "未找到 remote \"${UPSTREAM_REMOTE}\"。请先："
    err "    git remote add upstream https://github.com/hsliuping/TradingAgents-CN.git"
    exit 1
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "${CURRENT_BRANCH}" != "${WORK_BRANCH}" ]]; then
    warn "当前不在 ${WORK_BRANCH} 分支（当前: ${CURRENT_BRANCH}）"
    warn "脚本会在最后切回 ${WORK_BRANCH} 再 rebase"
fi

if [[ -n "$(git status --porcelain)" ]]; then
    err "工作区不干净，请先 commit 或 stash。"
    git status --short
    exit 1
fi

# 1. 拉取 upstream
log "拉取 ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}..."
git fetch "${UPSTREAM_REMOTE}" "${UPSTREAM_BRANCH}"

# 2. 看看有多少新提交
BEHIND=$(git rev-list --count "HEAD..${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" 2>/dev/null || echo 0)
if [[ "${BEHIND}" == "0" ]]; then
    log "已是最新，无需同步。"
    exit 0
fi
log "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH} 比当前分支多 ${BEHIND} 个提交"
git log --oneline "HEAD..${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" | head -20
echo

# --check 模式：只看不动
if [[ "${1:-}" == "--check" ]]; then
    log "(--check 模式) 不执行 rebase，请手动确认后重新运行不带 --check"
    exit 0
fi

# 3. 切到工作分支
if [[ "${CURRENT_BRANCH}" != "${WORK_BRANCH}" ]]; then
    log "切换到 ${WORK_BRANCH}..."
    git checkout "${WORK_BRANCH}"
fi

# 4. rebase
log "rebase 到 ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}..."
if git rebase "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"; then
    log "rebase 成功 ✅"
    log "记得推送：git push --force-with-lease origin ${WORK_BRANCH}"
else
    warn "rebase 出现冲突。"
    warn "查看冲突文件："
    git status --short | grep '^UU\|^AA\|^DD' || true
    echo
    warn "解决步骤："
    warn "  1) 编辑冲突文件，保留 # === start/end edit by my_fund_support === 标记内的内容"
    warn "  2) git add <file>"
    warn "  3) git rebase --continue"
    warn "  4) 或中止: git rebase --abort"
    exit 1
fi
