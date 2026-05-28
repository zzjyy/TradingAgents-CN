#!/usr/bin/env python3
"""
基金/ETF 分析 CLI 入口（my_fund_support 分支新增）

用法:
    # 仅取数据（无需 LLM Key）
    python scripts/run_fund_analysis.py --symbol 510300 --data-only

    # 完整分析（需要 DeepSeek API Key）
    python scripts/run_fund_analysis.py --symbol 005827

    # 自定义日期
    python scripts/run_fund_analysis.py --symbol 510300 --date 2026-05-28

数据来源: AKShare（免费，无需 API Key）
LLM: 默认 DeepSeek（环境变量 DEEPSEEK_API_KEY）
"""

import argparse
import os
import sys
from datetime import date, datetime

# 把项目根加入 sys.path，便于直接运行
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="公募基金/ETF 多智能体分析 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_fund_analysis.py --symbol 510300 --data-only
  python scripts/run_fund_analysis.py --symbol 005827 --recent-days 90
""",
    )
    parser.add_argument("--symbol", required=True, help="基金代码 (6位数字)，如 510300、005827")
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="分析日期 yyyy-mm-dd（默认今天）")
    parser.add_argument("--recent-days", type=int, default=60,
                        help="净值历史回看天数（默认 60）")
    parser.add_argument("--data-only", action="store_true",
                        help="只拉数据并打印，跳过 LLM 分析（无需 API Key）")
    parser.add_argument("--llm-provider", default="deepseek",
                        help="LLM 提供商（默认 deepseek）")
    parser.add_argument("--model", default="deepseek-chat",
                        help="LLM 模型名（默认 deepseek-chat）")
    return parser.parse_args()


def _load_akshare_fund_module():
    """直接加载 akshare_fund.py，避开 tradingagents.dataflows 包的重依赖链。

    数据层只需 akshare/pandas/toml，无需 openai/yfinance/chromadb 等。
    """
    import importlib.util
    path = os.path.join(PROJ_ROOT, "tradingagents", "dataflows",
                        "providers", "china", "akshare_fund.py")
    spec = importlib.util.spec_from_file_location("akshare_fund_isolated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_fund_code_local(symbol: str) -> bool:
    """本地版基金代码识别（与 stock_validator._is_fund_code 同步）。"""
    if not symbol or len(symbol) != 6 or not symbol.isdigit():
        return False
    return symbol[:3] in {
        "510", "511", "512", "513", "515", "516", "517", "518",
        "159", "501", "502", "588", "560", "561", "562", "563",
        "519", "161", "162", "163", "164", "165", "166", "167", "168", "169",
    }


def run_data_only(symbol: str, recent_days: int) -> int:
    """不调用 LLM，只验证数据层是否能跑通。"""
    print(f"\n{'='*60}")
    print(f"基金/ETF 数据快照: {symbol}")
    print(f"{'='*60}\n")

    is_fund = _is_fund_code_local(symbol)
    print(f"[识别] 代码 {symbol} → {'基金/ETF' if is_fund else '非基金（按 A股 走）'}")
    if not is_fund:
        print("⚠️  该代码不在已知基金前缀范围内（510/159/501/588/519/161-169 等）")
        print("   如果确实是基金，仍可继续，但需要手动确认。")

    mod = _load_akshare_fund_module()
    p = mod.get_akshare_fund_provider()
    if not p.connected:
        print("❌ AKShare 未就绪。请确认已 pip install akshare")
        return 1

    is_etf = symbol[:3] in (
        "510", "511", "512", "513", "515", "516", "517", "518",
        "159", "501", "502", "588", "560", "561", "562", "563",
    )

    # 1. 档案
    ok, df, err = p.get_fund_profile(symbol)
    if ok:
        print("\n[1] 基础档案:")
        for row in df.to_dict("records"):
            print(f"    {row.get('item'):<12s}: {row.get('value')}")
    else:
        print(f"\n[1] 档案接口失败: {err}")

    # 2. 名称索引
    ok, info, err = p.lookup_fund_name(symbol)
    if ok:
        print(f"\n[2] 名称索引: {info.get('基金简称')} | 类型: {info.get('基金类型')}")

    # 3. ETF 实时（仅 ETF）
    if is_etf:
        ok, row, err = p.get_etf_spot(symbol)
        if ok:
            print("\n[3] ETF 实时快照:")
            for k in ("最新价", "IOPV实时估值", "基金折价率", "涨跌幅", "成交额"):
                if k in row:
                    print(f"    {k:<12s}: {row[k]}")

    # 4. 净值走势
    ok, df, err = p.get_open_fund_nav(symbol)
    if ok:
        tail = df.tail(recent_days)
        first_nav = float(tail.iloc[0]["单位净值"])
        last_nav = float(tail.iloc[-1]["单位净值"])
        chg = (last_nav / first_nav - 1) * 100 if first_nav else 0
        print(f"\n[4] 单位净值（最近 {len(tail)} 期）:")
        print(f"    期初: {first_nav:.4f}（{tail.iloc[0]['净值日期']}）")
        print(f"    最新: {last_nav:.4f}（{tail.iloc[-1]['净值日期']}）")
        print(f"    区间涨跌: {chg:+.2f}%")
    else:
        print(f"\n[4] 净值接口失败: {err}")

    # 5. 持仓
    year = str(datetime.now().year)
    ok, df, err = p.get_fund_holdings(symbol, year)
    if not ok:
        ok, df, err = p.get_fund_holdings(symbol, str(int(year) - 1))
        year = str(int(year) - 1)
    if ok and not df.empty:
        latest_q = df["季度"].iloc[0] if "季度" in df.columns else year
        top = df[df["季度"] == latest_q].head(10) if "季度" in df.columns else df.head(10)
        print(f"\n[5] 最新重仓（{latest_q}） Top 10:")
        for row in top.to_dict("records"):
            print(f"    {row.get('股票代码','-'):<8s} {row.get('股票名称','-'):<12s} "
                  f"{row.get('占净值比例','-')}%")
    else:
        print(f"\n[5] 持仓接口失败: {err}")

    print(f"\n{'='*60}")
    print("✅ 数据层验证通过（--data-only 模式不调用 LLM）")
    print(f"{'='*60}\n")
    return 0


def run_full(symbol: str, analysis_date: str, llm_provider: str, model: str) -> int:
    """完整 LangGraph 多智能体分析。需要 LLM API Key。"""
    print(f"\n{'='*60}")
    print(f"基金/ETF 多智能体分析: {symbol} @ {analysis_date}")
    print(f"LLM: {llm_provider} / {model}")
    print(f"{'='*60}\n")

    if llm_provider == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ DEEPSEEK_API_KEY 未设置。请 export DEEPSEEK_API_KEY=... 或使用 --data-only")
        return 2

    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = llm_provider
    config["deep_think_llm"] = model
    config["quick_think_llm"] = model
    config["online_tools"] = True
    config["max_debate_rounds"] = 1
    # my_fund_support: 标记资产类型，供下游路由（B 阶段 setup.py 会用）
    config["asset_type"] = "fund"

    ta = TradingAgentsGraph(debug=True, config=config)
    _, decision = ta.propagate(symbol, analysis_date)

    print("\n" + "=" * 60)
    print("最终决策:")
    print("=" * 60)
    print(decision)
    return 0


def main() -> int:
    args = parse_args()

    if args.data_only:
        return run_data_only(args.symbol, args.recent_days)
    return run_full(args.symbol, args.date, args.llm_provider, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
