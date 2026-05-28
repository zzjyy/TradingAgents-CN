"""
基金市场分析师（my_fund_support 分支新增）

替代 market_analyst 用于 asset_type=fund 的场景。
- ETF (510/159/588 等)：分析 K 线、IOPV/折溢价、成交量
- 开放基金 (519/161-169)：分析单位净值走势、累计净值、区间收益

复用 market_report 状态字段，以便下游 (bull/bear, risk debate, trader)
无需感知资产类型，统一从 market_report 读取。
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.utils.logging_init import get_logger
from tradingagents.agents.utils.google_tool_handler import GoogleToolCallHandler

logger = get_logger("default")


def _is_etf(symbol: str) -> bool:
    return bool(symbol and len(symbol) >= 3 and symbol[:3] in (
        "510", "511", "512", "513", "515", "516", "517", "518",
        "159", "501", "502", "588", "560", "561", "562", "563",
    ))


def _get_fund_display_name(symbol: str) -> str:
    """通过 AKShareFundProvider 拿一个尽量友好的展示名。"""
    try:
        from tradingagents.dataflows.providers.china import get_akshare_fund_provider
        p = get_akshare_fund_provider()
        if not p.connected:
            return f"基金{symbol}"
        ok, info, _ = p.lookup_fund_name(symbol)
        if ok and info.get("基金简称"):
            return info["基金简称"]
    except Exception as e:
        logger.debug(f"[基金市场分析师] 名称查询失败: {e}")
    return f"基金{symbol}"


def create_fund_market_analyst(llm, toolkit):
    """创建基金市场分析师节点。签名与 create_market_analyst 一致，便于 setup.py 路由替换。"""

    def fund_market_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        is_etf = _is_etf(ticker)
        asset_kind = "场内 ETF" if is_etf else "场外开放式基金"
        display_name = _get_fund_display_name(ticker)

        logger.info(f"📊 [基金市场分析师] 开始: {ticker} ({display_name}, {asset_kind})")

        # 根据 ETF / 开放基金选择不同工具组合
        if is_etf:
            tools = [
                toolkit.get_etf_realtime_unified,
                toolkit.get_etf_market_data_unified,
                toolkit.get_fund_basic_info_unified,
            ]
            tool_workflow = (
                "1. 先用 get_etf_realtime_unified({ticker}) 看实时价格、IOPV、折溢价\n"
                "2. 再用 get_etf_market_data_unified({ticker}, 近 60 个交易日) 看 K 线\n"
                "3. 用 get_fund_basic_info_unified({ticker}) 看跟踪标的、规模、费率\n"
                "4. 一轮工具调用够了，第二轮直接出报告"
            )
            scenario_focus = (
                "ETF 关注重点：\n"
                "- IOPV vs 二级市场价格的折溢价（>1% 视为显著套利空间）\n"
                "- 成交额变化（流动性是否充足）\n"
                "- K 线趋势 + 与跟踪指数的偏离\n"
                "- 规模（<5 亿存在清盘风险）"
            )
        else:
            tools = [
                toolkit.get_fund_basic_info_unified,
                toolkit.get_fund_nav_unified,
            ]
            tool_workflow = (
                "1. 先用 get_fund_basic_info_unified({ticker}) 看基础档案\n"
                "2. 再用 get_fund_nav_unified({ticker}, recent_days=120) 看 120 期净值\n"
                "3. 一轮工具调用够了，第二轮直接出报告"
            )
            scenario_focus = (
                "开放基金关注重点：\n"
                "- 单位净值近 1M/3M/6M/1Y 区间涨跌\n"
                "- 最大回撤与波动率（凭日增长率估算）\n"
                "- 规模变化趋势（暴增/暴减都是信号）\n"
                "- 基金类型与业绩比较基准的匹配度"
            )

        system_message = (
            "你是一位专业的公募基金/ETF 市场分析师，与其他分析师协作。\n"
            "\n"
            "📋 **分析对象：**\n"
            f"- 基金简称: {display_name}\n"
            f"- 基金代码: {ticker}\n"
            f"- 资产类型: {asset_kind}\n"
            f"- 分析日期: {current_date}\n"
            "\n"
            "🔧 **工具调用流程：**\n"
            f"{tool_workflow}\n"
            "\n"
            "🎯 **分析重点：**\n"
            f"{scenario_focus}\n"
            "\n"
            "📝 **输出格式（严格遵守）：**\n"
            "\n"
            "## 📊 基金基本信息\n"
            f"- 基金简称: {display_name}\n"
            f"- 基金代码: {ticker}\n"
            f"- 资产类型: {asset_kind}\n"
            "- 跟踪标的/投资策略: [从档案中提取]\n"
            "- 最新规模: [从档案中提取]\n"
            "\n"
            "## 📈 行情/净值分析\n"
            "[ETF: K 线趋势 + IOPV 折溢价；开放基金: 净值走势 + 区间收益 + 估算回撤]\n"
            "\n"
            "## 💧 流动性与规模评估\n"
            "[场内 ETF 看成交额；场外基金看规模变化]\n"
            "\n"
            "## 💭 投资建议\n"
            f"[给出明确建议: {'买入/持有/卖出' if is_etf else '申购/持有/赎回'}, 并简述理由]\n"
            "\n"
            "⚠️ **重要提醒：**\n"
            "- 全部用中文\n"
            "- 不要在标题中使用「最终交易建议」前缀，最终决策需要综合所有分析师\n"
            f"- {'ETF 使用「买入/持有/卖出」' if is_etf else '场外基金使用「申购/持有/赎回」'}\n"
            "- 数值要给具体数字，不要泛泛而谈\n"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(ticker=ticker)

        tool_names = []
        for t in tools:
            if hasattr(t, "name"):
                tool_names.append(t.name)
            elif hasattr(t, "__name__"):
                tool_names.append(t.__name__)
            else:
                tool_names.append(str(t))
        logger.info(f"📊 [基金市场分析师] 绑定工具: {tool_names}")

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke({"messages": state["messages"]})

        if GoogleToolCallHandler.is_google_model(llm):
            analysis_prompt_template = GoogleToolCallHandler.create_analysis_prompt(
                ticker=ticker,
                company_name=display_name,
                analyst_type="基金市场分析",
                specific_requirements=(
                    "对 ETF 关注 IOPV/折溢价/流动性；"
                    "对开放基金关注净值趋势、回撤、规模变化。"
                ),
            )
            report, _ = GoogleToolCallHandler.handle_google_tool_calls(
                result=result, llm=llm, tools=tools, state=state,
                analysis_prompt_template=analysis_prompt_template,
                analyst_name="基金市场分析师",
            )
        else:
            report = ""
            if not getattr(result, "tool_calls", None):
                report = result.content

        return {
            "messages": [result],
            "market_report": report,
            "sender": "FundMarketAnalyst",
        }

    return fund_market_analyst_node
