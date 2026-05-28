"""
基金持仓穿透分析师（my_fund_support 分支新增）

替代 fundamentals_analyst 用于 asset_type=fund 的场景。

基金的"基本面"与个股不同：
- 个股看 PE/PB/ROE
- 基金看「持有什么、行业集中度、风格漂移、与基准的偏离」

输出写入 fundamentals_report 字段，供下游 (bull/bear, risk debate) 消费。
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.utils.logging_init import get_logger
from tradingagents.agents.utils.google_tool_handler import GoogleToolCallHandler

logger = get_logger("default")


def create_fund_holdings_analyst(llm, toolkit):
    """创建基金持仓穿透分析师。签名与 create_fundamentals_analyst 一致。"""

    def fund_holdings_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]

        logger.info(f"🧬 [基金持仓分析师] 开始: {ticker} @ {current_date}")

        tools = [
            toolkit.get_fund_holdings_unified,
            toolkit.get_fund_basic_info_unified,
        ]

        system_message = (
            "你是一位专业的基金持仓穿透分析师，与其他分析师协作。\n"
            "\n"
            "📋 **分析对象：**\n"
            f"- 基金代码: {ticker}\n"
            f"- 分析日期: {current_date}\n"
            "\n"
            "🔧 **工具调用流程（一次完成，不要重复）：**\n"
            f"1. get_fund_holdings_unified({ticker}) 获取最新季度披露的 Top 10 重仓\n"
            f"2. get_fund_basic_info_unified({ticker}) 拿投资策略 / 基准描述\n"
            "3. 拿到数据后立即出报告\n"
            "\n"
            "🎯 **分析维度：**\n"
            "- **持仓集中度**: Top10 占比，是否过度集中（>60% 视为高集中）\n"
            "- **行业/风格识别**: 重仓股属于什么行业，是否成长/价值/红利风格\n"
            "- **港股/A 股配比**: 持仓中是否有 5 位数港股代码（如 00700、09988）\n"
            "- **与基准的匹配度**: 持仓是否真的执行了档案里写的策略\n"
            "- **风险信号**: 单只股票 >10% 是否触及双十限制、个股暴雷的连带影响\n"
            "\n"
            "📝 **输出格式（严格遵守）：**\n"
            "\n"
            "## 🧬 持仓穿透概览\n"
            "- 数据期：[最新披露季度]\n"
            "- Top10 占净值合计：X%\n"
            "- 港股占比：X%（如有）\n"
            "\n"
            "## 📊 Top 10 重仓\n"
            "| 排名 | 代码 | 名称 | 占净值 | 简评 |\n"
            "| - | - | - | - | - |\n"
            "[填表，每行一只股票，简评 ≤20 字]\n"
            "\n"
            "## 🎨 风格与行业分布\n"
            "[识别主要行业、风格倾向、与基准的偏离]\n"
            "\n"
            "## ⚠️ 风险信号\n"
            "[列出集中度过高/单股权重过大/风格漂移等具体问题，若无则写「未见显著风险信号」]\n"
            "\n"
            "## 💭 投资视角结论\n"
            "[基于持仓判断该基金当前的 beta 与暴露偏向，给出对申购/赎回的支持依据]\n"
            "\n"
            "⚠️ 注意：\n"
            "- 全部使用中文\n"
            "- 数据来自最新一期季报/年报，可能滞后 1-2 个月，需在报告中说明\n"
            "- 不要凭空捏造未在工具结果中出现的持仓\n"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(ticker=ticker)

        tool_names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools]
        logger.info(f"🧬 [基金持仓分析师] 绑定工具: {tool_names}")

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke({"messages": state["messages"]})

        if GoogleToolCallHandler.is_google_model(llm):
            analysis_prompt_template = GoogleToolCallHandler.create_analysis_prompt(
                ticker=ticker,
                company_name=f"基金{ticker}",
                analyst_type="基金持仓穿透分析",
                specific_requirements="关注 Top10 持仓、行业集中度、风格漂移、港股配比。",
            )
            report, _ = GoogleToolCallHandler.handle_google_tool_calls(
                result=result, llm=llm, tools=tools, state=state,
                analysis_prompt_template=analysis_prompt_template,
                analyst_name="基金持仓分析师",
            )
        else:
            report = ""
            if not getattr(result, "tool_calls", None):
                report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
            "sender": "FundHoldingsAnalyst",
        }

    return fund_holdings_analyst_node
