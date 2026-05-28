"""
基金经理画像分析师（my_fund_support 分支新增）

替代 social_media_analyst 用于 asset_type=fund 的场景。

国内公募「人」是核心要素之一：明星基金经理、新人、风格切换都直接影响业绩。
本分析师产出经理画像、任期、风格判断，对应"软性 / 主观"信息维度，
写入 sentiment_report 字段（下游消费者读取此字段不变）。
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.utils.logging_init import get_logger
from tradingagents.agents.utils.google_tool_handler import GoogleToolCallHandler

logger = get_logger("default")


def create_fund_manager_analyst(llm, toolkit):
    """创建基金经理画像分析师。签名与 create_social_media_analyst 一致。"""

    def fund_manager_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]

        logger.info(f"👤 [基金经理分析师] 开始: {ticker} @ {current_date}")

        tools = [
            toolkit.get_fund_basic_info_unified,
            toolkit.get_fund_holdings_unified,
        ]

        system_message = (
            "你是一位专业的基金经理画像分析师，与其他分析师协作。\n"
            "国内公募基金「认人」很关键：经理资历、管理规模、风格稳定度都直接影响选基决策。\n"
            "\n"
            "📋 **分析对象：**\n"
            f"- 基金代码: {ticker}\n"
            f"- 分析日期: {current_date}\n"
            "\n"
            "🔧 **工具调用流程（一次完成）：**\n"
            f"1. get_fund_basic_info_unified({ticker}) 拿基金经理姓名、任期、基金公司\n"
            f"2. get_fund_holdings_unified({ticker}) 通过持仓反推经理风格（成长 vs 价值 / 集中 vs 分散）\n"
            "3. 拿到后立即出报告，不要重复调用\n"
            "\n"
            "🎯 **分析维度：**\n"
            "- **基金经理信息**: 现任经理姓名、任期长度、是否多人共管\n"
            "- **基金公司背景**: 公司规模、口碑、是否有大平台支持\n"
            "- **风格判断**（从持仓反推）: 价值/成长/红利/主题，集中/分散，A股/港股配比\n"
            "- **管理稳定性**: 是否频繁更换经理，是否经理同时管理多只基金\n"
            "- **信任度评估**: 基于以上给出综合可信度评级\n"
            "\n"
            "📝 **输出格式（严格遵守）：**\n"
            "\n"
            "## 👤 基金经理画像\n"
            "- 现任经理：[姓名 1 / 姓名 2，如多人共管]\n"
            "- 基金公司：[基金公司]\n"
            "- 托管银行：[托管银行]\n"
            "- 成立至今：[X 年 Y 月]\n"
            "\n"
            "## 🎨 投资风格判断\n"
            "[基于持仓推断的风格 + 是否与档案里的投资策略一致]\n"
            "\n"
            "## 📐 管理稳定性\n"
            "[共管/单管？经理是否新近上任？]\n"
            "\n"
            "## 🧭 综合评估\n"
            "[给出对该基金经理的可信度评级（高/中/低）与理由]\n"
            "\n"
            "## 💭 投资视角结论\n"
            "[基于「人」的维度，对该基金的申购/赎回建议倾向]\n"
            "\n"
            "⚠️ 注意：\n"
            "- 全部使用中文\n"
            "- 不要编造经理的历史业绩数据；仅基于工具结果给出判断\n"
            "- 若工具未能返回经理信息，直接说明并给出降级评估\n"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(ticker=ticker)

        tool_names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools]
        logger.info(f"👤 [基金经理分析师] 绑定工具: {tool_names}")

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke({"messages": state["messages"]})

        if GoogleToolCallHandler.is_google_model(llm):
            analysis_prompt_template = GoogleToolCallHandler.create_analysis_prompt(
                ticker=ticker,
                company_name=f"基金{ticker}",
                analyst_type="基金经理画像分析",
                specific_requirements="重点关注基金经理资历、管理稳定性、风格判断。",
            )
            report, _ = GoogleToolCallHandler.handle_google_tool_calls(
                result=result, llm=llm, tools=tools, state=state,
                analysis_prompt_template=analysis_prompt_template,
                analyst_name="基金经理分析师",
            )
        else:
            report = ""
            if not getattr(result, "tool_calls", None):
                report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
            "sender": "FundManagerAnalyst",
        }

    return fund_manager_analyst_node
