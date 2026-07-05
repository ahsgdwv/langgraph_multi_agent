# -*- coding: utf-8 -*-
"""Internship resume content — edit placeholders before export."""

GITHUB = "https://github.com/ahsgdwv/langgraph_multi_agent"

RESUME = {
    "name": "XXX",
    "contact": "电话/微信：XXXXXX    邮箱：XXXXXX@qq.com    GitHub：github.com/ahsgdwv",
    "meta": "22岁  男    应聘岗位：AI 应用开发工程师（实习）",
    "education": [
        "天津科技大学    计算机/软件/物联网工程（请按实际填写）    本科",
        "20XX年09月 - 20XX年07月    天津",
        "GPA：X.XX/4.0（如有请填写，无则删除本行）",
    ],
    "skills": [
        "熟练使用 Python，熟悉面向对象、异常处理、类型注解及常用标准库；",
        "熟悉 LangGraph / LangChain 多 Agent 编排，了解 StateGraph、Supervisor 路由、",
        "  HITL 中断与 Checkpoint 持久化；",
        "熟悉 LLM 应用开发，包括 Function Calling、RAG、结构化输出及 DeepSeek 兼容接口；",
        "熟悉 FastAPI、Gradio，具备 HTTP API 封装与 Web Demo 联调经验；",
        "熟悉 pandas、SQLite，能完成数据导入、只读 SQL 查询及常见统计分析；",
        "了解 Chroma 向量库及文档切块、Embedding、相似度检索等 RAG 流程；",
        "熟悉 Git 版本管理，了解 Pydantic 数据校验及集成测试写法；",
        "了解数据结构与算法基础，持续刷题中。",
    ],
    "project_title": "快消渠道数据分析多 Agent 系统",
    "project_time": "20XX年XX月 - 20XX年XX月",
    "project_stack": "Python、LangGraph、Chroma、pandas、SQLite、FastAPI、Gradio、Pydantic",
    "project_desc": (
        "面向饮料快消行业的通用渠道周报自动化系统（不绑定单一品牌）。"
        "用户用中文列出分析任务，系统自动拆分、调度多 Agent 执行，"
        "整合 SQL、pandas 与 RAG 政策文档，输出结构化 Markdown 周报。"
    ),
    "project_work": [
        "基于 LangGraph 设计 Supervisor 多 Agent 工作流（10 节点），"
        "实现文档加载、任务拆分、HITL 确认、多 Worker 执行、质量复核重试与汇总；",
        "实现三类 Worker（data_analyst / researcher / executor）关键词路由，"
        "并设计 5 个 Skill 资产库，触发词匹配优先走确定性 SQL/pandas/RAG 逻辑；",
        "搭建 Chroma RAG 知识库与 SQLite 销售分析库，SQL 工具限制只读与行数上限；",
        "实现 Reflection 失败重试、Checkpoint 断点续跑及 Gradio + FastAPI 双入口；",
        "支持 BRAND_NAME 环境变量切换演示品牌，默认通用快消场景便于多岗位投递；",
        "编写 without_llm() 集成测试 12+ 用例，保证无 API Key 时稳定回归。",
    ],
    "project_gain": (
        "理解 Multi-Agent「规划—执行—质检—汇总」分层设计；"
        "积累 LangGraph 状态管理、工具可靠性设计与中文任务拆分等工程经验。"
    ),
    "project_link": f"项目开源：{GITHUB}",
    "honors": [
        "竞赛经历：（请按实际填写，无则删除）",
        "英语：CET-4 / CET-6（请按实际填写）",
        "奖学金：（请按实际填写，无则删除）",
        "其它：GitHub 维护开源 Agent 项目；阅读 LLM 应用开发相关文档。",
    ],
}
