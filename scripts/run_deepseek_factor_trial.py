"""Generate exactly one isolated experimental factor with DeepSeek."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from quantmind_integration.policy import validate_candidate_batch


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env.deepseek"
MODEL = "deepseek-v4-flash"
ENDPOINT = "https://api.deepseek.com/chat/completions"
MEMORY_FILE = ROOT / "reports/factor_mining_memory.json"


def load_env() -> None:
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _install_ssl_certifi_fallback() -> None:
    """Windows 证书库损坏(ASN1 NOT_ENOUGH_DATA)时回退到 certifi CA bundle。"""
    try:
        import ssl
        import certifi
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass


def extract_json(text: str) -> dict:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", clean):
            try:
                value, _ = decoder.raw_decode(clean[match.start():])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
        raise ValueError("DeepSeek响应中没有可解析的JSON对象") from None


ALLOWED_FIELDS = ("open,high,low,close,volume,amount,vwap,adjclose,pct_chg,total_mv,circ_mv,"
                  "pe_ttm,pb,ps_ttm,turnover_rate,roe,roe_waa,roa,roic,grossprofit_margin,"
                  "netprofit_margin,current_ratio,quick_ratio,debt_to_assets,assets_turn,tr_yoy,"
                  "netprofit_yoy,op_yoy,ocf_yoy,roe_yoy,q_sales_yoy,q_op_qoq,"
                  "IND_REL_RET_20,IND_RESID_RET_1D,IND_RESID_MOM_20")
ALLOWED_OPS = ("DELAY,DELTA,TS_PCT_CHANGE,TS_MEAN,TS_SUM,TS_STD,TS_MEDIAN,TS_QUANTILE,TS_MIN,"
               "TS_MAX,TS_SKEW,TS_KURT,TS_POSITION,TS_RANK_PCT,TS_CORR,TS_COV,TS_MAD,TS_COUNT,"
               "TS_RSQUARE,TS_SLOPE,TS_DECAY_LINEAR,SAFE_DIV,CLIP,ABS,SIGN,LOG1P,SQRT,WHERE,"
               "FIN_LAG_REPORT,FIN_DELTA_REPORT")

# 本地核心库25因子族,用于告知模型哪些概念已覆盖(>0.7冗余会被剔除)。
CORE25_FAMILIES = ("LOW_TURNOVER/LN_TURNOVER(低换手)、AMIHUD_20(非流动性)、"
                   "CORR20/CORR60/CORD5/CORD60(价量相关)、VOL_CV20/RET_VOL_CORR5/RET_AMT_CORR20(量价配合)、"
                   "OBV_CHG_20(能量潮)、VSTD5/VOL_OF_VOL_20(波动)、KLOW/NEG_SKEW_20/IMAX60/MAX_DD60_V2/"
                   "CNTD10/CNTD60/CONSEC_UP5/WVMA30(分布极值/连涨/形态)、VOL_PRICE_DIVERGENCE、BP/EP(估值)、"
                   "Q_SALES_YOY(单季营收同比)")

# 已通过训练期周频10bps关卡的QM准入因子(与core25并列的冗余参照)。改动请同步config/quantmind_admitted_registry.json。
ADMITTED_QM = ("QM_DS_LIQUIDITY_AMPLIFICATION_TURNOVER_RETVOL(流动性放大/换手复合,负向)、"
               "QM_DS_IDIOVOL_60D(60日特质波动率,负向,周换手≈0.12)")


def _load_memory() -> dict:
    if not MEMORY_FILE.exists():
        return {}
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def build_prompt() -> str:
    """组装分层提示词:角色约束→中选标准→禁区→可探索方向→经验记忆注入。

    关键设计:把「什么是优质候选」的正面模板与「已证伪」的负面清单放在同一
    视野内,并让方向地图(category_direction)决定模型该去哪个经济大类探索,
    而不是靠人工在提示词里来回摇摆。
    """
    memory = _load_memory()
    sections: list[str] = []

    # 1) 角色与任务
    sections.append(
        "你是A股量化研究员,为一套周频调仓、10bps单边成本、long-only前10%分位组合挖掘新的实验因子。"
        "每次只提出一个候选,名称以QM_DS_开头。不评价收益,不访问测试期(2025-01-02之后一律禁止)。"
        "严格输出单个JSON对象(不要Markdown,不要解释性文字)。格式:\n"
        '{"FACTOR_NAME":{"description":"...","formula":"...","formulation":"与formula相同",'
        '"variables":{"close":"收盘价"},"inputs":["close"],"lookback":20,'
        '"availability":"after_close","direction":"positive|negative|learned","economic_rationale":"..."}}'
    )

    # 2) DSL硬性约束
    sections.append(
        f"DSL约束(违反即被程序拒绝):\n"
        f"可用字段(公式中必须带美元符号,如TS_MEAN($close,20);inputs写不带$字段名): {ALLOWED_FIELDS};\n"
        f"可用算子: {ALLOWED_OPS};\n"
        "禁止复合多个经济概念(一个候选只表达一个概念);禁止引用未来数据;\n"
        "财务字段($roe/$roa/$roic/$grossprofit_margin/$netprofit_margin/$current_ratio/$quick_ratio/"
        "$debt_to_assets/$assets_turn/$tr_yoy/$netprofit_yoy/$op_yoy/$ocf_yoy/$roe_yoy/$q_sales_yoy/$q_op_qoq)是公告后按日PIT对齐的低频值:"
        "禁止对财务字段使用DELAY/DELTA/任何TS_交易日滚动窗口;跨财报期只能用FIN_LAG_REPORT($field,n)/FIN_DELTA_REPORT($field,n),n为正整数。"
    )

    # 3) 中选标准(什么算好候选)
    sections.append(
        "中选标准(周频10bps主关卡,sharpe>=0.5、年化超额>=0、最大回撤>=-0.5):\n"
        "训练期5日标签|rank_ic|>=0.01且增量|partial_rank_ic|>=0.005只是入场券,真正决定命中的是周频回测。"
        "历史命中者(3个)共同画像:①来自价量/行业残差字段而非财务字段;②慢速窗口(>=20日)低换手——"
        "周均单边换手<=0.27是命中分界线,高换手概念在10bps下必被成本拖垮;③方向以负向为主(风险溢价/拥挤/异质波动等利空定价);"
        "④单一经济逻辑清晰。请先自检:我的候选换手代理(窗口长度、输入稳定性)是否达标?若窗口<20日或依赖逐日剧烈变化的输入,大概率失败。"
    )

    # 4) 禁区(已证伪/已收敛/已覆盖,分条列出,禁止触碰)
    forbidden = [
        "IND_RESID_RET_1D窗口搜索已收敛:TS_STD($IND_RESID_RET_1D,20/60)已命中(负向)。禁止再提出该字段的任何纯波动/偏度/矩/更长或更短窗口变体;"
        "120日版因缺数据率>15%被拒。若要使用$IND_RESID_RET_1D,只能走显著不同的变换(与其它字段交互成复合)。",
        "$IND_RESID_MOM_20、TS_MEAN($IND_RESID_RET_1D,60)、$IND_REL_RET_20、TS_SKEW($IND_RESID_RET_1D,20)作为方向/分布信号在周频失败。",
        "财务字段作主导概念已证伪:连续8+财务复合候选5日|rank_ic|<0.01或增量不足被拒(PIT公告对齐与5日收益窗口错配)。财务只能作二级调味,主导算子必须来自价量字段。",
        "财务单变量跨期差分($roe/$roic/$debt_to_assets/$assets_turn/$grossprofit_margin的FIN_DELTA_REPORT、$ocf_yoy-$netprofit_yoy)无增量。",
        "任何两个财务利润率字段直接相除(如净利率/毛利率)无增量且与估值因子冗余。",
        "任何两个同比增速字段直接相减/相加(增速剪刀差)无增量。",
        "Amihud非流动性族(成交额缩放绝对收益/价格冲击)与core25 AMIHUD_20冗余(秩相关0.57)。",
        "流通比例/股本供给类(如circ_mv/total_mv)无增量。",
        "价格水平与成交量水平的直接滚动相关性、成交量/换手率滚动变异系数(CV)——均已被core25覆盖。",
        "收盘价日位置类TS_MEAN(($close-$low)/($high-$low),N)已在周频失败。",
        "成交额放大比率SAFE_DIV(TS_MEAN($amount,20),TS_MEAN($amount,60))-1(流动性边际扩张,负向)在周频10bps失败:超额为负sharpe≈0.19。单独的长度比类流动性扩张已不工作,若要表达拥挤需与价格行为/波动复合。",
        "噪声放大复合TS_CORR(ABS($IND_RESID_RET_1D),$turnover_rate,20)(负向)周频10bps失败:换手0.47超额为负。ABS($IND_RESID_RET_1D)与$turnover_rate的滚动相关类已证伪。",
        "inputs数组必须是纯字段名列表(如[\"close\",\"vwap\"]),禁止放入公式片段或表达式——否则会触发PAYLOAD_SHAPE校验被拒。",
        f"与core25或已准入QM因子日秩相关绝对值>=0.7即被程序剔除。core25已覆盖族:{CORE25_FAMILIES}。已准入QM(新候选也须与它们<0.7):{ADMITTED_QM}。",
    ]
    sections.append("禁区(勿触碰,逐条阅读):\n" + "\n".join(f"- {line}" for line in forbidden))

    # 5) 方向地图:由记忆数据决定探索方向,替代人工摇摆
    category_map = memory.get("category_direction", {})
    if category_map:
        open_cats = [f"{v['label']}({k})" for k, v in sorted(category_map.items(), key=lambda x: -x[1]['trials'])
                     if v["status"] == "open"]
        closed_cats = [f"{v['label']}({k})" for k, v in sorted(category_map.items())
                       if v["status"] == "closed_sterile"]
        probing_cats = [f"{v['label']}({k})" for k, v in sorted(category_map.items(), key=lambda x: -x[1]['trials'])
                        if v["status"] == "probing"]
        dir_section = "方向地图(来自30次本地试验统计,status=open表明该大类出过命中):\n"
        if open_cats:
            dir_section += f"- 已产出命中、可继续探索的open大类: {', '.join(open_cats)}。请优先在这些大类内寻找与新命中者经济逻辑不同的新概念。\n"
        if probing_cats:
            dir_section += f"- 仍在试探(probing)的大类: {', '.join(probing_cats)}。若你确有强经济逻辑可在此探索,但需避开上面禁区清单中的具体证伪操作。\n"
        if closed_cats:
            dir_section += f"- 已证伪关闭(closed_sterile,>=3次试验且最佳|rank_ic|<0.02)的大类: {', '.join(closed_cats)}。禁止提出以此为主导的概念。\n"
        sections.append(dir_section)

    # 6) 经验记忆注入(压缩后附加)
    if memory:
        compact = {
            "outcome_counts": memory.get("outcome_counts", {}),
            "field_usage": memory.get("field_usage", {}),
            "exact_formulas_do_not_repeat": memory.get("exact_formulas_do_not_repeat", [])[-40:],
            "recent_experience": memory.get("recent_experience", [])[-6:],
            "generation_rules": memory.get("generation_rules", []),
        }
        sections.append(
            "本地经验记忆(压缩)。必须从失败中学习,禁止重复公式或只改名:\n"
            + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n".join(sections)


def main() -> int:
    load_env()
    _install_ssl_certifi_fallback()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(".env.deepseek中的DEEPSEEK_API_KEY为空")

    prompt = build_prompt()
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "不要输出任何思考、解释、分析或占位文本。content字段必须直接是可解析的JSON对象，且恰好包含一个候选因子，键为真实因子名（以QM_DS_开头），禁止使用FACTOR_NAME等占位键。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "stream": False,
        "response_format": {"type": "json_object"},
        "max_tokens": 16000,
    }).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw_response = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Never echo response headers or request data; the status/body contains no local key.
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from None

    choice = raw_response["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or message.get("reasoning_content") or ""
    if not content.strip():
        raise RuntimeError(f"DeepSeek返回空正文，finish_reason={choice.get('finish_reason')}")
    if not re.search(r"\{", content):
        import sys as _sys
        _sys.stderr.write("NO_JSON_CONTENT_PREFIX: " + repr(content[:1500]) + "\n")
    payload = extract_json(content)
    import sys as _sys2
    if not isinstance(payload, dict) or any(not isinstance(v, dict) for v in payload.values()):
        _sys2.stderr.write("PAYLOAD_SHAPE: " + repr(str(payload)[:800]) + "\n")
    candidates = []
    for factor_name, spec in payload.items():
        candidates.append({
            "factor_name": factor_name,
            "formula": spec.get("formula", spec.get("formulation")),
            "inputs": spec.get("inputs"),
            "lookback": spec.get("lookback"),
            "availability": spec.get("availability"),
            "direction": spec.get("direction"),
            "economic_rationale": spec.get("economic_rationale", spec.get("description")),
        })
    validate_candidate_batch(candidates)

    now = datetime.now(timezone.utc)
    run_dir = ROOT / "reports" / "quantmind_trials" / now.strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact = {
        "status": "experimental",
        "sota_write": False,
        "model": MODEL,
        "generated_at": now.isoformat(),
        "candidate_count": 1,
        "candidate": candidates[0],
        "provider_request_id": raw_response.get("id"),
        "usage": raw_response.get("usage", {}),
    }
    output = run_dir / "candidate.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(output), "candidate": candidates[0]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
