# -*- coding: utf-8 -*-
"""LightGBM 滚动重训子包。

模块构成：
    config.py    超参、滚动窗口、标签等配置（借鉴 Qlib 官方 benchmark）
    data.py      特征/标签加载、横截面预处理、滚动窗口划分
    evaluate.py  预测评估（IC / ICIR / RankIC）
    train.py     滚动重训主入口（main()，可独立运行）
"""
