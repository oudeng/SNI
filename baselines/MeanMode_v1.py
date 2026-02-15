# MeanMode_v1.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class MeanModeImputer:
    """
    对混合型数据（连续 + 分类）执行简单的均值/众数插补：
      - 连续变量：缺失值替换为该列在完整数据中的全局均值
      - 分类变量：缺失值替换为该列在完整数据中的全局众数

    接口示例：
        from MeanMode_v1 import MeanModeImputer
        imp = MeanModeImputer(
            categorical_vars=[...],
            continuous_vars=[...]
        )
        X_imputed, _ = imp.impute(X_incomplete_df, X_missing_df)
    """

    def __init__(self, categorical_vars, continuous_vars):
        """
        参数：
            categorical_vars: list of str
                分类特征列名列表（必须与 DataFrame 中列名完全一致）。
            continuous_vars: list of str
                连续特征列名列表。
        """
        self.categorical_vars = categorical_vars
        self.continuous_vars = continuous_vars

    def impute(self, X_incomplete, X_missing):
        """
        对缺失数据执行均值/众数插补。

        参数：
            X_incomplete: pandas.DataFrame
                “完整”数据，DataFrame 形状与列顺序与 X_missing 一致，仅用于计算全局均值/众数。
            X_missing: pandas.DataFrame
                带缺失数据的 DataFrame（shape 与 X_incomplete 完全相同）。

        返回：
            X_imputed: pandas.DataFrame
                插补后的 DataFrame（与 X_missing 形状相同，分类列保持原 dtype）。
            None: 本方法不返回模型。
        """
        # 1. 校验形状一致
        if X_incomplete.shape != X_missing.shape:
            raise ValueError("X_incomplete 和 X_missing 的形状必须一致！")

        # 2. 复制一份 X_missing，防止原数据被修改
        df_imp = X_missing.copy().reset_index(drop=True)

        # 3. 对连续列执行均值插补
        for col in self.continuous_vars:
            if col not in df_imp.columns:
                raise ValueError(f"连续变量 '{col}' 不存在于 DataFrame 中。")
            # 计算 X_incomplete 上的全局均值（忽略缺失）
            col_mean = X_incomplete[col].astype(float).mean(skipna=True)
            # 将 df_imp[col] 上的缺失位置填充为该均值
            df_imp[col] = df_imp[col].astype(float).fillna(col_mean)

        # 4. 对分类列执行众数插补
        for col in self.categorical_vars:
            if col not in df_imp.columns:
                raise ValueError(f"分类变量 '{col}' 不存在于 DataFrame 中。")
            # 计算 X_incomplete 上的众数（mode）。可能有多个，要选第一个
            mode_series = X_incomplete[col].mode(dropna=True)
            if mode_series.shape[0] == 0:
                # 如果完整数据中该列全为 NaN，则无法计算众数，跳过
                continue
            col_mode = mode_series.iloc[0]
            # 将 df_imp[col] 上的缺失位置填充为该众数
            df_imp[col] = df_imp[col].fillna(col_mode)

        return df_imp, None