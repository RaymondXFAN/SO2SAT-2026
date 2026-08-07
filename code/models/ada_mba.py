# code/models/ada_mba.py
# Adaptive Multi-Branch Adapter (Ada-MBA): 空间门控 + 跨注意力融合
# ★ Phase 6: 支持skip_gate(消融no_gate模式) + n_branches可配置(2或3)
# ★ Phase 6核心: STMN作为独立第3纹理分支，不再加法增强S2
import torch
import torch.nn as nn


class AdaMBA(nn.Module):
    """自适应多分支融合。
    ★ Phase 6: 支持2分支(S2+S1)和3分支(S2+S1+STMN纹理)
    输入: feats = [f_s2_norm, f_s1_norm] (2分支) 或 [f_s2_norm, f_s1_norm, f_stmn_norm] (3分支)
    输出: (B, 256)
    融合方式: 自注意力 + 跨注意力 + 门控(可选skip_gate)
    """
    def __init__(self, d=256, heads=8, n_branches=3, skip_gate=False):
        super().__init__()
        self.d = d
        self.n_branches = n_branches
        self.skip_gate = skip_gate  # ★ Phase 6: no_gate消融时跳过门控

        # 自注意力（每分支内部）
        self.sa = nn.ModuleList([
            nn.MultiheadAttention(d, heads, batch_first=True)
            for _ in range(n_branches)
        ])

        # 跨注意力（所有对的组合）
        keys = [(i, j) for i in range(n_branches) for j in range(n_branches) if i < j]
        self.keys = keys
        self.cross = nn.ModuleDict({
            f"{i}-{j}": nn.MultiheadAttention(d, heads, batch_first=True)
            for i, j in keys
        })

        # 门控: α = σ(Wg [F_m; F_n])
        # ★ skip_gate=True时仍创建Wg(保持结构一致)，但forward中不使用
        self.Wg = nn.ModuleDict({
            f"{i}-{j}": nn.Linear(2 * d, 1)
            for i, j in keys
        })

        # 融合降维: (n_branches + n_cross) 个 (B,d) → (B,d)
        total_branches = n_branches + len(keys)
        self.reduce = nn.Conv1d(total_branches, 1, 1)

    def gate(self, a, b, key):
        return torch.sigmoid(self.Wg[key](torch.cat([a, b], -1)))  # (B, 1)

    def forward(self, feats):
        """
        feats: list[n_branches] of (B, 256)
        ★ skip_gate=True时: 跨注意力结果直接输出，不做α混合
        """
        outs = []
        # 自注意力
        for i, f in enumerate(feats):
            f_in = f.unsqueeze(1)  # (B, 1, d)
            sa_out, _ = self.sa[i](f_in, f_in, f_in)
            outs.append(sa_out.squeeze(1))  # (B, d)

        # 跨注意力 + 门控(或skip_gate)
        for i, j in self.keys:
            key = f"{i}-{j}"
            ca, _ = self.cross[key](
                feats[i].unsqueeze(1), feats[j].unsqueeze(1), feats[j].unsqueeze(1)
            )
            ca = ca.squeeze(1)  # (B, d)
            if self.skip_gate:
                # ★ Phase 6 no_gate消融: 直接用跨注意力结果，不做门控混合
                outs.append(ca)
            else:
                a = self.gate(feats[i], feats[j], key)  # (B, 1)
                outs.append(a * ca + (1 - a) * feats[i])

        stacked = torch.stack(outs, 1)  # (B, total_branches, d)
        return self.reduce(stacked).squeeze(1)  # (B, d)

    def gate_stats(self, feats):
        """返回门权重均值（用于可视化/分析）。★ skip_gate时仍可计算(但不影响forward)"""
        with torch.no_grad():
            return {
                f"{i}-{j}": float(self.gate(feats[i], feats[j], f"{i}-{j}").mean())
                for i, j in self.keys
            }
