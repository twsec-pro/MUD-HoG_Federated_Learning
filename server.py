from __future__ import print_function

from copy import deepcopy

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import torch
import torch.nn.functional as F
import logging
from datetime import datetime
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from collections import defaultdict, Counter, deque
from scipy.spatial.distance import cosine

from utils import utils
from utils.backdoor_semantic_utils import SemanticBackdoor_Utils
from utils.backdoor_utils import Backdoor_Utils
import time
import json
import copy
import clients as cl


def find_threshold_by_max_gap(values, return_gap_info=False):
    """
    Finds a threshold by identifying the largest gap in a sorted list of values.
    This is useful for separating clusters of data.
    """
    if len(values) < 2:
        return 0.0 if not return_gap_info else (0.0, 0.0, 0.0, 0.0)

    sorted_values = sorted(values)
    max_gap = 0
    max_gap_idx = 0

    for i in range(len(sorted_values) - 1):
        gap = sorted_values[i + 1] - sorted_values[i]
        if gap > max_gap:
            max_gap = gap
            max_gap_idx = i

    lower_bound = sorted_values[max_gap_idx]
    upper_bound = sorted_values[max_gap_idx + 1]
    threshold = (lower_bound + upper_bound) / 2.0

    if return_gap_info:
        return threshold, max_gap, lower_bound, upper_bound
    else:
        return threshold


def detect_sign_flipping_with_global(short_HoGs, global_short):
    """
    Sign-Flipping Detection: Detects clients whose updates oppose the global direction.
    """
    flip_sign_id = set()
    if len(short_HoGs) < 3:
        return flip_sign_id
    global_norm = np.linalg.norm(global_short)
    if global_norm < 1e-10:
        return flip_sign_id
    for client_id, client_short in short_HoGs.items():
        client_short_vec = np.array(list(client_short))
        client_norm = np.linalg.norm(client_short_vec)
        if client_norm < 1e-10:
            continue
        cos_sim = np.dot(global_short, client_short_vec) / (global_norm * client_norm)
        if cos_sim < 0:
            flip_sign_id.add(client_id)
    logging.info(f"[Sign-Flip Detection] Detected: {len(flip_sign_id)} clients - {sorted(list(flip_sign_id))}")
    return flip_sign_id


def detect_noise_injection_with_global(short_HoGs, global_short, excluded_ids):
    """
    Noise Injection Detection: Detects clients with abnormally large gradient magnitudes.
    （本函数保留，但在当前版本的 mud_hog 中不再调用）
    """
    noise_injection_id = set()
    remaining_clients = {k: v for k, v in short_HoGs.items() if k not in excluded_ids}
    if len(remaining_clients) < 3:
        return noise_injection_id
    magnitudes = {cid: np.linalg.norm(np.array(list(c_short))) for cid, c_short in remaining_clients.items()}
    if not magnitudes:
        return noise_injection_id
    mag_values = np.array(list(magnitudes.values()))
    q1, q3 = np.percentile(mag_values, [25, 75])
    upper_bound = q3 + 1.5 * (q3 - q1)
    for client_id, mag in magnitudes.items():
        if mag > upper_bound:
            noise_injection_id.add(client_id)
    logging.info(f"[Noise-Injection Detection] Detected: {len(noise_injection_id)} clients - {sorted(list(noise_injection_id))}")
    return noise_injection_id


def find_best_threshold_adaptive(values, min_gap_size=0.05, gap_cos_upper=0.7):
    """
    自适应间隙选择
    
    Parameters:
    -----------
    values : list
        相似度值列表
    min_gap_size : float
        最小间隙阈值
    gap_cos_upper : float
        间隙中点上限
    
    Returns:
    --------
    float or None : 选定的阈值，无合适阈值返回None
    """
    if not values or len(values) < 2:
        logging.info(f"  [Adaptive-Gap] Too few values ({len(values) if values else 0})")
        return None

    sorted_values = sorted(values)
    all_gaps = []

    # 计算所有间隙
    for i in range(len(sorted_values) - 1):
        gap = sorted_values[i + 1] - sorted_values[i]
        midpoint = (sorted_values[i] + sorted_values[i + 1]) / 2
        all_gaps.append((gap, midpoint, i, sorted_values[i], sorted_values[i + 1]))

    if not all_gaps:
        logging.info(f"  [Adaptive-Gap] No gaps to analyze")
        return None

    # 按间隙大小降序排序
    all_gaps_sorted = sorted(all_gaps, key=lambda x: x[0], reverse=True)

    logging.info(f"  [Adaptive-Gap] Analyzing {len(all_gaps)} gaps, top 3:")
    for rank, (gap, mid, idx, low, up) in enumerate(all_gaps_sorted[:3], 1):
        logging.info(f"    #{rank}: gap={gap:.4f}, mid={mid:.4f}, range=[{low:.4f}, {up:.4f}]")

    # 选择第一个满足条件的间隙
    for gap, midpoint, idx, low, up in all_gaps_sorted:
        meets_size = gap > min_gap_size
        meets_upper = midpoint < gap_cos_upper

        if meets_size and meets_upper:
            logging.info(f"  [Adaptive-Gap] ✅ Selected: gap={gap:.4f}, mid={midpoint:.4f}")
            return midpoint

    logging.info(f"  [Adaptive-Gap] ❌ No gap satisfies both conditions (size>{min_gap_size}, mid<{gap_cos_upper})")
    return None


# ==================================================================================
# ✅✅✅ 改进：基于共识符号选举的两级检测 ✅✅✅
# ==================================================================================

def detect_label_flipping_two_level(long_HoGs, excluded_ids,
                                    hard_threshold=0.0,
                                    min_gap_size=0.1,
                                    gap_cos_upper=0.7,
                                    min_normal_ratio=0.3):
    """
    ✅ 改进版两级检测策略：基于共识符号选举
    
    核心改进：
    1. 先计算初步共识方向
    2. 统计客户端与共识方向的余弦相似度符号（正/负）
    3. 通过投票决定最终共识符号：超过一半为正则共识为正，否则为负
    4. Level 1: 检测与共识符号相反的客户端
    5. Level 2: 在同符号客户端中使用自适应间隙检测微妙攻击
    """
    detected_attackers = set()
    remaining_clients = {k: v for k, v in long_HoGs.items() if k not in excluded_ids}

    total_clients = len(remaining_clients)

    if total_clients < 4:
        logging.info("[LFD-ConsensusVoting] Too few clients, skipping detection.")
        return detected_attackers

    logging.info("=" * 80)
    logging.info("LABEL-FLIPPING DETECTION (Consensus Sign Voting + Two-Level)")
    logging.info("=" * 80)
    logging.info(f"Total clients to analyze: {total_clients}")
    logging.info(f"Excluded by previous detection: {sorted(list(excluded_ids))}")
    logging.info(f"Parameters:")
    logging.info(f"  hard_threshold:        {hard_threshold}")
    logging.info(f"  min_gap_size:          {min_gap_size}")
    logging.info(f"  gap_cos_upper:         {gap_cos_upper}")
    logging.info(f"  min_normal_ratio:      {min_normal_ratio}")

    # ========================================================================
    # STEP 1: 提取长历史向量（最后两层参数）
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STEP 1: Extracting Long-History Vectors (Last 2 Layers)")
    logging.info("=" * 80)

    try:
        sample_state_dict = next(iter(remaining_clients.values()))
        last_layer_keys = list(sample_state_dict.keys())[-2:]
        logging.info(f"Using last 2 layers: {last_layer_keys}")
    except (StopIteration, IndexError):
        logging.error("[LFD-ConsensusVoting] Cannot extract layer keys.")
        return detected_attackers

    client_vectors = []
    client_ids = []

    for cid, state_dict in remaining_clients.items():
        try:
            vec = torch.cat([state_dict[key].flatten() for key in last_layer_keys]).cpu().numpy()
        except KeyError:
            logging.warning(f"  Client {cid} missing keys. Skipping.")
            continue

        if np.isfinite(vec).all():
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                client_vectors.append(vec / norm)
                client_ids.append(cid)
            else:
                logging.warning(f"  Client {cid} has zero norm. Skipping.")
        else:
            logging.warning(f"  Client {cid} has non-finite values. Skipping.")

    if len(client_ids) < 4:
        logging.warning("[LFD-ConsensusVoting] Not enough valid vectors after filtering.")
        return detected_attackers

    logging.info(f"✅ Valid vectors: {len(client_ids)} clients -> {client_ids}")

    # ========================================================================
    # STEP 2: 计算初步加权共识方向
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STEP 2: Computing Initial Weighted Consensus Direction")
    logging.info("=" * 80)

    client_vectors_np = np.array(client_vectors)

    # 计算余弦相似度矩阵
    cosine_matrix = client_vectors_np @ client_vectors_np.T

    # 计算支持分数（每个客户端与其他客户端的相似度之和）
    support_scores = np.sum(cosine_matrix, axis=1) - 1  # 减1排除自己

    # 使用softmax计算权重
    weights = np.exp(support_scores) / np.sum(np.exp(support_scores))

    # 加权平均得到初步共识方向
    consensus_initial = np.average(client_vectors_np, axis=0, weights=weights)
    consensus_norm = np.linalg.norm(consensus_initial)

    if consensus_norm < 1e-10:
        logging.error("[LFD-ConsensusVoting] Consensus norm too small. Aborting.")
        return detected_attackers

    consensus_initial_normalized = consensus_initial / consensus_norm

    logging.info(f"✅ Initial consensus direction computed (norm={consensus_norm:.6f})")

    # ========================================================================
    # STEP 3: 计算每个客户端与初步共识的余弦相似度
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STEP 3: Computing Initial Similarities to Consensus")
    logging.info("=" * 80)

    initial_similarities = {}
    for i, cid in enumerate(client_ids):
        initial_similarities[cid] = np.dot(client_vectors_np[i], consensus_initial_normalized)

    # 打印初步相似度
    logging.info(f"\nInitial Cosine Similarities (sorted ascending):")
    logging.info(f"{'CID':<5} {'Similarity':<12} {'Sign':<6}")
    logging.info("-" * 25)

    for cid in sorted(client_ids, key=lambda x: initial_similarities[x]):
        sim = initial_similarities[cid]
        sign = "POS" if sim >= 0 else "NEG"
        logging.info(f"{cid:<5} {sim:+.6f}      {sign}")

    # ========================================================================
    # STEP 4: 🗳️ 共识符号选举（投票机制）
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STEP 4: 🗳️ Consensus Sign Voting (Majority Vote)")
    logging.info("=" * 80)

    # 统计正负符号的客户端数量
    positive_count = sum(1 for sim in initial_similarities.values() if sim >= 0)
    negative_count = len(initial_similarities) - positive_count

    # 投票决定共识符号
    consensus_sign = 1 if positive_count > len(initial_similarities) / 2 else -1
    consensus_sign_str = "POSITIVE" if consensus_sign > 0 else "NEGATIVE"

    logging.info(f"Voting Results:")
    logging.info(f"  Positive similarity clients: {positive_count} ({positive_count/len(initial_similarities):.1%})")
    logging.info(f"  Negative similarity clients: {negative_count} ({negative_count/len(initial_similarities):.1%})")
    logging.info(f"  {'─' * 60}")
    logging.info(f"  ✅ Consensus Sign (by majority): {consensus_sign_str} (sign={consensus_sign:+d})")

    # 如果需要，翻转共识方向
    if consensus_sign < 0:
        consensus_normalized = -consensus_initial_normalized
        logging.info(f"  🔄 Flipping consensus direction to match negative majority")
    else:
        consensus_normalized = consensus_initial_normalized
        logging.info(f"  ✅ Keeping consensus direction (positive majority)")

    # 重新计算调整后的相似度
    final_similarities = {}
    for i, cid in enumerate(client_ids):
        final_similarities[cid] = np.dot(client_vectors_np[i], consensus_normalized)

    # 打印最终相似度
    logging.info(f"\n" + "=" * 80)
    logging.info("STEP 5: Final Similarities After Consensus Adjustment")
    logging.info("=" * 80)
    logging.info(f"{'CID':<5} {'Similarity':<12} {'Sign':<6}")
    logging.info("-" * 25)

    for cid in sorted(client_ids, key=lambda x: final_similarities[x]):
        sim = final_similarities[cid]
        sign = "POS" if sim >= 0 else "NEG"
        logging.info(f"{cid:<5} {sim:+.6f}      {sign}")

    # 统计信息
    sims_array = np.array(list(final_similarities.values()))
    logging.info(f"\nFinal Similarity Statistics:")
    logging.info(f"  Min:    {np.min(sims_array):+.6f}")
    logging.info(f"  Max:    {np.max(sims_array):+.6f}")
    logging.info(f"  Mean:   {np.mean(sims_array):+.6f}")
    logging.info(f"  Median: {np.median(sims_array):+.6f}")
    logging.info(f"  Std:    {np.std(sims_array):.6f}")

    # ========================================================================
    # STEP 6: 两级检测策略
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STEP 6: Two-Level Detection Strategy")
    logging.info("=" * 80)

    level1_attackers = set()
    level2_attackers = set()

    # === Level 1: 符号相反检测（与共识符号相反） ===
    logging.info(f"\n┌─ Level 1: Sign Mismatch Detection (similarity < {hard_threshold}) ─┐")
    level1_attackers = {cid for cid, sim in final_similarities.items() if sim < hard_threshold}

    if level1_attackers:
        logging.warning(f"│ 🚨 Level 1 detected {len(level1_attackers)} attackers (opposite to consensus):")
        for cid in sorted(level1_attackers):
            logging.warning(f"│   Client {cid:02d}: similarity={final_similarities[cid]:+.6f}")
    else:
        logging.info(f"│ ✅ No attackers with opposite sign to consensus")

    logging.info(f"└{'─' * 60}┘")

    # === Level 2: 自适应间隙法（在同符号客户端中检测微妙攻击） ===
    logging.info(f"\n┌─ Level 2: Adaptive Gap Method (among same-sign clients) ─┐")
    remaining_for_gap = {cid: sim for cid, sim in final_similarities.items() if cid not in level1_attackers}

    if len(remaining_for_gap) >= 2:
        logging.info(f"│ Remaining clients for gap analysis: {len(remaining_for_gap)}")
        logging.info(f"│ Similarity range: [{min(remaining_for_gap.values()):+.6f}, {max(remaining_for_gap.values()):+.6f}]")

        gap_threshold = find_best_threshold_adaptive(
            list(remaining_for_gap.values()),
            min_gap_size=min_gap_size,
            gap_cos_upper=gap_cos_upper
        )

        if gap_threshold is not None:
            level2_attackers = {cid for cid, sim in remaining_for_gap.items() if sim < gap_threshold}

            logging.info(f"│ Gap threshold: {gap_threshold:+.6f}")

            if level2_attackers:
                logging.warning(f"│ 🚨 Level 2 detected {len(level2_attackers)} subtle attackers:")
                for cid in sorted(level2_attackers):
                    logging.warning(f"│   Client {cid:02d}: similarity={remaining_for_gap[cid]:+.6f}")
            else:
                logging.info(f"│ ✅ No subtle attackers detected by gap method")
        else:
            logging.info(f"│ ℹ️  No significant gap found")
    else:
        logging.info(f"│ ⚠️  Skipped (not enough remaining clients)")

    logging.info(f"└{'─' * 60}┘")

    # 综合两级检测结果
    detected_attackers = level1_attackers | level2_attackers

    # ========================================================================
    # STEP 7: 安全检查（基数验证）
    # ========================================================================
    num_detected = len(detected_attackers)
    num_remaining = total_clients - num_detected
    remaining_ratio = num_remaining / total_clients

    logging.info("\n" + "=" * 80)
    logging.info("🛡️  STEP 7: Safety Check - Cardinality Validation")
    logging.info("=" * 80)
    logging.info(f"  Total clients:         {total_clients}")
    logging.info(f"  Detected as malicious: {num_detected} ({num_detected/total_clients:.1%})")
    logging.info(f"  Remaining as normal:   {num_remaining} ({remaining_ratio:.1%})")
    logging.info(f"  Minimum required:      {int(total_clients * min_normal_ratio)} ({min_normal_ratio:.1%})")

    if remaining_ratio < min_normal_ratio:
        logging.error("=" * 80)
        logging.error("⚠️⚠️⚠️  CRITICAL ALERT: DETECTION ANOMALY!")
        logging.error("=" * 80)
        logging.error(f"  Too many clients flagged as malicious!")
        logging.error(f"  SAFETY ACTION: Using only Level 1 (sign mismatch)...")

        # 退化策略：只使用Level 1
        if len(level1_attackers) <= total_clients * (1 - min_normal_ratio):
            detected_attackers = level1_attackers
            logging.error(f"  → Using only Level 1: {sorted(list(detected_attackers))}")
        else:
            logging.error(f"  → Even Level 1 flagged too many. DISABLING detection this round.")
            detected_attackers = set()

        logging.error("=" * 80)
    else:
        logging.info(f"✅ SAFETY CHECK PASSED: {remaining_ratio:.1%} >= {min_normal_ratio:.1%}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("LABEL-FLIPPING DETECTION - FINAL SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Detection Method: Consensus Sign Voting + Two-Level")
    logging.info(f"Consensus Sign: {consensus_sign_str}")
    logging.info(f"")
    logging.info(f"Level 1 (Sign Mismatch):   {len(level1_attackers):>2} clients -> {sorted(list(level1_attackers))}")
    logging.info(f"Level 2 (Adaptive Gap):    {len(level2_attackers):>2} clients -> {sorted(list(level2_attackers))}")
    logging.info(f"{'─' * 80}")
    logging.info(f"Total Detected:            {len(detected_attackers):>2} clients -> {sorted(list(detected_attackers))}")

    normal_clients = sorted([cid for cid in client_ids if cid not in detected_attackers])
    logging.info(f"Normal Clients:            {len(normal_clients):>2} clients -> {normal_clients}")
    logging.info("=" * 80 + "\n")

    return detected_attackers


class Server():
    def __init__(self, model, dataLoader, criterion=F.nll_loss, device='cpu'):
        self.clients = []
        self.model = model
        self.dataLoader = dataLoader
        self.device = device
        self.emptyStates = None
        self.init_stateChange()
        self.Delta = None
        self.iter = 0
        self.AR = self.FedAvg
        self.func = torch.mean
        self.isSaveChanges = False
        self.savePath = './AggData'
        self.criterion = criterion
        self.path_to_aggNet = ""
        self.sims = None
        self.mal_ids = set()
        self.uAtk_ids = set()
        self.tAtk_ids = set()
        self.flip_sign_ids = set()
        self.unreliable_ids = set()
        self.suspicious_id = set()
        self.log_sims = None
        self.log_norms = None
        self.tao_0 = 2
        self.delay_decision = 2
        self.pre_mal_id = defaultdict(int)
        self.count_unreliable = defaultdict(int)
        self.long = deepcopy(self.model.state_dict())
        self.short = deepcopy(self.model.state_dict())
        self.K_avg_s = 3
        self.hog_avg_s = deque(maxlen=self.K_avg_s)
        self.normal_clients = []

        # 声誉系统参数
        self.malice_scores = defaultdict(int)
        self.MALICE_THRESHOLD = 3
        self.MALICE_DECAY = 1
        self.MALICE_INCREASE = 1

        # Label-flipping detection parameters (两级检测)
        self.label_flip_hard_threshold = 0.0
        self.label_flip_min_gap_size = 0.1
        self.label_flip_gap_cos_upper = 0.7
        self.label_flip_min_normal_ratio = 0.3

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def set_log_path(self, log_path, exp_name, t_run):
        self.log_path = log_path
        self.savePath = log_path
        self.log_results = f'{log_path}/acc_prec_rec_f1_{exp_name}_{t_run}.txt'
        self.output_file = open(self.log_results, 'w', encoding='utf-8')

    def close(self):
        if hasattr(self, 'output_file') and not self.output_file.closed:
            self.output_file.close()

    def saveChanges(self, clients):
        if not self.isSaveChanges:
            return
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        param_trainable = utils.getTrainableParameters(self.model)
        param_nontrainable = [param for param in Delta.keys() if param not in param_trainable]
        for param in param_nontrainable:
            del Delta[param]
        for param in param_trainable:
            param_stack = torch.stack([delta[param] for delta in deltas], -1)
            Delta[param] = param_stack.view(-1, len(clients))
        savepath = f'{self.savePath}/{self.iter}.pt'
        torch.save(Delta, savepath)
        logging.info(f'[Server] Update vectors saved to {savepath}')

    def attach(self, c):
        self.clients.append(c)
        self.num_clients = len(self.clients)

    def distribute(self):
        for c in self.clients:
            c.setModelParameter(self.model.state_dict())

    def test(self):
        logging.info("[Server] Start testing")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        count = 0
        nb_classes = 10  # for MNIST, Fashion-MNIST, CIFAR-10
        cf_matrix = torch.zeros(nb_classes, nb_classes)

        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss

                if output.dim() == 1:
                    pred = torch.round(torch.sigmoid(output))
                else:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                correct += pred.eq(target.view_as(pred)).sum().item()
                count += pred.shape[0]

                # 构建混淆矩阵
                for t, p in zip(target.view(-1), pred.view(-1)):
                    cf_matrix[t.long(), p.long()] += 1

        if count == 0:
            return 0.0, 0.0

        test_loss /= count
        accuracy = 100. * correct / count

        self.model.cpu()  # avoid occupying gpu when idle

        logging.info(
            '[Server] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
                test_loss, correct, count, accuracy))

        logging.info(f"[Sever] Confusion matrix:\n {cf_matrix.detach().cpu()}")

        # 计算精度、召回率、F1分数
        cf_matrix_np = cf_matrix.detach().cpu().numpy()
        row_sum = np.sum(cf_matrix_np, axis=0)  # predicted counts
        col_sum = np.sum(cf_matrix_np, axis=1)  # targeted counts
        diag = np.diag(cf_matrix_np)

        precision = diag / row_sum  # tp/(tp+fp), p is predicted positive.
        recall = diag / col_sum  # tp/(tp+fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        m_acc = np.sum(diag) / np.sum(cf_matrix_np)

        # 构建输出结果字典
        results = {
            'accuracy': accuracy,
            'test_loss': test_loss,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'confusion': cf_matrix_np.tolist(),
            'epoch': self.iter
        }

        # 写入文件
        if hasattr(self, 'output_file') and not self.output_file.closed:
            json.dump(results, self.output_file)
            self.output_file.write("\n")
            self.output_file.flush()

        logging.info(f"[Server] Precision={precision},\n Recall={recall},\n F1-score={f1},\n my_accuracy={m_acc*100.}[%]")

        return test_loss, accuracy

    def train(self, group):
        selectedClients = [self.clients[i] for i in group]
        for c in selectedClients:
            c.train()
            c.update()
        if self.isSaveChanges:
            self.saveChanges(selectedClients)
        Delta = self.AR(selectedClients)
        if Delta is None:
            self.iter += 1
            return
        for param in self.model.state_dict():
            self.model.state_dict()[param] += Delta[param]
            self.long[param] += self.model.state_dict()[param]
        K_s = len(self.hog_avg_s)
        for param in self.model.state_dict():
            if K_s == 0:
                self.short[param] = self.model.state_dict()[param]
            elif K_s < self.K_avg_s:
                self.short[param] = (self.short[param] * K_s + self.model.state_dict()[param]) / (K_s + 1)
            else:
                self.short[param] += (self.model.state_dict()[param] - self.hog_avg_s[0][param]) / self.K_avg_s
        self.hog_avg_s.append(deepcopy(self.model.state_dict()))
        self.iter += 1

    def set_AR_param(self, hard_threshold=0.0, min_gap_size=0.1, gap_cos_upper=0.7, min_normal_ratio=0.3):
        """设置标签翻转检测参数（两级检测）"""
        self.label_flip_hard_threshold = hard_threshold
        self.label_flip_min_gap_size = min_gap_size
        self.label_flip_gap_cos_upper = gap_cos_upper
        self.label_flip_min_normal_ratio = min_normal_ratio

        logging.info(f"[Label-Flip Params - Consensus Voting]")
        logging.info(f"  hard_threshold:        {hard_threshold}")
        logging.info(f"  min_gap_size:          {min_gap_size}")
        logging.info(f"  gap_cos_upper:         {gap_cos_upper}")
        logging.info(f"  min_normal_ratio:      {min_normal_ratio}")

    def set_AR(self, ar):
        if ar == 'mudhog':
            self.AR = self.mud_hog
        else:
            self.AR = self.FedAvg
        logging.info(f"Aggregation rule set to: {ar}")

    def add_mal_id_reputation(self, suspicions):
        """
        声誉系统：根据检测结果更新客户端恶意分数

        1. 检测到恶意 → 分数+1
        2. 未检测到恶意 → 分数-1（但不低于0）
        3. 分数 >= MALICE_THRESHOLD → 加入黑名单
        4. 分数降至0才能从黑名单移除
        """
        logging.info("-" * 80)
        logging.info("REPUTATION SYSTEM: Updating Malice Scores")
        logging.info("-" * 80)

        for i in range(self.num_clients):
            score_before = self.malice_scores[i]
            was_blacklisted = i in self.mal_ids

            if i in suspicions:
                # 检测到恶意行为：分数增加
                self.malice_scores[i] += self.MALICE_INCREASE
                logging.info(
                    f"  Client {i:02d}: SUSPICIOUS.   Score: {score_before} -> {self.malice_scores[i]} (+{self.MALICE_INCREASE})")
            elif self.malice_scores[i] > 0:
                # 未检测到恶意行为：分数衰减
                self.malice_scores[i] = max(0, self.malice_scores[i] - self.MALICE_DECAY)
                logging.info(
                    f"  Client {i:02d}: NORMAL.        Score decay: {score_before} -> {self.malice_scores[i]} (-{self.MALICE_DECAY})")

            # 判断是否加入黑名单
            if self.malice_scores[i] >= self.MALICE_THRESHOLD:
                if i not in self.mal_ids:
                    logging.warning(f"  🚨 Client {i:02d} CROSSED MALICE THRESHOLD! Blacklisted.")
                    self.mal_ids.add(i)

            # 只有当分数降至0时，才能从黑名单移除
            elif i in self.mal_ids:
                if self.malice_scores[i] == 0:
                    logging.info(f"  🛡️ Client {i:02d} score reduced to 0. Pardoned from blacklist.")
                    self.mal_ids.remove(i)
                else:
                    logging.info(
                        f"  ⚠️  Client {i:02d} still blacklisted (score={self.malice_scores[i]}, needs 0 to be pardoned)")

        logging.info("-" * 80)
        logging.info(f"Current Blacklist: {sorted(list(self.mal_ids))}")
        logging.info(f"Malice Scores: {dict(sorted([(k, v) for k, v in self.malice_scores.items() if v > 0]))}")
        logging.info("-" * 80)

    def mud_hog(self, clients):
        """
        MUD-HoG with Sign-Flipping + Label-Flipping (Two-Level) Detection + Reputation System

        根据要求：只检测符号翻转攻击和标签翻转攻击，不检测噪声注入攻击。
        """
        if self.iter >= self.tao_0 and (self.iter + 1) % (self.tao_0 + 1) == 0:
            logging.info("\n" + "=" * 100)
            logging.info(f"{'':=^100}")
            logging.info(f"{'DETECTION ROUND ' + str(self.iter):=^100}")
            logging.info(f"{'':=^100}")
            logging.info("=" * 100 + "\n")

            # 收集长短历史
            long_HoGs = {i: c.get_sum_hog_old() for i, c in enumerate(clients)}
            short_HoGs = {i: c.get_avg_grad().detach().cpu().numpy() for i, c in enumerate(clients)}
            global_short = torch.cat([v.flatten() for v in self.short.values()]).cpu().numpy()

            # Stage 1: 符号翻转检测
            logging.info("┌" + "─" * 78 + "┐")
            logging.info("│" + " STAGE 1: SIGN-FLIPPING DETECTION".center(78) + "│")
            logging.info("└" + "─" * 78 + "┘")
            flip_sign_id = detect_sign_flipping_with_global(short_HoGs, global_short)

            # Stage 2: 标签翻转检测（共识符号选举 + 两级检测）
            logging.info("\n┌" + "─" * 78 + "┐")
            logging.info("│" + " STAGE 2: LABEL-FLIPPING (Consensus Voting + Two-Level)".center(78) + "│")
            logging.info("└" + "─" * 78 + "┘")
            all_pre_detected = flip_sign_id  # 仅将符号翻转检测结果作为前置排除

            tAtk_id = detect_label_flipping_two_level(
                long_HoGs,
                all_pre_detected,
                hard_threshold=self.label_flip_hard_threshold,
                min_gap_size=self.label_flip_min_gap_size,
                gap_cos_upper=self.label_flip_gap_cos_upper,
                min_normal_ratio=self.label_flip_min_normal_ratio
            )

            # 不进行噪声注入检测，保持空集合占位
            uAtk_id = set()

            # Stage 3: 声誉系统
            flip_sign_id = flip_sign_id or set()
            tAtk_id = tAtk_id or set()
            all_suspicions = flip_sign_id.union(tAtk_id)

            logging.info("\n┌" + "─" * 78 + "┐")
            logging.info("│" + " STAGE 3: REPUTATION SYSTEM UPDATE".center(78) + "│")
            logging.info("└" + "─" * 78 + "┘")
            self.add_mal_id_reputation(all_suspicions)

            # 过滤正常客户端
            self.normal_clients = [c for i, c in enumerate(clients) if i not in self.mal_ids]
            if not self.normal_clients:
                logging.warning("[MUD-HoG] All clients blacklisted! Using all clients to prevent collapse.")
                self.normal_clients = clients

            # 最终汇总
            logging.info("\n" + "=" * 100)
            logging.info(f"{'DETECTION SUMMARY - Round ' + str(self.iter):^100}")
            logging.info("=" * 100)
            logging.info(f"  Sign-Flip Detection:     {len(flip_sign_id):>3} clients -> {sorted(list(flip_sign_id))}")
            logging.info(f"  Noise-Injection:         {len(uAtk_id):>3} clients -> {sorted(list(uAtk_id))}")
            logging.info(f"  Label-Flip (Voting):     {len(tAtk_id):>3} clients -> {sorted(list(tAtk_id))}")
            logging.info("-" * 100)
            logging.info(f"  Total Suspicions:        {len(all_suspicions):>3} clients -> {sorted(list(all_suspicions))}")
            logging.info(f"  Current Blacklist:       {len(self.mal_ids):>3} clients -> {sorted(list(self.mal_ids))}")
            logging.info(f"  Aggregating From:        {len(self.normal_clients):>3} normal clients")
            logging.info("=" * 100 + "\n")

            Delta = self._multi_chain_aggregate_fast(self.normal_clients)
        else:
            # 非检测轮次：使用上一轮的正常客户端列表
            if not hasattr(self, 'normal_clients') or not self.normal_clients:
                self.normal_clients = clients
            Delta = self._multi_chain_aggregate_fast(self.normal_clients)

        return Delta

    def _multi_chain_aggregate_fast(self, client_list):
        """Fast aggregation using vectorized operations"""
        Delta = deepcopy(self.emptyStates)
        N = len(client_list)
        if N == 0:
            return Delta
        vecs_list = [utils.net2vec(c.getDelta()) for c in client_list]
        vecs = torch.stack([v for v in vecs_list if torch.isfinite(v).all()])
        if vecs.shape[0] == 0:
            return Delta
        w_avg = torch.mean(vecs, dim=0)
        utils.vec2net(w_avg, Delta)
        return Delta

    def FedAvg(self, clients):
        """Standard FedAvg aggregation"""
        return self._multi_chain_aggregate_fast(clients)

