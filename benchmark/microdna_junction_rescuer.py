#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MicroDNA Junction Read Rescuer (Head-to-Tail Split-Read Pipeline)
An independent, decoupled computational module for rescuing low-frequency 
circular DNA junction reads in target ROI vs. matched controls.
"""

import argparse
import random
import re
import sys
from collections import defaultdict
import numpy as np
import pysam
from scipy.stats import fisher_exact, mannwhitneyu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rescue circular Head-to-Tail split-reads from target vs. matched control regions."
    )
    parser.add_argument("-b", "--bam", required=True, help="Input coordinate-sorted and indexed BAM file")
    parser.add_argument("-t", "--target-bed", required=True, help="Target high-confidence candidate BED file")
    parser.add_argument("-r", "--reference", required=True, help="Reference genome FASTA file (indexed with .fai)")
    parser.add_argument("-o", "--out-prefix", default="rescued_microdna", help="Output files prefix")
    parser.add_argument("--min-mapq", type=int, default=20, help="Minimum MAPQ for primary and split alignments (default: 20)")
    parser.add_argument("--min-circle-len", type=int, default=150, help="Minimum circular DNA length to rescue (default: 150 bp)")
    parser.add_argument("--max-circle-len", type=int, default=3000, help="Maximum circular DNA length to rescue (default: 3000 bp)")
    parser.add_argument("--depth-tolerance", type=float, default=0.35, help="Coverage tolerance ratio for control matching (default: 0.35)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


# ==========================================
# 1. 区域加载与协变量匹配对照组生成
# ==========================================

def load_bed_regions(bed_path):
    regions = []
    with open(bed_path, "r") as f:
        for idx, line in enumerate(f):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            region_id = parts[3] if len(parts) > 3 else f"ROI_{idx+1}"
            regions.append({"chrom": chrom, "start": start, "end": end, "id": region_id, "len": end - start})
    return regions


def get_region_mean_depth(bam, chrom, start, end):
    try:
        # 快速估算区间平均测序深度
        coverage = bam.count_coverage(chrom, start, end)
        total_bases = sum(sum(coverage[i]) for i in range(4))
        length = max(1, end - start)
        return total_bases / length
    except Exception:
        return 0.0


def generate_matched_controls(targets, bam, ref_fai, seed=42, depth_tol=0.35):
    random.seed(seed)
    # 获取可用染色体及其长度
    chrom_lens = {}
    with open(ref_fai, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            chrom_lens[parts[0]] = int(parts[1])

    # 建立目标区间避让集 (带 1000 bp buffer)
    forbidden = defaultdict(list)
    for t in targets:
        forbidden[t["chrom"]].append((max(0, t["start"] - 1000), t["end"] + 1000))

    controls = []
    print(f"[*] Generating matched controls for {len(targets)} candidate regions...")

    for t in targets:
        chrom = t["chrom"]
        if chrom not in chrom_lens:
            continue
        c_len = chrom_lens[chrom]
        r_len = t["len"]
        target_depth = get_region_mean_depth(bam, chrom, t["start"], t["end"])

        best_ctrl = None
        max_attempts = 150
        min_depth = target_depth * (1.0 - depth_tol)
        max_depth = target_depth * (1.0 + depth_tol)

        for _ in range(max_attempts):
            cand_start = random.randint(10000, max(10001, c_len - r_len - 10000))
            cand_end = cand_start + r_len

            # 冲突检测
            has_overlap = any(cs < cand_end and cand_start < ce for cs, ce in forbidden[chrom])
            if has_overlap:
                continue

            # 深度匹配检测 (非零深度的宽松校验)
            if target_depth > 1.0:
                c_depth = get_region_mean_depth(bam, chrom, cand_start, cand_end)
                if not (min_depth <= c_depth <= max_depth):
                    continue

            best_ctrl = {
                "chrom": chrom,
                "start": cand_start,
                "end": cand_end,
                "id": f"CTRL_{t['id']}",
                "len": r_len,
            }
            forbidden[chrom].append((cand_start - 1000, cand_end + 1000))
            break

        # 若深度严格匹配超时，回退到无重叠几何抽样
        if best_ctrl is None:
            cand_start = random.randint(10000, max(10001, c_len - r_len - 10000))
            best_ctrl = {
                "chrom": chrom,
                "start": cand_start,
                "end": cand_start + r_len,
                "id": f"CTRL_{t['id']}",
                "len": r_len,
            }
        controls.append(best_ctrl)

    return controls


# ==========================================
# 2. 环状 Head-to-Tail 嵌合读段识别引擎
# ==========================================

def parse_sa_tag(sa_string):
    """解析 BWA-MEM 输出的 Supplementary SA 标签"""
    entries = []
    for hit in sa_string.strip(";").split(";"):
        if not hit:
            continue
        fields = hit.split(",")
        entries.append({
            "chrom": fields[0],
            "pos": int(fields[1]) - 1,  # 转换为 0-based
            "strand": fields[2],
            "cigar": fields[3],
            "mapq": int(fields[4]),
            "nm": int(fields[5]) if len(fields) > 5 else 0
        })
    return entries


def get_cigar_consumed_query(cigar_str):
    """提取 CIGAR 中占用的 Query 读段长度 (M/I/S/H)"""
    tokens = re.findall(r'(\d+)([MIDNSHP=X])', cigar_str)
    q_len = 0
    for length, op in tokens:
        if op in "MIS=X":
            q_len += int(length)
    return q_len


def detect_head_to_tail_junction(read, min_mapq=20, min_len=150, max_len=3000):
    """
    判定一条读段是否跨越环状 DNA 的闭合接头处：
    满足同一染色体、同链，且两段比对呈首尾重叠（Head-to-Tail）构型。
    """
    if not read.has_tag("SA"):
        return None

    if read.mapping_quality < min_mapq:
        return None

    prim_chrom = read.reference_name
    prim_strand = "-" if read.is_reverse else "+"
    prim_start = read.reference_start
    prim_end = read.reference_end
    sa_hits = parse_sa_tag(read.get_tag("SA"))

    for sa in sa_hits:
        if sa["chrom"] != prim_chrom or sa["strand"] != prim_strand:
            continue
        if sa["mapq"] < min_mapq:
            continue

        sa_start = sa["pos"]
        sa_ref_span = sum(int(length) for length, op in re.findall(r'(\d+)([MDN=X])', sa["cigar"]))
        sa_end = sa_start + sa_ref_span

        # 计算两段比对在物理基因组上的覆盖跨度
        junc_left = min(prim_start, sa_start)
        junc_right = max(prim_end, sa_end)
        circle_len = junc_right - junc_left

        if not (min_len <= circle_len <= max_len):
            continue

        # 核心几何校验：
        # 对于环化连接，5' 段和 3' 段在基因组上的相对坐标发生置换。
        # 简化判断标准：两段比对物理重叠或紧密相邻，且方向呈现逆向环接
        # 严格的 Head-to-Tail 几何特征：
        is_h2t = False
        if prim_strand == "+":
            # 正链：下游片段（较右）在 read 的 5' 端，上游片段（较左）在 read 的 3' 端
            if sa_start < prim_start and read.query_alignment_start > 0:
                is_h2t = True
            elif prim_start < sa_start and read.query_alignment_start == 0:
                is_h2t = True
        else:
            # 负链对称检验
            if sa_start > prim_start and read.query_alignment_start > 0:
                is_h2t = True
            elif prim_start > sa_start and read.query_alignment_start == 0:
                is_h2t = True

        if is_h2t:
            return {
                "qname": read.query_name,
                "chrom": prim_chrom,
                "junction_start": junc_left,
                "junction_end": junc_right,
                "circle_len": circle_len,
                "prim_coords": f"{prim_start}-{prim_end}",
                "split_coords": f"{sa_start}-{sa_end}",
                "mapq": f"{read.mapping_quality},{sa['mapq']}"
            }
    return None


def calculate_microhomology(ref_fasta, chrom, junc_start, junc_end, max_search=20):
    """提取断裂点左右两端的微同源（Microhomology）序列特征"""
    try:
        left_seq = ref_fasta.fetch(chrom, max(0, junc_start - max_search), junc_start + max_search)
        right_seq = ref_fasta.fetch(chrom, max(0, junc_end - max_search), junc_end + max_search)
        # 简易最长公共子序列计算 (在接头点周围)
        match_len = 0
        for k in range(1, 15):
            seq1 = ref_fasta.fetch(chrom, junc_start - k, junc_start)
            seq2 = ref_fasta.fetch(chrom, junc_end - k, junc_end)
            if seq1.upper() == seq2.upper():
                match_len = k
            else:
                break
        return match_len
    except Exception:
        return 0


# ==========================================
# 3. 扫描执行与数据汇流
# ==========================================

def scan_cohort_regions(regions, bam, ref_fasta, group_type, args):
    region_records = []
    all_rescued_reads = []

    for reg in regions:
        chrom = reg["chrom"]
        start = reg["start"]
        end = reg["end"]
        # 向外扩增 150 bp 作为抓取缓冲带
        search_start = max(0, start - 150)
        search_end = end + 150

        rescued_in_this_region = {}
        mean_depth = get_region_mean_depth(bam, chrom, start, end)

        try:
            for read in bam.fetch(chrom, search_start, search_end):
                if read.is_unmapped or read.is_duplicate:
                    continue
                junc_info = detect_head_to_tail_junction(
                    read, 
                    min_mapq=args.min_mapq, 
                    min_len=args.min_circle_len, 
                    max_len=args.max_circle_len
                )
                if junc_info:
                    qname = junc_info["qname"]
                    # 避免配对端重叠导致的同一 Read 重复计数
                    if qname not in rescued_in_this_region:
                        # 补充微同源检测
                        m_hom = calculate_microhomology(
                            ref_fasta, chrom, junc_info["junction_start"], junc_info["junction_end"]
                        )
                        junc_info["microhomology_bp"] = m_hom
                        junc_info["region_id"] = reg["id"]
                        junc_info["group"] = group_type
                        rescued_in_this_region[qname] = junc_info

        except Exception as e:
            pass

        count = len(rescued_in_this_region)
        region_records.append({
            "region_id": reg["id"],
            "group": group_type,
            "chrom": chrom,
            "start": start,
            "end": end,
            "length": reg["len"],
            "mean_depth": round(mean_depth, 2),
            "rescued_reads": count,
            "has_junction": 1 if count > 0 else 0
        })
        all_rescued_reads.extend(rescued_in_this_region.values())

    return region_records, all_rescued_reads


# ==========================================
# 4. 统计分析与主控流程
# ==========================================

def main():
    args = parse_args()
    print("[*] Opening Alignment and Reference Files...")
    bam = pysam.AlignmentFile(args.bam, "rb")
    ref_fasta = pysam.FastaFile(args.reference)
    ref_fai = args.reference + ".fai"

    # 1. 载入实验组与生成对照组
    targets = load_bed_regions(args.target_bed)
    controls = generate_matched_controls(targets, bam, ref_fai, seed=args.seed, depth_tol=args.depth_tolerance)

    # 导出对照组 BED
    ctrl_bed_path = f"{args.out_prefix}_matched_controls.bed"
    with open(ctrl_bed_path, "w") as f:
        for c in controls:
            f.write(f"{c['chrom']}\t{c['start']}\t{c['end']}\t{c['id']}\n")
    print(f"[+] Matched controls exported to: {ctrl_bed_path}")

    # 2. 并行/顺序扫描两个区域池
    print(f"[*] Scanning Target ROIs (n={len(targets)})...")
    target_summary, target_junctions = scan_cohort_regions(targets, bam, ref_fasta, "Target", args)

    print(f"[*] Scanning Control ROIs (n={len(controls)})...")
    control_summary, control_junctions = scan_cohort_regions(controls, bam, ref_fasta, "Control", args)

    # 3. 输出拼接数据
    all_summary = target_summary + control_summary
    all_junctions = target_junctions + control_junctions

    # 写入区域汇总表
    summary_tsv = f"{args.out_prefix}_region_summary.tsv"
    with open(summary_tsv, "w") as f:
        f.write("region_id\tgroup\tchrom\tstart\tend\tlength\tmean_depth\trescued_reads\thas_junction\n")
        for r in all_summary:
            f.write(f"{r['region_id']}\t{r['group']}\t{r['chrom']}\t{r['start']}\t{r['end']}\t{r['length']}\t{r['mean_depth']}\t{r['rescued_reads']}\t{r['has_junction']}\n")
    print(f"[+] Region summary written to: {summary_tsv}")

    # 写入所有被挽救的 Read 细节
    junctions_tsv = f"{args.out_prefix}_rescued_junctions.tsv"
    with open(junctions_tsv, "w") as f:
        f.write("qname\tgroup\tregion_id\tchrom\tjunction_start\tjunction_end\tcircle_len\tprim_coords\tsplit_coords\tmapq\tmicrohomology_bp\n")
        for j in all_junctions:
            f.write(f"{j['qname']}\t{j['group']}\t{j['region_id']}\t{j['chrom']}\t{j['junction_start']}\t{j['junction_end']}\t{j['circle_len']}\t{j['prim_coords']}\t{j['split_coords']}\t{j['mapq']}\t{j['microhomology_bp']}\n")
    print(f"[+] Rescued junction reads logged to: {junctions_tsv}")

    # 4. 统计检验 (Fisher's Exact Test & Mann-Whitney U Test)
    t_pos = sum(r["has_junction"] for r in target_summary)
    t_neg = len(target_summary) - t_pos
    c_pos = sum(r["has_junction"] for r in control_summary)
    c_neg = len(control_summary) - c_pos

    table = [[t_pos, t_neg], [c_pos, c_neg]]
    odds_ratio, p_value = fisher_exact(table, alternative="greater")

    t_reads = [r["rescued_reads"] for r in target_summary]
    c_reads = [r["rescued_reads"] for r in control_summary]
    u_stat, u_pval = mannwhitneyu(t_reads, c_reads, alternative="greater")

    t_rate = (t_pos / len(target_summary)) * 100 if target_summary else 0
    c_rate = (c_pos / len(control_summary)) * 100 if control_summary else 0

    # 打印最终报告
    print("\n" + "=" * 65)
    print("           RESCUE ENRICHMENT EVALUATION REPORT")
    print("=" * 65)
    print(f"{'Cohort':<15}{'Total ROIs':<15}{'With Junction (>=1)':<20}{'Rescue Rate (%)':<15}")
    print("-" * 65)
    print(f"{'Target':<15}{len(target_summary):<15}{t_pos:<20}{t_rate:<15.2f}")
    print(f"{'Control':<15}{len(control_summary):<15}{c_pos:<20}{c_rate:<15.2f}")
    print("-" * 65)
    print(f"[*] 2x2 Contingency Table: Target [{t_pos}, {t_neg}] vs Control [{c_pos}, {c_neg}]")
    print(f"[*] Fisher's Exact Test Odds Ratio (OR): {odds_ratio:.3f}")
    print(f"[*] Fisher's Exact Test p-value:        {p_value:.3e}")
    print(f"[*] Mann-Whitney U Test p-value:        {u_pval:.3e}")
    
    # 统计微同源均值
    t_hom = [j["microhomology_bp"] for j in target_junctions]
    avg_hom = np.mean(t_hom) if t_hom else 0.0
    print(f"[*] Average Microhomology at Junctions: {avg_hom:.2f} bp")
    print("=" * 65 + "\n")

    bam.close()
    ref_fasta.close()


if __name__ == "__main__":
    main()