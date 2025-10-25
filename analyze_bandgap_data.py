#!/usr/bin/env python3
"""
分析2D材料带隙数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_bandgap_data():
    """
    分析2D材料带隙数据
    """
    # 读取数据
    df = pd.read_csv('2d_materials_bandgap_simple.csv')
    
    print("=== 2D Materials Bandgap Data Analysis ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Total materials: {len(df)}")
    
    # 基本统计
    print(f"\nBasic Statistics:")
    print(f"Metals (bandgap = 0): {sum(df['bandgap_eV'] == 0.0)} ({sum(df['bandgap_eV'] == 0.0)/len(df)*100:.1f}%)")
    print(f"Semiconductors/Insulators (bandgap > 0): {sum(df['bandgap_eV'] > 0.0)} ({sum(df['bandgap_eV'] > 0.0)/len(df)*100:.1f}%)")
    print(f"Average bandgap: {df['bandgap_eV'].mean():.3f} eV")
    print(f"Median bandgap: {df['bandgap_eV'].median():.3f} eV")
    print(f"Bandgap range: {df['bandgap_eV'].min():.3f} - {df['bandgap_eV'].max():.3f} eV")
    
    # 按带隙范围分类
    print(f"\nBandgap Distribution:")
    metals = sum(df['bandgap_eV'] == 0.0)
    small_gap = sum((df['bandgap_eV'] > 0.0) & (df['bandgap_eV'] <= 0.5))
    medium_gap = sum((df['bandgap_eV'] > 0.5) & (df['bandgap_eV'] <= 2.0))
    large_gap = sum(df['bandgap_eV'] > 2.0)
    
    print(f"  Metals (0.0 eV): {metals} ({metals/len(df)*100:.1f}%)")
    print(f"  Small gap (0.0 < gap ≤ 0.5 eV): {small_gap} ({small_gap/len(df)*100:.1f}%)")
    print(f"  Medium gap (0.5 < gap ≤ 2.0 eV): {medium_gap} ({medium_gap/len(df)*100:.1f}%)")
    print(f"  Large gap (gap > 2.0 eV): {large_gap} ({large_gap/len(df)*100:.1f}%)")
    
    # 直接/间接带隙统计
    semiconductors = df[df['bandgap_eV'] > 0]
    direct_gap = sum(semiconductors['is_gap_direct'] == True)
    indirect_gap = sum(semiconductors['is_gap_direct'] == False)
    unknown_gap = sum(semiconductors['is_gap_direct'].isna())
    
    print(f"\nGap Type (for semiconductors/insulators only):")
    print(f"  Direct gap: {direct_gap} ({direct_gap/len(semiconductors)*100:.1f}%)")
    print(f"  Indirect gap: {indirect_gap} ({indirect_gap/len(semiconductors)*100:.1f}%)")
    print(f"  Unknown: {unknown_gap} ({unknown_gap/len(semiconductors)*100:.1f}%)")
    
    # 元素数量分布
    print(f"\nElement Count Distribution:")
    element_counts = df['nelements'].value_counts().sort_index()
    for n_elem, count in element_counts.items():
        print(f"  {n_elem} elements: {count} materials ({count/len(df)*100:.1f}%)")
    
    # 空间群统计
    print(f"\nTop 10 Space Groups:")
    top_sg = df['space_group_number'].value_counts().head(10)
    for sg, count in top_sg.items():
        sg_symbol = df[df['space_group_number'] == sg]['space_group_symbol'].iloc[0]
        print(f"  SG {sg} ({sg_symbol}): {count} materials ({count/len(df)*100:.1f}%)")
    
    # 最常见的化学系统
    print(f"\nTop 10 Chemical Systems:")
    top_chemsys = df['chemsys'].value_counts().head(10)
    for chemsys, count in top_chemsys.items():
        print(f"  {chemsys}: {count} materials ({count/len(df)*100:.1f}%)")
    
    # 创建可视化图表
    create_visualizations(df)
    
    return df

def create_visualizations(df):
    """
    创建数据可视化图表
    """
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('2D Materials Bandgap Analysis', fontsize=16, fontweight='bold')
    
    # 1. 带隙分布直方图
    ax1 = axes[0, 0]
    bins = np.linspace(0, df['bandgap_eV'].max(), 50)
    ax1.hist(df['bandgap_eV'], bins=bins, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Bandgap (eV)')
    ax1.set_ylabel('Count')
    ax1.set_title('Bandgap Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. 金属vs半导体饼图
    ax2 = axes[0, 1]
    metals = sum(df['bandgap_eV'] == 0.0)
    semiconductors = sum(df['bandgap_eV'] > 0.0)
    labels = ['Metals', 'Semiconductors/Insulators']
    sizes = [metals, semiconductors]
    colors = ['#ff9999', '#66b3ff']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Metal vs Semiconductor Distribution')
    
    # 3. 元素数量分布
    ax3 = axes[0, 2]
    element_counts = df['nelements'].value_counts().sort_index()
    ax3.bar(element_counts.index, element_counts.values, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Number of Elements')
    ax3.set_ylabel('Count')
    ax3.set_title('Element Count Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. 带隙vs晶格体积散点图
    ax4 = axes[1, 0]
    metals_mask = df['bandgap_eV'] == 0.0
    semiconductors_mask = df['bandgap_eV'] > 0.0
    
    ax4.scatter(df[metals_mask]['lattice_volume'], df[metals_mask]['bandgap_eV'], 
                alpha=0.5, c='red', s=20, label='Metals')
    ax4.scatter(df[semiconductors_mask]['lattice_volume'], df[semiconductors_mask]['bandgap_eV'], 
                alpha=0.5, c='blue', s=20, label='Semiconductors')
    ax4.set_xlabel('Lattice Volume (Ų)')
    ax4.set_ylabel('Bandgap (eV)')
    ax4.set_title('Bandgap vs Lattice Volume')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 密度vs带隙散点图
    ax5 = axes[1, 1]
    ax5.scatter(df[metals_mask]['density'], df[metals_mask]['bandgap_eV'], 
                alpha=0.5, c='red', s=20, label='Metals')
    ax5.scatter(df[semiconductors_mask]['density'], df[semiconductors_mask]['bandgap_eV'], 
                alpha=0.5, c='blue', s=20, label='Semiconductors')
    ax5.set_xlabel('Density (g/cm³)')
    ax5.set_ylabel('Bandgap (eV)')
    ax5.set_title('Bandgap vs Density')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 空间群分布（前10个）
    ax6 = axes[1, 2]
    top_sg = df['space_group_number'].value_counts().head(10)
    ax6.bar(range(len(top_sg)), top_sg.values, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Space Group Number')
    ax6.set_ylabel('Count')
    ax6.set_title('Top 10 Space Groups')
    ax6.set_xticks(range(len(top_sg)))
    ax6.set_xticklabels(top_sg.index, rotation=45)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2d_materials_bandgap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as: 2d_materials_bandgap_analysis.png")

def main():
    """
    主函数
    """
    df = analyze_bandgap_data()
    
    print(f"\n=== Analysis Complete ===")
    print(f"Data saved in: 2d_materials_bandgap_simple.csv")
    print(f"Analysis plot saved in: 2d_materials_bandgap_analysis.png")

if __name__ == "__main__":
    main()

