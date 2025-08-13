import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import io
from datetime import datetime

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(
    page_title="云模型综合评价系统",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "逆向云发生器"
if 'forward_cloud_drops' not in st.session_state:
    st.session_state.forward_cloud_drops = None
if 'forward_memberships' not in st.session_state:
    st.session_state.forward_memberships = None
if 'expert_scores' not in st.session_state:
    st.session_state.expert_scores = None
if 'indicator_weights' not in st.session_state:
    st.session_state.indicator_weights = None
if 'indicator_clouds' not in st.session_state:
    st.session_state.indicator_clouds = None
if 'comprehensive_cloud' not in st.session_state:
    st.session_state.comprehensive_cloud = None

# 记忆功能 - 逆向云发生器数据
if 'reverse_data_text' not in st.session_state:
    st.session_state.reverse_data_text = ""
if 'reverse_weight_text' not in st.session_state:
    st.session_state.reverse_weight_text = ""
if 'reverse_input_method' not in st.session_state:
    st.session_state.reverse_input_method = "手动输入"
if 'reverse_weight_method' not in st.session_state:
    st.session_state.reverse_weight_method = "等权重"

# 记忆功能 - 正向云发生器数据
if 'forward_ex' not in st.session_state:
    st.session_state.forward_ex = 50.0
if 'forward_en' not in st.session_state:
    st.session_state.forward_en = 8.33
if 'forward_he' not in st.session_state:
    st.session_state.forward_he = 0.1
if 'forward_num_drops' not in st.session_state:
    st.session_state.forward_num_drops = 1000
if 'forward_preset' not in st.session_state:
    st.session_state.forward_preset = '自定义'
if 'standard_clouds_data' not in st.session_state:
    st.session_state.standard_clouds_data = pd.DataFrame({
        '云名称': ['劣', '差', '一般', '良', '优', '综合评价云'],
        'Ex': [12.5, 37.5, 62.5, 82.5, 95.0, np.nan],
        'En': [4.17, 4.17, 4.17, 2.5, 1.67, np.nan],
        'He': [0.5, 0.5, 0.5, 0.5, 0.5, np.nan],
        '云滴数量': [1200, 1200, 1200, 1200, 1200, 1200],
        '颜色': ['red', 'blue', 'yellow', 'gray', 'orange', 'green'],
        '绘图符号': ['o', '*', '*', '*', 'o', 's']
    })

def generate_cloud_drops(ex, en, he, num_drops=1000):
    """生成云滴"""
    cloud_drops = []
    memberships = []
    
    for _ in range(num_drops):
        # 生成正态随机数
        en_prime = np.random.normal(en, he)
        x = np.random.normal(ex, abs(en_prime))
        
        # 计算隶属度
        membership = np.exp(-0.5 * ((x - ex) / en) ** 2)
        
        cloud_drops.append(x)
        memberships.append(membership)
    
    return np.array(cloud_drops), np.array(memberships)

def calculate_reverse_cloud_params(data):
    """计算逆向云模型参数"""
    data = np.array(data)
    n = len(data)
    
    # 计算期望值 Ex
    ex = np.mean(data)
    
    # 计算一阶样本绝对中心矩
    s1 = np.mean(np.abs(data - ex))
    
    # 计算样本方差
    s2 = np.var(data, ddof=1)
    
    # 计算熵 En
    en = np.sqrt(np.pi / 2) * s1
    
    # 计算超熵 He
    he = np.sqrt(abs(s2 - en**2))
    
    return ex, en, he

def calculate_indicator_clouds(expert_scores, weights):
    """计算指标评价云"""
    indicator_clouds = []
    
    for i, scores in enumerate(expert_scores.T):  # 按指标遍历
        # 计算该指标的云模型参数
        ex, en, he = calculate_reverse_cloud_params(scores)
        indicator_clouds.append({
            '指标': f'指标{i+1}',
            'Ex': ex,
            'En': en,
            'He': he,
            '权重': weights[i] if i < len(weights) else 0
        })
    
    return indicator_clouds

def calculate_comprehensive_cloud(indicator_clouds):
    """计算综合评价云"""
    # 提取参数和权重
    exs = np.array([cloud['Ex'] for cloud in indicator_clouds])
    ens = np.array([cloud['En'] for cloud in indicator_clouds])
    hes = np.array([cloud['He'] for cloud in indicator_clouds])
    weights = np.array([cloud['权重'] for cloud in indicator_clouds])
    
    # 权重归一化
    weights = weights / np.sum(weights)
    
    # 计算综合云参数
    ex_comp = np.sum(weights * exs)
    en_comp = np.sqrt(np.sum(weights * (ens**2 + (exs - ex_comp)**2)))
    he_comp = np.sqrt(np.sum(weights * hes**2))
    
    return ex_comp, en_comp, he_comp

def plot_scatter(cloud_drops, memberships, title="云滴散点图", xlabel="云滴值", ylabel="隶属度"):
    """绘制散点图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(cloud_drops, memberships, alpha=0.6, c=memberships, cmap='viridis')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='隶属度')
    return fig

def plot_histogram(cloud_drops, memberships, title="云滴分布直方图", xlabel="云滴值", ylabel="频数"):
    """绘制直方图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(cloud_drops, bins=50, alpha=0.7, edgecolor='black')
    
    # 根据隶属度着色
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for i, (patch, center) in enumerate(zip(patches, bin_centers)):
        # 找到最接近的云滴点来确定颜色
        closest_idx = np.argmin(np.abs(cloud_drops - center))
        color_intensity = memberships[closest_idx]
        patch.set_facecolor(plt.cm.viridis(color_intensity))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig

def plot_cloud_visualization(ex, en, he, cloud_drops, memberships, title="云模型可视化", xlabel="云滴值", ylabel="隶属度"):
    """绘制云模型可视化图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制实际云滴
    ax.scatter(cloud_drops, memberships, alpha=0.6, c='blue', s=20, label='实际云滴')
    
    # 绘制理论云模型曲线
    x_theory = np.linspace(cloud_drops.min(), cloud_drops.max(), 1000)
    y_theory = np.exp(-0.5 * ((x_theory - ex) / en) ** 2)
    ax.plot(x_theory, y_theory, 'r-', linewidth=2, label='理论云模型')
    
    # 标记特征点
    ax.axvline(x=ex, color='green', linestyle='--', alpha=0.7, label=f'Ex = {ex:.2f}')
    ax.axvline(x=ex-en, color='orange', linestyle='--', alpha=0.7, label=f'Ex-En = {ex-en:.2f}')
    ax.axvline(x=ex+en, color='orange', linestyle='--', alpha=0.7, label=f'Ex+En = {ex+en:.2f}')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_combined_visualization(ex, en, he, cloud_drops, memberships, title="组合可视化图", xlabel="云滴值", ylabel="隶属度/频数"):
    """绘制组合图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 上图：散点图
    scatter = ax1.scatter(cloud_drops, memberships, alpha=0.6, c=memberships, cmap='viridis')
    ax1.set_ylabel('隶属度')
    ax1.set_title(f'{title} - 散点图')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1)
    
    # 下图：直方图
    n, bins, patches = ax2.hist(cloud_drops, bins=50, alpha=0.7, edgecolor='black')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for i, (patch, center) in enumerate(zip(patches, bin_centers)):
        closest_idx = np.argmin(np.abs(cloud_drops - center))
        color_intensity = memberships[closest_idx]
        patch.set_facecolor(plt.cm.viridis(color_intensity))
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('频数')
    ax2.set_title(f'{title} - 直方图')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_standard_clouds(standard_data, title="评价标准云图", xlabel="评分值", ylabel="隶属度"):
    """绘制评价标准云图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for _, row in standard_data.iterrows():
        ex, en, he = row['Ex'], row['En'], row['He']
        
        # 检查数据有效性，如果Ex、En、He任一为空或无效，则跳过该行
        if pd.isna(ex) or pd.isna(en) or pd.isna(he) or ex == 0 or en == 0:
            continue
            
        # 处理云滴数量，防止NaN值
        try:
            num_drops = int(row['云滴数量']) if pd.notna(row['云滴数量']) else 1200
        except (ValueError, TypeError):
            num_drops = 1200  # 默认值
            
        color = row['颜色'] if pd.notna(row['颜色']) else 'blue'
        marker = row['绘图符号'] if pd.notna(row['绘图符号']) else 'o'
        name = row['云名称'] if pd.notna(row['云名称']) else '未命名'
        
        # 生成云滴
        drops, memberships = generate_cloud_drops(ex, en, he, num_drops)
        
        # 绘制散点
        ax.scatter(drops, memberships, alpha=0.6, c=color, marker=marker, s=20, label=name)
        
        # 绘制理论曲线
        x_theory = np.linspace(drops.min(), drops.max(), 200)
        y_theory = np.exp(-0.5 * ((x_theory - ex) / en) ** 2)
        ax.plot(x_theory, y_theory, color=color, linewidth=2, alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_comprehensive_with_standards(comprehensive_cloud, standard_data, num_drops=1000, title="综合评价云与标准云对比图", xlabel="评分值", ylabel="隶属度"):
    """绘制综合评价云与标准评价云对比图"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 绘制标准评价云
    for _, row in standard_data.iterrows():
        ex, en, he = row['Ex'], row['En'], row['He']
        
        # 检查数据有效性，如果Ex、En、He任一为空或无效，则跳过该行
        if pd.isna(ex) or pd.isna(en) or pd.isna(he) or ex == 0 or en == 0:
            continue
            
        # 处理云滴数量，防止NaN值
        try:
            std_num_drops = int(row['云滴数量']) if pd.notna(row['云滴数量']) else 1200
        except (ValueError, TypeError):
            std_num_drops = 1200  # 默认值
            
        color = row['颜色'] if pd.notna(row['颜色']) else 'blue'
        marker = row['绘图符号'] if pd.notna(row['绘图符号']) else 'o'
        name = row['云名称'] if pd.notna(row['云名称']) else '未命名'
        
        # 生成标准云滴
        drops, memberships = generate_cloud_drops(ex, en, he, std_num_drops)
        
        # 绘制标准云散点
        ax.scatter(drops, memberships, alpha=0.4, c=color, marker=marker, s=15, label=f'标准-{name}')
        
        # 绘制标准云理论曲线
        x_theory = np.linspace(0, 100, 200)
        y_theory = np.exp(-0.5 * ((x_theory - ex) / en) ** 2)
        ax.plot(x_theory, y_theory, color=color, linewidth=1.5, alpha=0.6, linestyle='--')
    
    # 绘制综合评价云
    comp_ex = comprehensive_cloud['Ex']
    comp_en = comprehensive_cloud['En']
    comp_he = comprehensive_cloud['He']
    
    # 生成综合评价云滴
    comp_drops, comp_memberships = generate_cloud_drops(comp_ex, comp_en, comp_he, num_drops)
    
    # 绘制综合评价云散点（突出显示）
    ax.scatter(comp_drops, comp_memberships, alpha=0.8, c='black', marker='D', s=30, label='综合评价云', edgecolors='white', linewidth=0.5)
    
    # 绘制综合评价云理论曲线（突出显示）
    x_theory_comp = np.linspace(0, 100, 200)
    y_theory_comp = np.exp(-0.5 * ((x_theory_comp - comp_ex) / comp_en) ** 2)
    ax.plot(x_theory_comp, y_theory_comp, color='black', linewidth=3, alpha=0.9, label='综合评价云理论曲线')
    
    # 标记综合评价云的特征点
    ax.axvline(x=comp_ex, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'综合Ex = {comp_ex:.2f}')
    ax.axvline(x=comp_ex-comp_en, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.axvline(x=comp_ex+comp_en, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    return fig

def plot_comprehensive_with_standards_button():
    """处理标准对比图按钮的绘制逻辑"""
    if st.session_state.comprehensive_cloud is not None:
        num_drops = st.number_input("云滴数量", value=1000, min_value=100, max_value=5000, step=100, key="std_compare_drops")
        fig = plot_comprehensive_with_standards(
            st.session_state.comprehensive_cloud, 
            st.session_state.standard_clouds_data, 
            num_drops,
            "综合评价云与标准云对比图", 
            "评价值", 
            "隶属度"
        )
        st.pyplot(fig)
        
        # 添加评价结果分析
        st.markdown("**评价结果分析：**")
        comp_ex = st.session_state.comprehensive_cloud['Ex']
        
        # 判断综合评价结果属于哪个等级
        if comp_ex < 25:
            level = "劣"
            color = "🔴"
        elif comp_ex < 50:
            level = "差"
            color = "🟠"
        elif comp_ex < 75:
            level = "一般"
            color = "🟡"
        elif comp_ex < 90:
            level = "良"
            color = "🟢"
        else:
            level = "优"
            color = "🟢"
        
        st.info(f"{color} 综合评价结果：**{level}** (评分值: {comp_ex:.2f})")
    else:
        st.warning("请先生成综合评价云")

def main():
    st.title("☁️ 云模型综合评价系统")
    
    # 侧边栏导航
    with st.sidebar:
        st.header("功能选择")
        
        # 逆向云发生器按钮（放在上面）
        if st.button("🔙 逆向云发生器", use_container_width=True,
                    type="primary" if st.session_state.current_page == "逆向云发生器" else "secondary"):
            st.session_state.current_page = "逆向云发生器"
            st.rerun()
        
        # 正向云发生器按钮
        if st.button("🔄 正向云发生器", use_container_width=True, 
                    type="primary" if st.session_state.current_page == "正向云发生器" else "secondary"):
            st.session_state.current_page = "正向云发生器"
            st.rerun()
        
        st.divider()
        st.caption("云模型综合评价 v1.0.0")
    
    # 根据当前页面显示对应内容
    if st.session_state.current_page == "正向云发生器":
        forward_cloud_generator()
    else:
        reverse_cloud_generator()

def forward_cloud_generator():
    """正向云发生器界面"""
    st.header("🔄 正向云发生器")
    st.markdown("从云模型参数生成云滴")
    
    # 评价标准云配置
    st.subheader("📊 评价标准云配置")
    
    # 添加复制按钮
    col_title, col_copy_std = st.columns([4, 1])
    with col_copy_std:
        if st.button("📋 复制配置", key="copy_standard_table", help="复制标准云配置到剪贴板"):
            # 生成可复制的文本格式
            csv_text = st.session_state.standard_clouds_data.to_csv(index=False, sep='\t')
            st.code(csv_text, language=None)
            st.success("数据已生成，请手动复制上方文本框中的内容")
    
    # 使用data_editor来编辑标准云数据
    edited_data = st.data_editor(
        st.session_state.standard_clouds_data,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Ex": st.column_config.NumberColumn("Ex", min_value=0.0, max_value=100.0, step=0.1),
            "En": st.column_config.NumberColumn("En", min_value=0.0, max_value=50.0, step=0.01),
            "He": st.column_config.NumberColumn("He", min_value=0.0, max_value=10.0, step=0.01),
            "云滴数量": st.column_config.NumberColumn("云滴数量", min_value=100, max_value=5000, step=100),
            "颜色": st.column_config.SelectboxColumn(
                "颜色",
                options=['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black']
            ),
            "绘图符号": st.column_config.SelectboxColumn(
                "绘图符号",
                options=['o', 's', '^', 'v', '<', '>', 'd', 'p', '*', 'h', 'H', '+', 'x', 'D']
            )
        }
    )
    
    # 更新session state
    st.session_state.standard_clouds_data = edited_data
    
    # 主要内容区域
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ 参数设置")
        
        # 预设参数选择
        preset = st.selectbox(
            "预设参数",
            ['自定义', '劣', '差', '一般', '良', '优', '低变异', '中变异', '高变异', '极高变异'],
            index=['自定义', '劣', '差', '一般', '良', '优', '低变异', '中变异', '高变异', '极高变异'].index(st.session_state.forward_preset)
        )
        st.session_state.forward_preset = preset
        
        # 根据预设设置默认值
        preset_params = {
            '劣': (12.5, 4.17, 0.5),
            '差': (37.5, 4.17, 0.5),
            '一般': (62.5, 4.17, 0.5),
            '良': (82.5, 2.5, 0.5),
            '优': (95.0, 1.67, 0.5),
            '低变异': (50, 5, 0.1),
            '中变异': (50, 8, 0.5),
            '高变异': (50, 12, 1.0),
            '极高变异': (50, 15, 2.0)
        }
        
        if preset != '自定义' and preset in preset_params:
            default_ex, default_en, default_he = preset_params[preset]
        else:
            default_ex, default_en, default_he = 50, 8.33, 0.1
        
        # 参数输入
        if preset != '自定义' and preset in preset_params:
            ex = st.number_input("期望值 Ex", value=float(default_ex), step=0.1)
            en = st.number_input("熵 En", value=float(default_en), step=0.01)
            he = st.number_input("超熵 He", value=float(default_he), step=0.01)
        else:
            ex = st.number_input("期望值 Ex", value=float(st.session_state.forward_ex), step=0.1)
            en = st.number_input("熵 En", value=float(st.session_state.forward_en), step=0.01)
            he = st.number_input("超熵 He", value=float(st.session_state.forward_he), step=0.01)
        
        num_drops = st.number_input("云滴数量", value=st.session_state.forward_num_drops, min_value=100, max_value=10000, step=100, format="%d")
        
        # 保存参数到session state
        st.session_state.forward_ex = ex
        st.session_state.forward_en = en
        st.session_state.forward_he = he
        st.session_state.forward_num_drops = num_drops
        
        enhanced_mode = st.checkbox("增强模式（计算隶属度）", value=True)
        
        # 生成按钮
        if st.button("🎯 生成云滴", type="primary"):
            if num_drops > 0:
                cloud_drops, memberships = generate_cloud_drops(ex, en, he, num_drops)
                st.session_state.forward_cloud_drops = cloud_drops
                st.session_state.forward_memberships = memberships
                st.success(f"成功生成 {num_drops} 个云滴！")
            else:
                st.error("云滴数量必须大于0")
        
        # 操作按钮
        st.subheader("📋 操作")
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("📤 导出数据"):
                if st.session_state.forward_cloud_drops is not None:
                    df = pd.DataFrame({
                        '云滴值': st.session_state.forward_cloud_drops,
                        '隶属度': st.session_state.forward_memberships
                    })
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="下载CSV文件",
                        data=csv,
                        file_name=f"cloud_drops_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("请先生成云滴数据")
        
        with col_btn2:
            if st.button("🗑️ 清空结果"):
                st.session_state.forward_cloud_drops = None
                st.session_state.forward_memberships = None
                st.success("结果已清空")
    
    with col2:
        st.subheader("📈 结果显示")
        
        if st.session_state.forward_cloud_drops is not None:
            # 统计信息
            cloud_drops = st.session_state.forward_cloud_drops
            memberships = st.session_state.forward_memberships
            
            st.markdown("**统计信息：**")
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                st.metric("云滴数量", len(cloud_drops))
            with stats_col2:
                st.metric("平均值", f"{np.mean(cloud_drops):.2f}")
            with stats_col3:
                st.metric("标准差", f"{np.std(cloud_drops):.2f}")
            with stats_col4:
                st.metric("平均隶属度", f"{np.mean(memberships):.3f}")
            
            # 数据表格（前100个）
            st.markdown("**云滴数据（前100个）：**")
            display_data = pd.DataFrame({
                '云滴值': cloud_drops[:100],
                '隶属度': memberships[:100]
            })
            
            # 添加复制按钮
            col_table, col_copy = st.columns([4, 1])
            with col_table:
                st.dataframe(display_data, use_container_width=True)
            with col_copy:
                st.markdown("<br>", unsafe_allow_html=True)  # 添加间距
                if st.button("📋 复制表格", key="copy_forward_table", help="复制云滴数据到剪贴板"):
                    # 生成可复制的文本格式
                    csv_text = display_data.to_csv(index=False, sep='\t')
                    st.code(csv_text, language=None)
                    st.success("数据已生成，请手动复制上方文本框中的内容")
        else:
            st.info("请先生成云滴数据以查看结果")
    
    # 可视化部分
    if st.session_state.forward_cloud_drops is not None:
        st.subheader("📊 数据可视化")
        
        # 图表标签自定义
        with st.expander("🎨 自定义图表标签"):
            custom_title = st.text_input("图表标题", value="云模型可视化")
            custom_xlabel = st.text_input("X轴标签", value="云滴值")
            custom_ylabel = st.text_input("Y轴标签", value="隶属度")
        
        # 可视化选项
        viz_option = st.selectbox(
            "选择可视化类型",
            ["散点图", "直方图", "云模型图", "组合图"]
        )
        
        cloud_drops = st.session_state.forward_cloud_drops
        memberships = st.session_state.forward_memberships
        
        if viz_option == "散点图":
            fig = plot_scatter(cloud_drops, memberships, custom_title, custom_xlabel, custom_ylabel)
            st.pyplot(fig)
        elif viz_option == "直方图":
            fig = plot_histogram(cloud_drops, memberships, custom_title, custom_xlabel, "频数")
            st.pyplot(fig)
        elif viz_option == "云模型图":
            fig = plot_cloud_visualization(ex, en, he, cloud_drops, memberships, custom_title, custom_xlabel, custom_ylabel)
            st.pyplot(fig)
        elif viz_option == "组合图":
            fig = plot_combined_visualization(ex, en, he, cloud_drops, memberships, custom_title, custom_xlabel, custom_ylabel)
            st.pyplot(fig)
    
    # 评价标准云图
    st.subheader("🌟 评价标准云图")
    with st.expander("🎨 自定义标准云图标签"):
        std_title = st.text_input("标准云图标题", value="评价标准云图")
        std_xlabel = st.text_input("标准云图X轴标签", value="评分值")
        std_ylabel = st.text_input("标准云图Y轴标签", value="隶属度")
    
    if st.button("📊 绘制评价标准云图"):
        fig = plot_standard_clouds(st.session_state.standard_clouds_data, std_title, std_xlabel, std_ylabel)
        st.pyplot(fig)

def reverse_cloud_generator():
    """逆向云发生器界面 - 综合评价云生成"""
    st.header("🔙 逆向云发生器 - 综合评价云生成")
    st.markdown("通过专家打分和权重生成指标评价云，进而生成综合评价云")
    
    # 步骤1：专家打分数据输入
    st.subheader("📝 步骤1：专家打分数据输入")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**数据输入方式**")
        input_method = st.radio(
            "选择输入方式",
            ["手动输入", "文件上传", "示例数据"],
            index=["手动输入", "文件上传", "示例数据"].index(st.session_state.reverse_input_method)
        )
        st.session_state.reverse_input_method = input_method
        
        expert_scores = None
        
        if input_method == "手动输入":
            st.markdown("**格式说明：** 每行代表一个专家，每列代表一个指标，用逗号分隔")
            st.markdown("**提示：** 可以从Excel中复制数据直接粘贴到下方文本框")
            data_text = st.text_area(
                "输入专家打分数据（直接Ctrl+V即可）",
                value=st.session_state.reverse_data_text,
                placeholder="85,78,92,88\n82,85,89,91\n88,82,85,87\n...",
                height=150,
                help="可以从Excel表格中复制数据直接粘贴，系统会自动处理制表符和逗号分隔"
            )
            st.session_state.reverse_data_text = data_text
            
            if data_text:
                try:
                    lines = [line.strip() for line in data_text.split('\n') if line.strip()]
                    # 处理制表符分隔的数据（从Excel复制的数据通常是制表符分隔）
                    processed_lines = []
                    for line in lines:
                        if '\t' in line:  # 制表符分隔
                            processed_lines.append([float(x.strip()) for x in line.split('\t') if x.strip()])
                        else:  # 逗号分隔
                            processed_lines.append([float(x.strip()) for x in line.split(',') if x.strip()])
                    expert_scores = np.array(processed_lines)
                except ValueError:
                    st.error("请输入有效的数值，支持逗号或制表符分隔")
        
        elif input_method == "文件上传":
            uploaded_file = st.file_uploader(
                "选择文件（CSV或Excel）",
                type=['csv', 'xlsx', 'xls'],
                help="支持CSV和Excel文件格式，文件中每行代表一个专家，每列代表一个指标"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, header=None)  # 不使用第一行作为列名
                    else:
                        df = pd.read_excel(uploaded_file, header=None)  # 不使用第一行作为列名
                    expert_scores = df.values
                    st.success(f"成功读取文件：{uploaded_file.name}")
                except Exception as e:
                    st.error(f"文件读取错误：{str(e)}")
                    st.info("请确保文件格式正确，每行代表一个专家，每列代表一个指标")
        
        else:  # 示例数据
            st.info("使用示例数据：5个专家对4个指标的打分")
            expert_scores = np.array([
                [85, 78, 92, 88],
                [82, 85, 89, 91],
                [88, 82, 85, 87],
                [79, 88, 91, 85],
                [86, 80, 88, 89]
            ])
    
    with col2:
        if expert_scores is not None:
            # 标题和复制按钮
            preview_title_col, preview_copy_col = st.columns([3, 1])
            with preview_title_col:
                st.markdown("**专家打分数据预览**")
            with preview_copy_col:
                if st.button("📋 复制数据", key="copy_expert_data", help="复制专家打分数据"):
                    df_preview = pd.DataFrame(expert_scores, 
                                            columns=[f'指标{i+1}' for i in range(expert_scores.shape[1])],
                                            index=[f'专家{i+1}' for i in range(expert_scores.shape[0])])
                    # 生成可复制的文本格式
                    csv_text = df_preview.to_csv(sep='\t')
                    st.code(csv_text, language=None)
                    st.success("数据已生成，请手动复制上方文本框中的内容")
            
            df_preview = pd.DataFrame(expert_scores, 
                                    columns=[f'指标{i+1}' for i in range(expert_scores.shape[1])],
                                    index=[f'专家{i+1}' for i in range(expert_scores.shape[0])])
            st.dataframe(df_preview, use_container_width=True)
            
            st.markdown("**数据统计**")
            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.metric("专家数量", expert_scores.shape[0])
            with stat_col2:
                st.metric("指标数量", expert_scores.shape[1])
    
    # 步骤2：权重设置
    st.subheader("⚖️ 步骤2：指标权重设置")
    
    if expert_scores is not None:
        num_indicators = expert_scores.shape[1]
        
        weight_method = st.radio(
            "权重设置方式",
            ["等权重", "自定义权重"]
        )
        
        weight_input_method = st.radio(
            "权重输入方式",
            ["等权重", "手动输入权重", "上传权重文件"],
            index=["等权重", "手动输入权重", "上传权重文件"].index(st.session_state.reverse_weight_method)
        )
        st.session_state.reverse_weight_method = weight_input_method
        
        if weight_input_method == "等权重":
            weights = np.ones(num_indicators) / num_indicators
            st.info(f"所有指标权重均为：{1/num_indicators:.3f}")
        elif weight_input_method == "手动输入权重":
             st.markdown("**自定义各指标权重（系统会自动归一化）**")
             st.markdown("**提示：** 可以从Excel中复制权重数据直接粘贴到下方文本框")
             
             # 提供文本区域输入方式，支持从Excel粘贴
             weight_text = st.text_area(
                 "输入权重（直接Ctrl+V即可）",
                 value=st.session_state.reverse_weight_text,
                 placeholder="0.3,0.25,0.25,0.2\n或\n0.3\n0.25\n0.25\n0.2",
                 height=100,
                 help="支持逗号分隔、制表符分隔或换行分隔，可以从Excel表格中复制数据直接粘贴"
             )
             st.session_state.reverse_weight_text = weight_text
             
             if weight_text:
                 try:
                     # 处理多种分隔符格式
                     weight_text = weight_text.strip()
                     if '\n' in weight_text:  # 换行分隔
                         weights = [float(x.strip()) for x in weight_text.split('\n') if x.strip()]
                     elif '\t' in weight_text:  # 制表符分隔（从Excel复制）
                         weights = [float(x.strip()) for x in weight_text.split('\t') if x.strip()]
                     else:  # 逗号分隔
                         weights = [float(x.strip()) for x in weight_text.split(',') if x.strip()]
                     
                     weights = np.array(weights)
                     
                     if len(weights) != num_indicators:
                         st.error(f"权重数量({len(weights)})与指标数量({num_indicators})不匹配")
                         weights = np.ones(num_indicators) / num_indicators
                     elif np.sum(weights) > 0:
                         weights = weights / np.sum(weights)  # 归一化
                         st.success("权重数据解析成功")
                     else:
                         st.error("权重总和不能为0")
                         weights = np.ones(num_indicators) / num_indicators
                 except ValueError:
                     st.error("请输入有效的数值，支持逗号、制表符或换行分隔")
                     weights = np.ones(num_indicators) / num_indicators
             else:
                 # 使用滑块输入
                 st.markdown("**或使用滑块设置权重：**")
                 weight_cols = st.columns(min(num_indicators, 4))  # 最多显示4列
                 weights = []
                 
                 for i in range(num_indicators):
                     with weight_cols[i % 4]:
                         weight = st.number_input(f"指标{i+1}权重", value=1.0, min_value=0.0, step=0.1, key=f"weight_{i}")
                         weights.append(weight)
                 
                 weights = np.array(weights)
                 if np.sum(weights) > 0:
                     weights = weights / np.sum(weights)  # 归一化
        else:  # 上传权重文件
            weight_file = st.file_uploader(
                "上传权重文件（CSV或Excel）",
                type=['csv', 'xlsx', 'xls'],
                help="文件应包含一行或一列权重数据",
                key="weight_file"
            )
            
            if weight_file is not None:
                try:
                    if weight_file.name.endswith('.csv'):
                        weight_df = pd.read_csv(weight_file, header=None)
                    else:
                        weight_df = pd.read_excel(weight_file, header=None)
                    
                    # 尝试从第一行或第一列读取权重
                    if weight_df.shape[0] == 1:  # 一行数据
                        weights = weight_df.iloc[0].values
                    elif weight_df.shape[1] == 1:  # 一列数据
                        weights = weight_df.iloc[:, 0].values
                    else:
                        weights = weight_df.iloc[0].values  # 默认取第一行
                    
                    weights = np.array(weights, dtype=float)
                    
                    if len(weights) != num_indicators:
                        st.error(f"权重数量({len(weights)})与指标数量({num_indicators})不匹配")
                        weights = np.ones(num_indicators) / num_indicators
                    elif np.sum(weights) > 0:
                        weights = weights / np.sum(weights)  # 归一化
                        st.success(f"成功读取权重文件：{weight_file.name}")
                    else:
                        st.error("权重总和不能为0")
                        weights = np.ones(num_indicators) / num_indicators
                except Exception as e:
                    st.error(f"权重文件读取错误：{str(e)}")
                    weights = np.ones(num_indicators) / num_indicators
            else:
                weights = np.ones(num_indicators) / num_indicators
        
        # 显示归一化后的权重
        if len(weights) > 0:
            st.markdown("**当前权重分配：**")
            weight_display_cols = st.columns(min(num_indicators, 5))  # 最多显示5列
            for i, w in enumerate(weights):
                with weight_display_cols[i % 5]:
                    st.metric(f"指标{i+1}", f"{w:.3f}")
        
        st.session_state.expert_scores = expert_scores
        st.session_state.indicator_weights = weights
    
    # 步骤3：生成指标评价云
    st.subheader("☁️ 步骤3：生成指标评价云")
    
    if st.button("🎯 生成指标评价云", type="primary"):
        if expert_scores is not None and len(weights) > 0:
            indicator_clouds = calculate_indicator_clouds(expert_scores, weights)
            st.session_state.indicator_clouds = indicator_clouds
            st.success("指标评价云生成完成！")
        else:
            st.error("请先输入专家打分数据和权重")
    
    # 显示指标评价云结果
    if st.session_state.indicator_clouds is not None:
        st.markdown("**指标评价云参数：**")
        indicator_df = pd.DataFrame(st.session_state.indicator_clouds)
        
        # 添加复制按钮
        col_table, col_copy = st.columns([4, 1])
        with col_table:
            st.dataframe(indicator_df, use_container_width=True)
        with col_copy:
            st.markdown("<br>", unsafe_allow_html=True)  # 添加间距
            if st.button("📋 复制表格", key="copy_indicator_table", help="复制指标评价云参数到剪贴板"):
                # 生成可复制的文本格式
                csv_text = indicator_df.to_csv(index=False, sep='\t')
                st.code(csv_text, language=None)
                st.success("数据已生成，请手动复制上方文本框中的内容")
    
    # 步骤4：生成综合评价云
    st.subheader("🌟 步骤4：生成综合评价云")
    
    if st.button("🎯 生成综合评价云", type="primary"):
        if st.session_state.indicator_clouds is not None:
            ex_comp, en_comp, he_comp = calculate_comprehensive_cloud(st.session_state.indicator_clouds)
            st.session_state.comprehensive_cloud = {
                'Ex': ex_comp,
                'En': en_comp,
                'He': he_comp
            }
            st.success("综合评价云生成完成！")
        else:
            st.error("请先生成指标评价云")
    
    # 显示综合评价云结果
    if st.session_state.comprehensive_cloud is not None:
        # 标题和复制按钮
        comp_title_col, comp_copy_col = st.columns([4, 1])
        with comp_title_col:
            st.markdown("**综合评价云参数：**")
        with comp_copy_col:
            if st.button("📋 复制参数", key="copy_comprehensive_params", help="复制综合评价云参数"):
                comp_cloud = st.session_state.comprehensive_cloud
                # 生成可复制的文本格式
                params_text = f"Ex\tEn\tHe\n{comp_cloud['Ex']:.4f}\t{comp_cloud['En']:.4f}\t{comp_cloud['He']:.4f}"
                st.code(params_text, language=None)
                st.success("参数已生成，请手动复制上方文本框中的内容")
        
        comp_cloud = st.session_state.comprehensive_cloud
        
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            st.metric("期望值 Ex", f"{comp_cloud['Ex']:.4f}")
        with comp_col2:
            st.metric("熵 En", f"{comp_cloud['En']:.4f}")
        with comp_col3:
            st.metric("超熵 He", f"{comp_cloud['He']:.4f}")
        
        # 添加到标准云配置按钮
        st.markdown("---")
        add_col1, add_col2 = st.columns([3, 1])
        with add_col1:
            cloud_name = st.text_input("云名称", value="综合评价云", key="add_cloud_name", help="为要添加的云指定名称")
        with add_col2:
            st.markdown("<br>", unsafe_allow_html=True)  # 添加间距对齐
            if st.button("➕ 导入到标准云配置", key="add_to_standard", help="将当前综合评价云参数导入到标准云配置表格的最后一行"):
                # 直接更新最后一行（综合评价云行）的数据
                last_index = len(st.session_state.standard_clouds_data) - 1
                st.session_state.standard_clouds_data.loc[last_index, '云名称'] = cloud_name if cloud_name.strip() else "综合评价云"
                st.session_state.standard_clouds_data.loc[last_index, 'Ex'] = comp_cloud['Ex']
                st.session_state.standard_clouds_data.loc[last_index, 'En'] = comp_cloud['En']
                st.session_state.standard_clouds_data.loc[last_index, 'He'] = comp_cloud['He']
                st.session_state.standard_clouds_data.loc[last_index, '云滴数量'] = 1200
                st.session_state.standard_clouds_data.loc[last_index, '颜色'] = 'green'
                st.session_state.standard_clouds_data.loc[last_index, '绘图符号'] = 's'
                
                st.success(f"已将综合评价云参数导入到标准云配置表格的最后一行")
                st.rerun()  # 刷新页面以显示更新后的标准云配置
    
    # 步骤5：可视化和正向云发生
    if st.session_state.comprehensive_cloud is not None:
        st.subheader("📊 步骤5：综合评价云可视化")
        
        # 生成综合评价云的云滴
        comp_cloud = st.session_state.comprehensive_cloud
        num_drops = st.number_input("云滴数量", value=1000, min_value=100, max_value=5000, step=100)
        
        # 可视化按钮布局
        st.markdown("**选择可视化类型：**")
        viz_cols = st.columns(5)
        
        # 自定义标签输入（在按钮外面）
        with st.expander("🎨 自定义可视化标签"):
            viz_title = st.text_input("图表标题", value="综合评价云", key="viz_title")
            viz_xlabel = st.text_input("X轴标签", value="评价值", key="viz_xlabel")
            viz_ylabel = st.text_input("Y轴标签", value="隶属度", key="viz_ylabel")
        
        with viz_cols[0]:
            if st.button("📊 散点图", use_container_width=True):
                cloud_drops, memberships = generate_cloud_drops(
                    comp_cloud['Ex'], comp_cloud['En'], comp_cloud['He'], num_drops
                )
                fig = plot_scatter(cloud_drops, memberships, f"{viz_title}散点图", viz_xlabel, viz_ylabel)
                st.pyplot(fig)
        
        with viz_cols[1]:
            if st.button("📈 直方图", use_container_width=True):
                cloud_drops, memberships = generate_cloud_drops(
                    comp_cloud['Ex'], comp_cloud['En'], comp_cloud['He'], num_drops
                )
                fig = plot_histogram(cloud_drops, memberships, f"{viz_title}分布图", viz_xlabel, "频数")
                st.pyplot(fig)
        
        with viz_cols[2]:
            if st.button("☁️ 云模型图", use_container_width=True):
                cloud_drops, memberships = generate_cloud_drops(
                    comp_cloud['Ex'], comp_cloud['En'], comp_cloud['He'], num_drops
                )
                fig = plot_cloud_visualization(
                    comp_cloud['Ex'], comp_cloud['En'], comp_cloud['He'],
                    cloud_drops, memberships, f"{viz_title}模型", viz_xlabel, viz_ylabel
                )
                st.pyplot(fig)
        
        with viz_cols[3]:
            if st.button("🔄 组合图", use_container_width=True):
                cloud_drops, memberships = generate_cloud_drops(
                    comp_cloud['Ex'], comp_cloud['En'], comp_cloud['He'], num_drops
                )
                fig = plot_combined_visualization(
                    comp_cloud['Ex'], comp_cloud['En'], comp_cloud['He'],
                    cloud_drops, memberships, f"{viz_title}组合图", viz_xlabel, viz_ylabel
                )
                st.pyplot(fig)
        
        with viz_cols[4]:
            if st.button("⚖️ 标准对比图", use_container_width=True):
                # 自定义标签输入
                with st.expander("🎨 自定义对比图标签"):
                    comp_title = st.text_input("对比图标题", value="综合评价云与标准云对比图", key="comp_title")
                    comp_xlabel = st.text_input("对比图X轴标签", value="评价值", key="comp_xlabel")
                    comp_ylabel = st.text_input("对比图Y轴标签", value="隶属度", key="comp_ylabel")
                
                if st.session_state.comprehensive_cloud is not None:
                    num_drops = st.number_input("云滴数量", value=1000, min_value=100, max_value=5000, step=100, key="comp_drops")
                    fig = plot_comprehensive_with_standards(
                        st.session_state.comprehensive_cloud, 
                        st.session_state.standard_clouds_data, 
                        num_drops,
                        comp_title, 
                        comp_xlabel, 
                        comp_ylabel
                    )
                    st.pyplot(fig)
                    
                    # 添加评价结果分析
                    st.markdown("**评价结果分析：**")
                    comp_ex = st.session_state.comprehensive_cloud['Ex']
                    
                    # 判断综合评价结果属于哪个等级
                    if comp_ex < 25:
                        level = "劣"
                        color = "🔴"
                    elif comp_ex < 50:
                        level = "差"
                        color = "🟠"
                    elif comp_ex < 75:
                        level = "一般"
                        color = "🟡"
                    elif comp_ex < 90:
                        level = "良"
                        color = "🟢"
                    else:
                        level = "优"
                        color = "🟢"
                    
                    st.info(f"{color} 综合评价结果：**{level}** (评分值: {comp_ex:.2f})")
                else:
                    st.warning("请先生成综合评价云")
    
    # 操作按钮
    st.subheader("📋 数据操作")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("📤 导出指标云"):
            if st.session_state.indicator_clouds is not None:
                df = pd.DataFrame(st.session_state.indicator_clouds)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="下载指标云CSV",
                    data=csv,
                    file_name=f"indicator_clouds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("请先生成指标评价云")
    
    with col_btn2:
        if st.button("📤 导出综合云"):
            if st.session_state.comprehensive_cloud is not None:
                df = pd.DataFrame([st.session_state.comprehensive_cloud])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="下载综合云CSV",
                    data=csv,
                    file_name=f"comprehensive_cloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("请先生成综合评价云")
    
    with col_btn3:
        if st.button("🗑️ 清空所有数据"):
            st.session_state.expert_scores = None
            st.session_state.indicator_weights = None
            st.session_state.indicator_clouds = None
            st.session_state.comprehensive_cloud = None
            st.session_state.reverse_data_text = ""
            st.session_state.reverse_weight_text = ""
            st.success("所有数据已清空")
            st.rerun()
    
    # 导入到正向云发生器按钮
    if st.session_state.comprehensive_cloud is not None:
        st.subheader("🔄 导入到正向云发生器")
        st.markdown("将综合评价云参数导入到正向云发生器中进行进一步分析")
        
        comp_cloud = st.session_state.comprehensive_cloud
        st.markdown(f"**将要导入的参数：** Ex={comp_cloud['Ex']:.4f}, En={comp_cloud['En']:.4f}, He={comp_cloud['He']:.4f}")
        
        if st.button("🚀 导入到正向云发生器", type="primary"):
            # 将综合评价云参数设置到正向云发生器
            st.session_state.forward_ex = comp_cloud['Ex']
            st.session_state.forward_en = comp_cloud['En']
            st.session_state.forward_he = comp_cloud['He']
            st.session_state.forward_preset = '自定义'
            
            # 切换到正向云发生器页面
            st.session_state.current_page = "正向云发生器"
            st.success("参数已导入到正向云发生器！正在跳转...")
            st.rerun()

if __name__ == "__main__":
    main()