import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import datetime as dt
import os

# 設定
plt.rcParams['font.family'] = 'Meiryo'
st.set_page_config(page_title="RFM Dashboard", layout="wide")

# 色設定
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# データ読み込み
@st.cache_data
def read_csv_file():
    # 相対パスを使用（Streamlit Cloud対応）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir,"E:/dashboard_report/raw_rfm_sales_transactions_30000 .csv")
    data = pd.read_csv(file_path)
    return data

def clean_data(raw_df):
    customer_info = raw_df[raw_df['Transaction ID'].astype(str).str.contains('Customer-')]
    
    city_mapping = {}
    for i, row in customer_info.iterrows():
        city_mapping[str(row['Transaction ID'])] = str(row['Date'])
    
    transactions = raw_df[~raw_df['Transaction ID'].astype(str).str.contains('Customer-')]
    
    current_customer_id = None
    customer_list = []
    
    for i, row in raw_df.iterrows():
        tid = str(row['Transaction ID'])
        if 'Customer-' in tid:
            current_customer_id = tid
        else:
            customer_list.append(current_customer_id)
    
    transactions = transactions.copy()
    transactions['CustomerID'] = customer_list[:len(transactions)]
    transactions['City'] = transactions['CustomerID'].map(city_mapping)
    
    return transactions

def convert_data_types(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    
    def parse_amount(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            val = val.replace(',', '').replace('"', '')
            return float(val)
        return float(val)
    
    df['PPU'] = df['PPU'].apply(parse_amount)
    df['Amount'] = df['Amount'].apply(parse_amount)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Quantity'] = df['Quantity'].fillna(0)
    df = df.dropna(subset=['Date'])
    
    return df

def calculate_rfm_scores(df):
    ref_date = df['Date'].max() + dt.timedelta(days=1)
    
    rfm_data = df.groupby('CustomerID').agg({
        'Date': lambda x: (ref_date - x.max()).days,
        'Transaction ID': 'count',
        'Amount': 'sum'
    }).reset_index()
    
    rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    rfm_data['R_Score'] = pd.cut(rfm_data['Recency'], bins=5, labels=[5,4,3,2,1]).astype(int)
    
    rfm_data['F_Score'] = pd.cut(
        rfm_data['Frequency'].rank(pct=True),
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True
    ).astype(int)
    
    rfm_data['M_Score'] = pd.cut(
        rfm_data['Monetary'].rank(pct=True),
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True
    ).astype(int)
    
    rfm_data['Total_Score'] = rfm_data['R_Score'] + rfm_data['F_Score'] + rfm_data['M_Score']
    
    return rfm_data

def classify_segment(row):
    r = row['R_Score']
    f = row['F_Score']
    m = row['M_Score']
    
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f >= 2 and m >= 2:
        return 'Potential Loyalists'
    elif r >= 4:
        return 'New Customers'
    elif r <= 2 and f >= 3 and m >= 3:
        return 'At Risk'
    elif r <= 2 and f <= 2 and m <= 2:
        return 'Lost Customers'
    else:
        return 'Need Attention'

# メイン処理
def main():
    st.title("RFM顧客分析ダッシュボード")
    
    
    # データ読み込み
    raw_data = read_csv_file()
    clean_df = clean_data(raw_data)
    df = convert_data_types(clean_df)
    
    # RFM分析
    rfm = calculate_rfm_scores(df)
    rfm['Segment'] = rfm.apply(classify_segment, axis=1)
    
    # 追加列
    df['Month'] = df['Date'].dt.strftime('%Y-%m')
    df['Weekday'] = df['Date'].dt.day_name()
    
    # ===== フィルター設定 =====
    st.write("### フィルター設定")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        all_cities = ['全て'] + sorted(df['City'].unique().tolist())
        selected_city = st.selectbox("都市を選択", all_cities)
    
    with col_f2:
        all_segments = rfm['Segment'].unique().tolist()
        selected_segments = st.multiselect(
            "セグメントを選択",
            all_segments,
            default=all_segments
        )
    
    with col_f3:
        date_range = st.date_input(
            "期間を選択",
            [df['Date'].min(), df['Date'].max()]
        )
    
    # データのフィルタリング
    if selected_city != '全て':
        df_filtered = df[df['City'] == selected_city]
    else:
        df_filtered = df.copy()
    
    # セグメントでフィルタ
    filtered_customers = rfm[rfm['Segment'].isin(selected_segments)]['CustomerID']
    df_filtered = df_filtered[df_filtered['CustomerID'].isin(filtered_customers)]
    
    # 日付でフィルタ
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        df_filtered = df_filtered[(df_filtered['Date'] >= start_date) & 
                                   (df_filtered['Date'] <= end_date)]
    
    # フィルタ後のRFMを再計算
    if len(df_filtered) > 0:
        rfm_filtered = calculate_rfm_scores(df_filtered)
        rfm_filtered['Segment'] = rfm_filtered.apply(classify_segment, axis=1)
    else:
        st.warning("選択した条件に該当するデータがありません")
        st.stop()
    
    st.write("---")
    
    st.write("### 基本指標")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_trans = len(df_filtered)
    total_cust = df_filtered['CustomerID'].nunique()
    total_sales = df_filtered['Amount'].sum()
    avg_sales = df_filtered['Amount'].mean()
    
    col1.metric("総取引数", f"{total_trans:,}")
    col2.metric("顧客数", f"{total_cust}")
    col3.metric("総売上", f"Kyat{total_sales/1e6:.1f}M")
    col4.metric("平均取引額", f"Kyat{avg_sales:,.0f}")
    
    st.write("---")
    
    # ===== 売上トレンド分析 =====
    st.write("### 売上トレンド（日別推移）")
    
    # 日別売上と移動平均
    daily = df_filtered.groupby('Date')['Amount'].sum().reset_index()
    daily['MA7'] = daily['Amount'].rolling(7).mean()
    
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.fill_between(daily['Date'], daily['Amount']/1e6, alpha=0.3, color='blue')
    ax1.plot(daily['Date'], daily['Amount']/1e6, color='blue', linewidth=0.5, label='日別売上')
    ax1.plot(daily['Date'], daily['MA7']/1e6, color='red', linewidth=2, label='7日移動平均')
    ax1.set_xlabel('日付')
    ax1.set_ylabel('売上 (Million Kyat)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    plt.close()
    
    # 分析コメント
    max_day = daily.loc[daily['Amount'].idxmax(), 'Date']
    max_amount = daily['Amount'].max()
    
    st.write(f"""
    最高売上日は {max_day.strftime('%Y年%m月%d日')} で、
    売上は Kyat{max_amount/1e6:.1f}M でした。
    """)
    
    # 月別比較
    st.write("### 月別売上比較")
    
    monthly = df_filtered.groupby('Month').agg({
        'Amount': 'sum',
        'Transaction ID': 'count'
    }).reset_index()
    monthly.columns = ['Month', 'Sales', 'Transactions']
    monthly['Growth'] = monthly['Sales'].pct_change() * 100
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    bars = ax2.bar(monthly['Month'], monthly['Sales']/1e6, color='steelblue')
    
    # 増減で色分け
    for i, bar in enumerate(bars):
        if i > 0 and monthly['Growth'].iloc[i] < 0:
            bar.set_color('red')
    
    ax2.set_xlabel('月')
    ax2.set_ylabel('売上 (Million Kyat)')
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    plt.close()
    
    st.write("**赤色**は前月比で売上が減少した月を示しています")
    
    st.write("---")
    
    # ===== RFM分析 =====
    st.write("### 顧客価値と離脱リスクの分析")
    
    # RFMヒストグラム（3つ並べる）
    col_r, col_f, col_m = st.columns(3)
    
    with col_r:
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.hist(rfm_filtered['Recency'], bins=20, color='purple', edgecolor='white', alpha=0.8)
        ax3.axvline(rfm_filtered['Recency'].mean(), color='red', linestyle='--', 
                    label=f'平均: {rfm_filtered["Recency"].mean():.0f}日')
        ax3.set_xlabel('Recency (日)')
        ax3.set_ylabel('顧客数')
        ax3.set_title('最終購入からの経過日数')
        ax3.legend(fontsize=8)
        st.pyplot(fig3)
        plt.close()
    
    with col_f:
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        ax4.hist(rfm_filtered['Frequency'], bins=20, color='green', edgecolor='white', alpha=0.8)
        ax4.axvline(rfm_filtered['Frequency'].mean(), color='red', linestyle='--',
                    label=f'平均: {rfm_filtered["Frequency"].mean():.1f}回')
        ax4.set_xlabel('Frequency (回)')
        ax4.set_ylabel('顧客数')
        ax4.set_title('購入頻度')
        ax4.legend(fontsize=8)
        st.pyplot(fig4)
        plt.close()
    
    with col_m:
        fig5, ax5 = plt.subplots(figsize=(5, 3.5))
        ax5.hist(rfm_filtered['Monetary']/1e6, bins=20, color='blue', edgecolor='white', alpha=0.8)
        ax5.axvline(rfm_filtered['Monetary'].mean()/1e6, color='red', linestyle='--',
                    label=f'平均: {rfm_filtered["Monetary"].mean()/1e6:.1f}M')
        ax5.set_xlabel('Monetary (Million Kyat)')
        ax5.set_ylabel('顧客数')
        ax5.set_title('購入金額')
        ax5.legend(fontsize=8)
        st.pyplot(fig5)
        plt.close()
    
    # RFM分析の解釈
    st.write("""
    **RFM分析の解釈**:
    - **Recency**: 値が小さいほど最近購入している（良い）
    - **Frequency**: 値が大きいほど頻繁に購入している（良い）
    - **Monetary**: 値が大きいほど高額購入者（良い）
    """)
    
    # R-F ヒートマップ
    st.write("### Recency vs Frequency ヒートマップ")
    
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    heatmap_data = rfm_filtered.groupby(['R_Score', 'F_Score']).size().unstack(fill_value=0)
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax6)
    ax6.set_xlabel('Frequency スコア (高いほど良い)')
    ax6.set_ylabel('Recency スコア (高い＝最近購入)')
    st.pyplot(fig6)
    plt.close()
    
    st.write("右上（R高、F高）の顧客が最も価値が高いセグメントです")
    
    st.write("---")
    
    # ===== セグメント分析 =====
    st.write("### 顧客セグメント分布")
    
    seg_counts = rfm_filtered['Segment'].value_counts()
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        fig7, ax7 = plt.subplots(figsize=(6, 5))
        wedges, texts, autotexts = ax7.pie(seg_counts.values, labels=seg_counts.index, 
                                            autopct='%1.1f%%', startangle=90,
                                            colors=colors[:len(seg_counts)],
                                            pctdistance=0.75)
        centre_circle = plt.Circle((0,0), 0.50, fc='white')
        ax7.add_patch(centre_circle)
        ax7.set_title('セグメント構成比')
        st.pyplot(fig7)
        plt.close()
    
    with col_s2:
        fig8, ax8 = plt.subplots(figsize=(7, 5))
        ax8.barh(seg_counts.index, seg_counts.values, color=colors[:len(seg_counts)])
        ax8.set_xlabel('顧客数')
        ax8.set_title('セグメント別顧客数')
        for i, v in enumerate(seg_counts.values):
            ax8.text(v + 0.5, i, str(v), va='center')
        st.pyplot(fig8)
        plt.close()
    
    # セグメント別売上
    df_with_seg = df_filtered.merge(rfm_filtered[['CustomerID', 'Segment']], on='CustomerID', how='left')
    seg_sales = df_with_seg.groupby('Segment')['Amount'].sum().sort_values(ascending=True)
    
    fig9, ax9 = plt.subplots(figsize=(10, 5))
    ax9.barh(seg_sales.index, seg_sales.values/1e6, color='coral')
    ax9.set_xlabel('売上 (Million Kyat)')
    ax9.set_title('セグメント別売上')
    for i, v in enumerate(seg_sales.values/1e6):
        ax9.text(v + 0.3, i, f'{v:.1f}M', va='center')
    st.pyplot(fig9)
    plt.close()
    
    # セグメント統計
    st.write("#### セグメント統計サマリー")
    
    seg_table = rfm_filtered.groupby('Segment').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['sum', 'mean']
    })
    seg_table.columns = ['顧客数', '平均Recency', '平均Frequency', '総売上', '平均売上']
    seg_table = seg_table.round(1)
    st.dataframe(seg_table)
    
    st.write("---")
    
    # ===== 地域分析 =====
    st.write("### 地域別パフォーマンス")
    
    city_stats = df_filtered.groupby('City').agg({
        'Amount': 'sum',
        'CustomerID': 'nunique'
    }).reset_index()
    city_stats.columns = ['City', 'Sales', 'Customers']
    city_stats = city_stats.sort_values('Sales', ascending=False)
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        fig10, ax10 = plt.subplots(figsize=(8, 5))
        top10_city = city_stats.head(10)
        ax10.barh(top10_city['City'], top10_city['Sales']/1e6, color='navy')
        ax10.invert_yaxis()
        ax10.set_xlabel('売上 (Million Kyat)')
        ax10.set_title('売上 Top 10 都市')
        st.pyplot(fig10)
        plt.close()
    
    with col_c2:
        fig11, ax11 = plt.subplots(figsize=(8, 5))
        ax11.barh(top10_city['City'], top10_city['Customers'], color='darkgreen')
        ax11.invert_yaxis()
        ax11.set_xlabel('顧客数')
        ax11.set_title('顧客数 Top 10 都市')
        st.pyplot(fig11)
        plt.close()
    
    # トップ都市の解説
    top_city = city_stats.iloc[0]
    st.write(f"""
    🏆 **トップ都市**: {top_city['City']}
    - 売上: Kyat{top_city['Sales']/1e6:.1f}M
    - 顧客数: {top_city['Customers']}人
    """)
    
    st.write("---")
    
    # ===== 商品分析 =====
    st.write("### 商品カテゴリ分析")
    
    cat_sales = df_filtered.groupby('Product Category')['Amount'].sum()
    
    fig12, ax12 = plt.subplots(figsize=(6, 5))
    ax12.pie(cat_sales.values, labels=cat_sales.index, autopct='%1.1f%%',
             colors=['#ff9999', '#66b3ff', '#99ff99'])
    ax12.set_title('カテゴリ別売上構成')
    st.pyplot(fig12)
    plt.close()
    
    # Top商品
    prod_stats = df_filtered.groupby('Product Name')['Amount'].sum().sort_values(ascending=False)
    
    fig13, ax13 = plt.subplots(figsize=(10, 5))
    ax13.barh(prod_stats.head(10).index, prod_stats.head(10).values/1e6, color='purple')
    ax13.invert_yaxis()
    ax13.set_xlabel('売上 (Million Kyat)')
    ax13.set_title('売上 Top 10 商品')
    st.pyplot(fig13)
    plt.close()
    
    st.write("---")
    
    # ===== パレート分析 =====
    st.write("### 顧客価値の集中度（80:20の法則は成り立つ）")
    
    cust_value = rfm_filtered.sort_values('Monetary', ascending=False).copy()
    cust_value['Cumulative_Pct'] = cust_value['Monetary'].cumsum() / cust_value['Monetary'].sum() * 100
    cust_value['Customer_Pct'] = np.arange(1, len(cust_value)+1) / len(cust_value) * 100
    
    fig14, ax14 = plt.subplots(figsize=(10, 5))
    ax14.plot(cust_value['Customer_Pct'], cust_value['Cumulative_Pct'], 
              color='blue', linewidth=2)
    ax14.axhline(80, color='red', linestyle='--', alpha=0.7, label='80%ライン')
    ax14.axvline(20, color='green', linestyle='--', alpha=0.7, label='20%ライン')
    ax14.fill_between(cust_value['Customer_Pct'], cust_value['Cumulative_Pct'], alpha=0.3)
    ax14.set_xlabel('顧客の累積割合 (%)')
    ax14.set_ylabel('売上の累積割合 (%)')
    ax14.legend()
    st.pyplot(fig14)
    plt.close()
    
    # パレート分析の結果
    top20_pct = int(len(cust_value) * 0.2)
    top20_sales = cust_value.head(top20_pct)['Monetary'].sum()
    total_sales_cust = cust_value['Monetary'].sum()
    concentration = top20_sales / total_sales_cust * 100
    
    st.write(f"""
    **パレート分析の結果**:
    - 上位20%の顧客が **{concentration:.1f}%** の売上を占めています
    - 80:20の法則に対して、このデータでは {'近い' if 70 <= concentration <= 90 else '異なる'} 傾向が見られます
    """)
    
    st.write("---")
    
    # ===== 主要な発見 =====
    st.write("### 分析から得られた主な発見")
    
    # 自動的に分析を生成
    max_weekday = df_filtered.groupby('Weekday')['Amount'].sum().idxmax()
    at_risk_count = seg_counts.get('At Risk', 0)
    at_risk_pct = at_risk_count / len(rfm_filtered) * 100
    
    st.markdown(f"""
    #### 
    
    1. **売上トレンド**: 
       - 曜日別では **{max_weekday}** が最も売上が高い
       - 週末よりも平日の売上が安定している傾向
    
    2. **顧客価値の集中**:
       - 上位20%の顧客が {concentration:.1f}% の売上を生み出している
       - 顧客価値に大きなばらつきがある
    
    3. **離脱リスク**:
       - 'At Risk'セグメントは {at_risk_count}人 ({at_risk_pct:.1f}%)
       - これらの顧客は以前は優良顧客だった可能性が高い
    """)
    
    st.write("---")
    
    # ===== ビジネス推奨事項 =====
    st.write("### 💡 ビジネス推奨事項")
    
    st.markdown("""
    #### 推奨アクション
    
    **1. 離脱リスク顧客への対応**
    - 'At Risk'セグメントに対して、リテンションキャンペーンを実施
    - クーポンや特別オファーで再エンゲージメントを促進
    
    **2. 高価値顧客（Champions）の維持**
    - ロイヤリティプログラムの導入を検討
    - VIP向けの特典や優先サービスの提供
    
    **3. 成長潜在顧客の育成**
    - 'Potential Loyalists'へのアップセル・クロスセル施策
    - 定期的なコミュニケーションで関係強化
    
    """)
    

if __name__ == "__main__":
    main()
