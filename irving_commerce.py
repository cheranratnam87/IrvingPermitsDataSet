import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Load and Preprocess Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(
        r"https://raw.githubusercontent.com/cheranratnam87/IrvingPermitsDataSet/main/Commercial_Permits_Issued%253A_Feb_15_2022_Through_Present.csv"
    )

    df['ZIP_Code'] = df['Address'].str.extract(r'(\d{5})')
    df['Issued_Date_dt'] = pd.to_datetime(df['Issued_Date'], errors='coerce')
    df['Finaled_Date_dt'] = pd.to_datetime(df['Finaled_Date'], errors='coerce')
    df['Year'] = df['Issued_Date_dt'].dt.year

    df['Valuation_clean'] = df['Valuation'].replace(r"[\$,]", "", regex=True).astype(str)
    df['Fees_Paid_clean'] = df['Fees_Paid'].replace(r"[\$,]", "", regex=True).astype(str)

    df['Valuation_clean'] = pd.to_numeric(df['Valuation_clean'], errors='coerce')
    df['Fees_Paid_clean'] = pd.to_numeric(df['Fees_Paid_clean'], errors='coerce')

    df['Duration'] = (df['Finaled_Date_dt'] - df['Issued_Date_dt']).dt.days
    return df

# --- Similar Case Finder for Fee Estimation ---
def find_similar_cases(df, permit_type, sq_ft=None, valuation=None, k=5):
    filtered_df = df[df['Permit_Type'] == permit_type].dropna(subset=['Fees_Paid_clean'])

    if filtered_df.empty:
        return None, None

    weight_sq_ft = 1 if sq_ft else 0
    weight_valuation = 1 if valuation else 0
    total_weight = weight_sq_ft + weight_valuation

    if total_weight == 0:
        return filtered_df.sample(n=min(k, len(filtered_df))), filtered_df['Fees_Paid_clean'].mean()

    filtered_df = filtered_df.dropna(subset=['Square_Feet', 'Valuation_clean'])

    if filtered_df.empty:
        return None, None

    filtered_df['SqFt_Norm'] = (filtered_df['Square_Feet'] - filtered_df['Square_Feet'].mean()) / filtered_df['Square_Feet'].std()
    filtered_df['Valuation_Norm'] = (filtered_df['Valuation_clean'] - filtered_df['Valuation_clean'].mean()) / filtered_df['Valuation_clean'].std()

    sq_ft_norm = (sq_ft - filtered_df['Square_Feet'].mean()) / filtered_df['Square_Feet'].std() if sq_ft else None
    valuation_norm = (valuation - filtered_df['Valuation_clean'].mean()) / filtered_df['Valuation_clean'].std() if valuation else None

    def calculate_distance(row):
        dist_sq_ft = (row['SqFt_Norm'] - sq_ft_norm) ** 2 if sq_ft_norm is not None else 0
        dist_valuation = (row['Valuation_Norm'] - valuation_norm) ** 2 if valuation_norm is not None else 0
        return np.sqrt((weight_sq_ft * dist_sq_ft + weight_valuation * dist_valuation) / total_weight)

    filtered_df['Distance'] = filtered_df.apply(calculate_distance, axis=1)
    closest_cases = filtered_df.nsmallest(k, 'Distance')

    estimated_fee = closest_cases['Fees_Paid_clean'].mean()
    
    return closest_cases[['ZIP_Code', 'Permit_Type', 'Square_Feet', 'Valuation', 'Fees_Paid']], estimated_fee

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Commercial Permits Dashboard", layout="wide")

    st.markdown("""
    <h2 style='text-align: center; color: #2F5496;'>
        Commercial Permits Dashboard in Irving, TX
    </h2>
    <p style='text-align: center; font-size:14px; margin-top: 0;'>
        Dashboard Created by 
        <a href='https://cheranratnam.com/about/' style='color: #2F5496; font-weight: bold;'>
            Cheran Ratnam
        </a>
        <br>
        <a href='https://cheranratnam.com/about/' target='_blank'>Website</a> | 
        <a href='https://www.linkedin.com/in/cheranratnam/' target='_blank'>LinkedIn</a> | 
        <a href='https://data-cityofirving.opendata.arcgis.com/datasets/f5db890d65d640efa350c9419c248aad_0/explore'>Link To Dataset</a>
    </p>            
    <hr>
    """, unsafe_allow_html=True)

    df = load_data()

    # Sidebar Filters
    st.sidebar.header("Filters")

    zip_codes_available = sorted(df['ZIP_Code'].dropna().unique())
    selected_zip = st.sidebar.multiselect("Select ZIP Code(s):", zip_codes_available, default=[])

    years_available = sorted(df['Year'].dropna().unique())
    selected_years = st.sidebar.multiselect("Select Year(s):", years_available, default=[])

    permit_types_available = sorted(df['Permit_Type'].dropna().unique())
    selected_permit_types = st.sidebar.multiselect("Select Permit Type(s):", permit_types_available, default=[])

    statuses_available = sorted(df['Status'].dropna().unique())
    selected_statuses = st.sidebar.multiselect("Select Status(es):", statuses_available, default=[])

    # Apply Filters
    filtered_df = df.copy()
    if selected_zip:
        filtered_df = filtered_df[filtered_df['ZIP_Code'].isin(selected_zip)]
    if selected_years:
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
    if selected_permit_types:
        filtered_df = filtered_df[filtered_df['Permit_Type'].isin(selected_permit_types)]
    if selected_statuses:
        filtered_df = filtered_df[filtered_df['Status'].isin(selected_statuses)]

    # Tabs: Dashboard & Fee Estimator
    tab1, tab2 = st.tabs(["📊 Dashboard", "💰 Fee Estimate"])

    with tab1:
        # Permit Types by Count
        permit_type_counts = filtered_df['Permit_Type'].value_counts().reset_index()
        permit_type_counts.columns = ['Permit_Type', 'Count']

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=permit_type_counts, x='Permit_Type', y='Count', ax=ax1)
        ax1.set_title("Permit Types and Their Counts")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig1)

        # Permit Status Counts
        status_counts = filtered_df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=status_counts, x='Status', y='Count', ax=ax2, palette='viridis')
        ax2.set_title("Permit Status Counts")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig2)

        # Permit Counts Over the Years
        yearly_counts = filtered_df.groupby('Year').size().reset_index(name='Count')

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=yearly_counts, x='Year', y='Count', marker='o', ax=ax3)
        ax3.set_title("Number of Permits Issued Over the Years")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig3)

        # --- Valuation & Fees Analysis ---------------------------------------
    st.subheader("Valuation & Fees Analysis")

    # (A) Scatter Plot: Valuation vs Fees
    val_fees_df = filtered_df.dropna(subset=['Valuation_clean', 'Fees_Paid_clean'])
    if val_fees_df.empty:
        st.warning("No Valuation/Fees data available for the selected filters.")
    else:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=val_fees_df,
            x='Valuation_clean',
            y='Fees_Paid_clean',
            ax=ax4,
            alpha=0.6
        )
        ax4.set_title("Valuation vs. Fees Paid")
        ax4.set_xlabel("Valuation (USD)")
        ax4.set_ylabel("Fees Paid (USD)")
        plt.tight_layout()
        st.pyplot(fig4)

    # (B) Average Fees by Permit Type
    avg_fees = (
        filtered_df
        .dropna(subset=['Fees_Paid_clean'])
        .groupby('Permit_Type')['Fees_Paid_clean']
        .mean()
        .reset_index(name='Avg_Fees')
    )

    if avg_fees.empty:
        st.warning("No fee data available for the selected filters.")
    else:
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=avg_fees,
            x='Permit_Type',
            y='Avg_Fees',
            palette='Blues_d',
            ax=ax5
        )
        ax5.set_title("Average Fees Paid by Permit Type")
        ax5.set_xlabel("Permit Type")
        ax5.set_ylabel("Average Fees (USD)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig5)

    # --- Project Size & Duration Insights --------------------------------
    st.subheader("Project Size & Duration Insights")

    # (A) Average Square Footage by Permit Type
    sq_ft_df = filtered_df.dropna(subset=['Square_Feet'])
    if not sq_ft_df.empty:
        avg_sq_ft = (
            sq_ft_df
            .groupby('Permit_Type')['Square_Feet']
            .mean()
            .reset_index(name='Avg_SqFt')
        )
        
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=avg_sq_ft,
            x='Permit_Type',
            y='Avg_SqFt',
            palette='Greens_d',
            ax=ax6
        )
        ax6.set_title("Average Square Footage by Permit Type")
        ax6.set_xlabel("Permit Type")
        ax6.set_ylabel("Avg. Square Feet")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig6)
    else:
        st.info("No square footage data available for the selected filters.")

    # (B) Duration Boxplot by Permit Type (if Finaled) - With Outlier Handling
    duration_df = filtered_df.dropna(subset=['Duration', 'Permit_Type']).copy()

    # 1. Remove negative durations
    duration_df = duration_df[duration_df['Duration'] >= 0]

    # 2. Cap outliers at the 95th percentile
    upper_cap = duration_df['Duration'].quantile(0.95)
    duration_df.loc[duration_df['Duration'] > upper_cap, 'Duration'] = upper_cap

    if not duration_df.empty:
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=duration_df,
            x='Permit_Type',
            y='Duration',
            ax=ax7
        )
        ax7.set_title("Permit Duration (Days) by Permit Type (Outliers Capped)")
        ax7.set_xlabel("Permit Type")
        ax7.set_ylabel("Duration (Days)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig7)

        # Explanation for users about data cleaning
        st.markdown("""
        **Note**: 
        - Negative durations (where a final date precedes the issued date) were removed as invalid.  
        - Durations above the 95th percentile are capped to reduce the effect of extreme outliers.  
        This ensures the boxplot is more readable and focuses on the majority of typical projects.
        """)
    else:
        st.info("No duration data available (no finaled dates) for the selected filters.")

if __name__ == "__main__":
    main()

