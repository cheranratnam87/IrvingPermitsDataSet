import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    # Replace with your own CSV path or data source
    df = pd.read_csv(
        r"https://raw.githubusercontent.com/cheranratnam87/IrvingPermitsDataSet/refs/heads/main/Commercial_Permits_Issued%253A_Feb_15_2022_Through_Present.csv"
    )
    
    # Extract ZIP codes
    df['ZIP_Code'] = df['Address'].str.extract(r'(\d{5})')
    
    # Convert Issued_Date and Finaled_Date to datetime
    df['Issued_Date_dt'] = pd.to_datetime(df['Issued_Date'], errors='coerce')
    df['Finaled_Date_dt'] = pd.to_datetime(df['Finaled_Date'], errors='coerce')
    
    # Extract year
    df['Year'] = df['Issued_Date_dt'].dt.year
    
    # Clean valuation and fees for numeric analyses
    df['Valuation_clean'] = (
        df['Valuation']
        .replace('[\$,]', '', regex=True)
        .astype(float, errors='ignore')
    )
    df['Fees_Paid_clean'] = (
        df['Fees_Paid']
        .replace('[\$,]', '', regex=True)
        .astype(float, errors='ignore')
    )
    
    # Calculate permit duration (in days) if finaled
    df['Duration'] = (df['Finaled_Date_dt'] - df['Issued_Date_dt']).dt.days
    
    return df

def main():
    # --- Custom Header / Credits ---
    st.markdown("""
    <h2 style='text-align: center; color: #2F5496; margin-bottom: 0;'>
        Commercial Permits in Irving
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

    # -- Sidebar Filters ------------------------------------------------------
    st.sidebar.header("Filters")

    # 1) Filter by ZIP code
    zip_codes_available = sorted(df['ZIP_Code'].dropna().unique())
    selected_zip = st.sidebar.multiselect(
        "Select ZIP Code(s):",
        options=zip_codes_available,
        default=[]
    )

    # 2) Filter by year
    years_available = sorted(df['Year'].dropna().unique())
    selected_years = st.sidebar.multiselect(
        "Select Year(s):",
        options=years_available,
        default=[]
    )
    
    # 3) Filter by Permit Type
    permit_types_available = sorted(df['Permit_Type'].dropna().unique())
    selected_permit_types = st.sidebar.multiselect(
        "Select Permit Type(s):",
        options=permit_types_available,
        default=[]
    )
    
    # 4) Filter by Status
    statuses_available = sorted(df['Status'].dropna().unique())
    selected_statuses = st.sidebar.multiselect(
        "Select Status(es):",
        options=statuses_available,
        default=[]
    )

    # -- Apply Filters -------------------------------------------------------
    filtered_df = df.copy()

    if selected_zip:
        filtered_df = filtered_df[filtered_df['ZIP_Code'].isin(selected_zip)]
    if selected_years:
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
    if selected_permit_types:
        filtered_df = filtered_df[filtered_df['Permit_Type'].isin(selected_permit_types)]
    if selected_statuses:
        filtered_df = filtered_df[filtered_df['Status'].isin(selected_statuses)]

    # --- 1. Permit Types by Count -------------------------------------------
    st.subheader("1. Permit Types by Count")

    permit_type_counts = (
        filtered_df['Permit_Type']
        .value_counts()
        .reset_index(name='Count')
        .rename(columns={'index': 'Permit_Type'})
    )

    if permit_type_counts.empty:
        st.warning("No data available for the selected filters.")
    else:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=permit_type_counts,
            x='Permit_Type',
            y='Count',
            ax=ax1
        )
        ax1.set_title("Permit Types and Their Counts")
        ax1.set_xlabel("Permit Type")
        ax1.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig1)

    # --- 2. Permit Status Counts --------------------------------------------
    st.subheader("2. Permit Status Counts")

    status_counts = (
        filtered_df['Status']
        .value_counts()
        .reset_index(name='Count')
        .rename(columns={'index': 'Status'})
    )

    if status_counts.empty:
        st.warning("No data available for the selected filters.")
    else:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=status_counts,
            x='Status',
            y='Count',
            ax=ax2,
            palette='viridis'
        )
        ax2.set_title("Permit Status Counts")
        ax2.set_xlabel("Permit Status")
        ax2.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)

    # --- 3. Permit Counts Over the Years (Line Chart) -----------------------
    st.subheader("3. Permit Counts Over the Years")

    yearly_counts = (
        filtered_df
        .groupby('Year')
        .size()
        .reset_index(name='Count')
        .dropna(subset=['Year'])
    )

    if yearly_counts.empty:
        st.warning("No data available for the selected filters.")
    else:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=yearly_counts,
            x='Year',
            y='Count',
            marker='o',
            ax=ax3
        )
        ax3.set_title("Number of Permits Issued Over the Years")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig3)

    # --- 4. Valuation & Fees Analysis ---------------------------------------
    st.subheader("4. Valuation & Fees Analysis")

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

    # --- 5. Project Size & Duration Insights --------------------------------
    st.subheader("5. Project Size & Duration Insights")

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
