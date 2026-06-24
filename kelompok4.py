import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="KNN Regresi - Prediksi Omset UMKM",
    page_icon="📊",
    layout="wide"
)

# CSS untuk styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 KNN Regresi - Prediksi Omset UMKM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analisis dan Prediksi Omset UMKM berdasarkan berbagai fitur</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Pengaturan Model")

# Parameter KNN
k_value = st.sidebar.slider("Nilai K (Jumlah Tetangga)", 1, 50, 5, 1)
test_size = st.sidebar.slider("Ukuran Data Uji (Test Size)", 0.1, 0.5, 0.2, 0.05)

# Pilihan fitur
st.sidebar.subheader("📋 Pilih Fitur untuk Prediksi")
use_all_features = st.sidebar.checkbox("Gunakan Semua Fitur", value=True)

# Upload file
uploaded_file = st.file_uploader("📂 Upload file CSV (datase_usaha.csv)", type=['csv'])

if uploaded_file is not None:
    try:
        # Baca data
        df = pd.read_csv(uploaded_file)
        
        # Tampilkan info data
        st.subheader("📋 Preview Data")
        st.write(f"Total data: {len(df)} baris, {len(df.columns)} kolom")
        st.dataframe(df.head(10))
        
        # Preprocessing
        def preprocess_data(df):
            df_clean = df.copy()
            
            # Handle missing values - replace 'unknown' and empty strings with NaN
            df_clean = df_clean.replace(['unknown', 'Unknown', '', 'NULL', 'null'], np.nan)
            
            # Konversi kolom numerik
            numeric_cols = ['tenaga_kerja_perempuan', 'tenaga_kerja_laki_laki', 'aset', 
                            'omset', 'kapasitas_produksi', 'tahun_berdiri', 'laba', 
                            'biaya_karyawan', 'jumlah_pelanggan']
            
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Drop rows dengan target NaN
            df_clean = df_clean.dropna(subset=['omset'])
            
            # Handle missing values untuk fitur numerik
            for col in numeric_cols:
                if col in df_clean.columns and col != 'omset':
                    if df_clean[col].isna().sum() > 0:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
            # Handle missing values untuk fitur kategorikal
            cat_cols = ['jenis_usaha', 'marketplace', 'status_legalitas']
            for col in cat_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna('Unknown')
                    df_clean[col] = df_clean[col].astype(str)
            
            return df_clean, cat_cols, numeric_cols
        
        df_clean, cat_cols, numeric_cols = preprocess_data(df)
        
        # Encode categorical
        label_encoders = {}
        for col in cat_cols:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                label_encoders[col] = le
        
        # Pilih fitur
        st.subheader("🎯 Pemilihan Fitur")
        
        available_cols = [col for col in numeric_cols + cat_cols if col in df_clean.columns and col != 'omset']
        
        if use_all_features:
            selected_features = available_cols
        else:
            selected_features = st.multiselect(
                "Pilih fitur untuk prediksi:",
                available_cols,
                default=available_cols[:min(5, len(available_cols))]
            )
        
        if selected_features and len(selected_features) > 0:
            # Persiapan data untuk modeling
            X = df_clean[selected_features]
            y = df_clean['omset']
            
            # Standarisasi fitur numerik
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
            
            # Train KNN model
            knn = KNeighborsRegressor(n_neighbors=k_value, weights='distance')
            knn.fit(X_train, y_train)
            
            # Prediksi
            y_pred_train = knn.predict(X_train)
            y_pred_test = knn.predict(X_test)
            
            # Metrik evaluasi
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            
            # Tampilkan metrik
            st.subheader("📊 Evaluasi Model")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("RMSE Train", f"{rmse_train:,.0f}")
            with col2:
                st.metric("RMSE Test", f"{rmse_test:,.0f}")
            with col3:
                st.metric("R² Train", f"{r2_train:.4f}")
            with col4:
                st.metric("R² Test", f"{r2_test:.4f}")
            
            col5, col6 = st.columns(2)
            with col5:
                st.metric("MAE Train", f"{mae_train:,.0f}")
            with col6:
                st.metric("MAE Test", f"{mae_test:,.0f}")
            
            # Visualisasi
            st.subheader("📈 Visualisasi Hasil")
            
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Plot prediksi vs aktual - Training
                axes[0].scatter(y_train, y_pred_train, alpha=0.5)
                axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
                axes[0].set_xlabel('Aktual')
                axes[0].set_ylabel('Prediksi')
                axes[0].set_title(f'Training Set (R² = {r2_train:.4f})')
                
                # Plot prediksi vs aktual - Testing
                axes[1].scatter(y_test, y_pred_test, alpha=0.5)
                axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[1].set_xlabel('Aktual')
                axes[1].set_ylabel('Prediksi')
                axes[1].set_title(f'Testing Set (R² = {r2_test:.4f})')
                
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan visualisasi: {str(e)}")
            
            # Distribusi residual
            st.subheader("📊 Distribusi Residual")
            
            try:
                fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
                
                residuals_train = y_train - y_pred_train
                residuals_test = y_test - y_pred_test
                
                axes2[0].hist(residuals_train, bins=30, alpha=0.7, color='blue', edgecolor='black')
                axes2[0].axvline(x=0, color='red', linestyle='--')
                axes2[0].set_xlabel('Residual')
                axes2[0].set_ylabel('Frekuensi')
                axes2[0].set_title(f'Residual Training (Mean = {residuals_train.mean():.0f})')
                
                axes2[1].hist(residuals_test, bins=30, alpha=0.7, color='green', edgecolor='black')
                axes2[1].axvline(x=0, color='red', linestyle='--')
                axes2[1].set_xlabel('Residual')
                axes2[1].set_ylabel('Frekuensi')
                axes2[1].set_title(f'Residual Testing (Mean = {residuals_test.mean():.0f})')
                
                plt.tight_layout()
                st.pyplot(fig2)
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan visualisasi residual: {str(e)}")
            
            # Prediksi baru
            st.subheader("🔮 Prediksi Omset UMKM")
            st.write("Masukkan data UMKM untuk memprediksi omset:")
            
            # Input fitur
            input_values = {}
            cols_per_row = 3
            col_inputs = st.columns(cols_per_row)
            
            for idx, col_name in enumerate(selected_features[:12]):  # Batasi max 12 fitur
                col_idx = idx % cols_per_row
                with col_inputs[col_idx]:
                    if col_name in numeric_cols:
                        default_val = float(df_clean[col_name].median())
                        step = 100000.0 if col_name in ['aset', 'omset', 'laba', 'biaya_karyawan'] else 1.0
                        format_str = "%.0f"
                        input_values[col_name] = st.number_input(
                            f"{col_name.replace('_', ' ').title()}",
                            value=default_val,
                            step=step,
                            format=format_str
                        )
                    else:
                        # Untuk categorical
                        try:
                            unique_vals = sorted(df[col_name].dropna().unique())
                            if len(unique_vals) == 0:
                                unique_vals = ['Unknown']
                            input_values[col_name] = st.selectbox(
                                f"{col_name.replace('_', ' ').title()}",
                                unique_vals
                            )
                        except:
                            input_values[col_name] = st.text_input(
                                f"{col_name.replace('_', ' ').title()}",
                                value="Unknown"
                            )
            
            if st.button("🔮 Prediksi Omset", type="primary"):
                try:
                    # Prepare input untuk prediksi
                    input_array = []
                    for col_name in selected_features:
                        if col_name in cat_cols:
                            # Encode categorical
                            val = input_values[col_name]
                            if val in label_encoders[col_name].classes_:
                                encoded = label_encoders[col_name].transform([val])[0]
                            else:
                                encoded = 0
                            input_array.append(encoded)
                        else:
                            input_array.append(input_values[col_name])
                    
                    # Scale dan prediksi
                    input_scaled = scaler.transform([input_array])
                    prediction = knn.predict(input_scaled)[0]
                    
                    st.success(f"💰 Prediksi Omset: Rp {prediction:,.0f}")
                    
                    # Tampilkan informasi tambahan
                    st.info(f"""
                    **Informasi Prediksi:**
                    - Menggunakan K = {k_value} tetangga terdekat
                    - Fitur yang digunakan: {', '.join(selected_features[:10])}{'...' if len(selected_features) > 10 else ''}
                    - Akurasi Model (R²): {r2_test:.4f}
                    """)
                except Exception as e:
                    st.error(f"Error dalam prediksi: {str(e)}")
            
            # Feature importance
            st.subheader("📊 Pentingnya Fitur")
            
            try:
                feature_importance = {}
                X_mean = X_scaled.mean(axis=0)
                
                for idx, col_name in enumerate(selected_features):
                    X_var = np.tile(X_mean, (100, 1))
                    X_var[:, idx] = np.linspace(-3, 3, 100)
                    pred_var = knn.predict(X_var)
                    feature_importance[col_name] = pred_var.std()
                
                if feature_importance:
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    names = [x[0][:20] for x in sorted_features]  # Truncate long names
                    values = [x[1] for x in sorted_features]
                    
                    ax3.barh(names, values)
                    ax3.set_xlabel('Pengaruh (Standar Deviasi Prediksi)')
                    ax3.set_title('Pentingnya Fitur dalam Model KNN')
                    plt.tight_layout()
                    st.pyplot(fig3)
                else:
                    st.info("Tidak dapat menghitung feature importance")
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan feature importance: {str(e)}")
            
            # Download prediksi
            st.subheader("📥 Download Hasil Prediksi")
            
            try:
                all_pred = knn.predict(X_scaled)
                df_results = df_clean.copy()
                df_results['prediksi_omset'] = all_pred
                
                # Pilih kolom untuk export
                export_cols = ['nama_usaha'] if 'nama_usaha' in df_results.columns else []
                export_cols.extend(['omset', 'prediksi_omset'])
                export_cols = [col for col in export_cols if col in df_results.columns]
                
                if export_cols:
                    csv = df_results[export_cols].to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="prediksi_omset_knn.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Kolom untuk export tidak tersedia")
            except Exception as e:
                st.warning(f"Tidak dapat membuat file download: {str(e)}")
            
            # Analisis tambahan - Hubungan K dengan performa
            st.subheader("🔍 Analisis Nilai K")
            
            try:
                max_k = min(50, len(X_train) // 5)
                if max_k > 1:
                    k_values_analysis = range(1, max_k + 1)
                    train_scores = []
                    test_scores = []
                    
                    for k in k_values_analysis:
                        knn_temp = KNeighborsRegressor(n_neighbors=k, weights='distance')
                        knn_temp.fit(X_train, y_train)
                        train_scores.append(knn_temp.score(X_train, y_train))
                        test_scores.append(knn_temp.score(X_test, y_test))
                    
                    fig4, ax4 = plt.subplots(figsize=(10, 6))
                    ax4.plot(k_values_analysis, train_scores, label='Training R²', marker='o')
                    ax4.plot(k_values_analysis, test_scores, label='Testing R²', marker='s')
                    ax4.axvline(x=k_value, color='red', linestyle='--', label=f'K = {k_value}')
                    ax4.set_xlabel('Nilai K')
                    ax4.set_ylabel('R² Score')
                    ax4.set_title('Performa Model terhadap Nilai K')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig4)
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan analisis K: {str(e)}")
            
        else:
            st.warning("⚠️ Silakan pilih setidaknya satu fitur untuk prediksi.")
            
    except Exception as e:
        st.error(f"❌ Error dalam memproses data: {str(e)}")
        st.info("Pastikan file CSV memiliki format yang benar.")

else:
    st.info("👈 Silakan upload file CSV 'datase_usaha.csv' di sidebar untuk memulai.")
    
    # Tampilkan contoh
    with st.expander("📖 Format Data yang Diharapkan"):
        st.markdown("""
        File CSV harus memiliki kolom-kolom berikut:
        - **id_umkm**: ID unik UMKM
        - **nama_usaha**: Nama usaha
        - **jenis_usaha**: Kategori usaha
        - **tenaga_kerja_perempuan**: Jumlah tenaga kerja perempuan
        - **tenaga_kerja_laki_laki**: Jumlah tenaga kerja laki-laki
        - **aset**: Nilai aset
        - **omset**: Nilai omset (target prediksi)
        - **marketplace**: Platform marketplace
        - **kapasitas_produksi**: Kapasitas produksi
        - **status_legalitas**: Status legalitas
        - **tahun_berdiri**: Tahun berdiri
        - **laba**: Laba usaha
        - **biaya_karyawan**: Biaya karyawan
        - **jumlah_pelanggan**: Jumlah pelanggan
        """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Aplikasi KNN Regresi untuk Prediksi Omset UMKM | Dibuat dengan Streamlit
</div>
""", unsafe_allow_html=True)
