# ============= ANALISIS NILAI K =============
st.subheader("🔍 Analisis Mendalam Nilai K")

# Tab untuk berbagai analisis
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performa vs K", "📊 Error vs K", "🎯 Elbow Method", "📋 Rekomendasi"])

with tab1:
    st.markdown("### 📈 Pengaruh Nilai K terhadap Performa Model")
    
    # Hitung performa untuk berbagai nilai K
    max_k = min(100, len(X_train) // 3)
    k_values_analysis = range(1, max_k + 1)
    train_scores = []
    test_scores = []
    train_rmse = []
    test_rmse = []
    train_mae = []
    test_mae = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, k in enumerate(k_values_analysis):
        status_text.text(f"Menghitung untuk K = {k}...")
        knn_temp = KNeighborsRegressor(n_neighbors=k, weights='distance')
        knn_temp.fit(X_train, y_train)
        
        y_pred_train_k = knn_temp.predict(X_train)
        y_pred_test_k = knn_temp.predict(X_test)
        
        train_scores.append(knn_temp.score(X_train, y_train))
        test_scores.append(knn_temp.score(X_test, y_test))
        train_rmse.append(np.sqrt(mean_squared_error(y_train, y_pred_train_k)))
        test_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred_test_k)))
        train_mae.append(mean_absolute_error(y_train, y_pred_train_k))
        test_mae.append(mean_absolute_error(y_test, y_pred_test_k))
        
        progress_bar.progress((i + 1) / len(k_values_analysis))
    
    status_text.text("✅ Analisis selesai!")
    progress_bar.empty()
    
    # Plot R² Score
    fig_k1, ax_k1 = plt.subplots(figsize=(12, 6))
    ax_k1.plot(k_values_analysis, train_scores, label='Training R²', marker='o', markersize=4, linewidth=2)
    ax_k1.plot(k_values_analysis, test_scores, label='Testing R²', marker='s', markersize=4, linewidth=2)
    ax_k1.axvline(x=k_value, color='red', linestyle='--', linewidth=2, label=f'K saat ini = {k_value}')
    
    # Tandai K optimal
    optimal_k_idx = np.argmax(test_scores)
    optimal_k = k_values_analysis[optimal_k_idx]
    ax_k1.axvline(x=optimal_k, color='green', linestyle='-.', linewidth=2, label=f'K optimal = {optimal_k}')
    
    ax_k1.set_xlabel('Nilai K (Jumlah Tetangga)', fontsize=12)
    ax_k1.set_ylabel('R² Score', fontsize=12)
    ax_k1.set_title('Performa Model terhadap Nilai K', fontsize=14)
    ax_k1.legend(loc='best')
    ax_k1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_k1)
    
    # Tampilkan statistik
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 K Optimal", f"{optimal_k}")
    with col2:
        st.metric("📈 R² Terbaik Testing", f"{max(test_scores):.4f}")
    with col3:
        st.metric("📉 Selisih Train-Test", f"{max(train_scores) - max(test_scores):.4f}")

with tab2:
    st.markdown("### 📊 Error Metrics vs Nilai K")
    
    fig_k2, ax_k2 = plt.subplots(figsize=(12, 6))
    ax_k2.plot(k_values_analysis, train_rmse, label='Training RMSE', marker='o', markersize=4, linewidth=2)
    ax_k2.plot(k_values_analysis, test_rmse, label='Testing RMSE', marker='s', markersize=4, linewidth=2)
    ax_k2.axvline(x=k_value, color='red', linestyle='--', linewidth=2, label=f'K saat ini = {k_value}')
    ax_k2.axvline(x=optimal_k, color='green', linestyle='-.', linewidth=2, label=f'K optimal = {optimal_k}')
    ax_k2.set_xlabel('Nilai K (Jumlah Tetangga)', fontsize=12)
    ax_k2.set_ylabel('RMSE', fontsize=12)
    ax_k2.set_title('RMSE terhadap Nilai K', fontsize=14)
    ax_k2.legend(loc='best')
    ax_k2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_k2)
    
    # Plot MAE
    fig_k3, ax_k3 = plt.subplots(figsize=(12, 6))
    ax_k3.plot(k_values_analysis, train_mae, label='Training MAE', marker='o', markersize=4, linewidth=2)
    ax_k3.plot(k_values_analysis, test_mae, label='Testing MAE', marker='s', markersize=4, linewidth=2)
    ax_k3.axvline(x=k_value, color='red', linestyle='--', linewidth=2, label=f'K saat ini = {k_value}')
    ax_k3.axvline(x=optimal_k, color='green', linestyle='-.', linewidth=2, label=f'K optimal = {optimal_k}')
    ax_k3.set_xlabel('Nilai K (Jumlah Tetangga)', fontsize=12)
    ax_k3.set_ylabel('MAE', fontsize=12)
    ax_k3.set_title('MAE terhadap Nilai K', fontsize=14)
    ax_k3.legend(loc='best')
    ax_k3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_k3)

with tab3:
    st.markdown("### 🎯 Elbow Method - Menentukan K Optimal")
    
    # Hitung selisih R²
    test_scores_array = np.array(test_scores)
    differences = np.diff(test_scores_array)
    abs_differences = np.abs(differences)
    
    fig_k4, ax_k4 = plt.subplots(figsize=(12, 6))
    
    # Plot perubahan R²
    ax_k4.plot(k_values_analysis[1:], abs_differences, marker='o', markersize=4, linewidth=2, color='purple')
    ax_k4.axvline(x=optimal_k, color='green', linestyle='-.', linewidth=2, label=f'K optimal = {optimal_k}')
    ax_k4.set_xlabel('Nilai K', fontsize=12)
    ax_k4.set_ylabel('Perubahan Absolut R²', fontsize=12)
    ax_k4.set_title('Elbow Method - Perubahan R² terhadap Nilai K', fontsize=14)
    ax_k4.legend(loc='best')
    ax_k4.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_k4)
    
    # Tampilkan 5 K terbaik
    st.markdown("#### 🏆 5 Nilai K Terbaik")
    top_k = sorted([(k, test_scores[i]) for i, k in enumerate(k_values_analysis)], 
                   key=lambda x: x[1], reverse=True)[:5]
    
    top_k_df = pd.DataFrame(top_k, columns=['Nilai K', 'R² Score'])
    top_k_df['R² Score'] = top_k_df['R² Score'].apply(lambda x: f"{x:.4f}")
    st.dataframe(top_k_df, use_container_width=True)

with tab4:
    st.markdown("### 📋 Rekomendasi Nilai K")
    
    # Analisis rekomendasi
    best_k = optimal_k
    best_r2 = max(test_scores)
    best_rmse = test_rmse[optimal_k_idx]
    best_mae = test_mae[optimal_k_idx]
    
    # Cari K dengan selisih train-test terkecil (stabil)
    diff_train_test = np.array(train_scores) - np.array(test_scores)
    stable_k_idx = np.argmin(np.abs(diff_train_test))
    stable_k = k_values_analysis[stable_k_idx]
    
    # Cari K dengan error terkecil
    min_error_k_idx = np.argmin(test_rmse)
    min_error_k = k_values_analysis[min_error_k_idx]
    
    # Tampilkan rekomendasi
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745;">
            <h4 style="color: #155724; margin-top: 0;">🎯 K Optimal (R²)</h4>
            <p style="font-size: 2rem; font-weight: bold; margin: 0; color: #155724;">K = {}</p>
            <p style="margin: 5px 0; color: #155724;">R²: {:.4f}</p>
            <p style="margin: 5px 0; color: #155724;">RMSE: {:,.0f}</p>
            <p style="margin: 5px 0; color: #155724;">MAE: {:,.0f}</p>
        </div>
        """.format(best_k, best_r2, best_rmse, best_mae), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #cce5ff; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff;">
            <h4 style="color: #004085; margin-top: 0;">⚖️ K Paling Stabil</h4>
            <p style="font-size: 2rem; font-weight: bold; margin: 0; color: #004085;">K = {}</p>
            <p style="margin: 5px 0; color: #004085;">Selisih R²: {:.4f}</p>
            <p style="margin: 5px 0; color: #004085;">Train R²: {:.4f}</p>
            <p style="margin: 5px 0; color: #004085;">Test R²: {:.4f}</p>
        </div>
        """.format(stable_k, diff_train_test[stable_k_idx], 
                   train_scores[stable_k_idx], test_scores[stable_k_idx]), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107;">
            <h4 style="color: #856404; margin-top: 0;">📉 K Error Terkecil</h4>
            <p style="font-size: 2rem; font-weight: bold; margin: 0; color: #856404;">K = {}</p>
            <p style="margin: 5px 0; color: #856404;">RMSE: {:,.0f}</p>
            <p style="margin: 5px 0; color: #856404;">MAE: {:,.0f}</p>
            <p style="margin: 5px 0; color: #856404;">R²: {:.4f}</p>
        </div>
        """.format(min_error_k, test_rmse[min_error_k_idx], 
                   test_mae[min_error_k_idx], test_scores[min_error_k_idx]), unsafe_allow_html=True)
    
    # Kesimpulan
    st.markdown("---")
    st.markdown("### 💡 Kesimpulan dan Saran")
    
    if best_k == min_error_k and best_k == stable_k:
        st.success(f"""
        ✅ **K = {best_k}** adalah nilai K yang optimal karena:
        - Memberikan R² tertinggi pada data testing
        - Memberikan error (RMSE/MAE) terendah
        - Memiliki performa yang stabil antara training dan testing
        
        **Rekomendasi:** Gunakan K = {best_k} untuk prediksi.
        """)
    else:
        st.info(f"""
        📊 **Analisis Multi-Kriteria:**
        
        | Kriteria | Nilai K | Performa |
        |----------|---------|----------|
        | R² Tertinggi | {best_k} | R² = {best_r2:.4f} |
        | Error Terkecil | {min_error_k} | RMSE = {test_rmse[min_error_k_idx]:,.0f} |
        | Paling Stabil | {stable_k} | Selisih R² = {diff_train_test[stable_k_idx]:.4f} |
        
        **Rekomendasi:** 
        - Untuk akurasi tertinggi: Gunakan **K = {best_k}**
        - Untuk hasil stabil: Gunakan **K = {stable_k}**
        - Untuk error terkecil: Gunakan **K = {min_error_k}**
        """)
    
    # Tabel lengkap performa
    with st.expander("📊 Lihat Tabel Lengkap Performa"):
        performance_df = pd.DataFrame({
            'K': k_values_analysis,
            'Train R²': train_scores,
            'Test R²': test_scores,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Train MAE': train_mae,
            'Test MAE': test_mae
        })
        st.dataframe(performance_df, use_container_width=True)
